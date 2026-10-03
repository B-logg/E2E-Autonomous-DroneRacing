"""A hand-written controller that flies the reference lap, so that successful
laps exist to learn from.

NOT FROM THE PAPER.  See docs/deviations.md for why this is here: the
end-to-end agent converged to a value-accurate local optimum in which the
episode always ends after ~1.2 s, and under that belief flying faster is
strictly better, so nothing ever produced a completed lap to learn from.  This
module produces those laps.  It is a demonstration source, not a deliverable --
the policy that ships is still the DreamerV3 one driving four motors.

Structure is a textbook cascade:

    reference  ->  a_des  ->  (thrust magnitude, desired attitude)
                           ->  desired body rates  ->  desired angular
                           accelerations  ->  four motor commands

The only part specific to this simulator is the last step.  `dynamics.py`
integrates `domega/dt = M` directly, so `M` is an angular *acceleration* and
the Table II coefficients already fold in 1/J; and thrust and the roll/pitch
moments are linear in `w^2` while the yaw moment is linear in `w`.  That mix is
what `allocate` solves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple

from .dynamics import quat_to_matrix
from .params import PROP_RADIUS

GRAVITY_VEC = jnp.array([0.0, 0.0, 9.81])  # NED: +z is down


# --------------------------------------------------------------------------
# Control allocation:  (thrust magnitude, angular acceleration) -> w -> u
# --------------------------------------------------------------------------
def _thrust_gain(s, p):
    """`G` in `|thrust| = G * sum(w^2)`, including the angle-of-attack and
    advance-ratio corrections at the current state."""
    R = quat_to_matrix(s.q)
    v_b = jnp.einsum("...ji,...j->...i", R, s.v)
    w_sum = jnp.sum(s.motor, -1)
    denom = jnp.maximum(PROP_RADIUS * w_sum, 1e-6)
    alpha = jnp.arctan(v_b[..., 2] / denom)
    mu = jnp.arctan(jnp.sqrt(v_b[..., 0] ** 2 + v_b[..., 1] ** 2) / denom)
    return p.k_w * (1.0 + p.k_angle * alpha + p.k_hor * mu)


def _nullspace4(A):
    """The 1-D nullspace of a 3x4 matrix, as the generalised cross product:
    component i is the signed 3x3 minor with column i removed.  Closed form, so
    it differentiates and vmaps where an SVD would not."""
    cols = [jnp.delete(A, i, axis=1) for i in range(4)]
    return jnp.stack([(-1.0) ** i * jnp.linalg.det(c) for i, c in enumerate(cols)])


def allocate(s, p, thrust, M_des, iters: int = 6):
    """Motor speeds `w` that produce `thrust` (magnitude, m/s^2) and angular
    acceleration `M_des` (3,), at the current state.

    Thrust and the roll/pitch moments are linear in `x = w^2`:

        G sum(x)                                = thrust
        -kp1 x1 - kp2 x2 + kp3 x3 + kp4 x4      = Mx - Jx q r
        -kq1 x1 + kq2 x2 - kq3 x3 + kq4 x4      = My - Jy p r

    which is three equations in four unknowns.  The yaw moment is linear in
    `w = sqrt(x)` instead, so it cannot join that system; it fixes the one
    remaining degree of freedom.  Solve the linear part for its minimum-norm
    particular solution, then Newton on the single scalar that slides along the
    nullspace until yaw is right.

    The transient yaw term (`k_r5..k_r8`, proportional to `dw`) is included,
    which is not obvious -- it looks like authority that exists only while the
    motors are changing speed, and a steady-state inverse has no business
    using it.  Leaving it out is worse.  It is not small: at nominal
    coefficients a +-500 rad/s differential command produces 21.6 rad/s^2 of
    yaw through `dw`, against 18.6 rad/s^2 of *total* steady-state yaw
    authority, so every roll or pitch correction injects more yaw than the yaw
    axis can produce on purpose.  Treated as a disturbance it is one the rate
    loop cannot reject, because it arrives through the same command.  And it
    stays linear in the unknown: `dw = (w - motor)/tau` once `w` is what we are
    solving for, so it folds into the same scalar search.
    """
    pr, qr, rr = s.omega_b[..., 0], s.omega_b[..., 1], s.omega_b[..., 2]
    G = _thrust_gain(s, p)

    A = jnp.stack([
        jnp.stack([G, G, G, G]),
        jnp.stack([-p.k_p1, -p.k_p2, p.k_p3, p.k_p4]),
        jnp.stack([-p.k_q1, p.k_q2, -p.k_q3, p.k_q4]),
    ])
    b = jnp.stack([
        thrust,
        M_des[0] - p.J_x * qr * rr,
        M_des[1] - p.J_y * pr * rr,
    ])

    # Particular solution, as the smallest *change* from the motors' current
    # speeds rather than the smallest vector outright: it keeps the answer near
    # the physical operating point, starts the yaw search at t = 0, and avoids
    # proposing a motor speed the drone is nowhere near.
    #
    # The rows are normalised first.  They are not comparable as written --
    # thrust carries `G ~ 1.6e-6` while the roll row carries the `k_p`, so
    # `A A^T` has entries around 1e-11 and the 1e-12 ridge that looks
    # negligible is a 10% perturbation.  That alone left `A x_p - b` at -1.6
    # m/s^2 of thrust, which is 0.17 g of steady-state error.
    scale = jnp.maximum(jnp.linalg.norm(A, axis=1), 1e-30)
    An, bn = A / scale[:, None], b / scale
    x0 = s.motor**2
    x_p = x0 + An.T @ jnp.linalg.solve(
        An @ An.T + 1e-9 * jnp.eye(3), bn - An @ x0)
    nz = _nullspace4(An)
    nz = nz / jnp.maximum(jnp.linalg.norm(nz), 1e-12)

    lo, hi = p.w_min**2, p.w_max**2
    kr = jnp.stack([-p.k_r1, p.k_r2, p.k_r3, -p.k_r4])
    kr_d = jnp.stack([-p.k_r5, p.k_r6, p.k_r7, -p.k_r8]) / p.tau
    yaw_target = M_des[2] - p.J_z * pr * qr

    def yaw_of(t):
        w = jnp.sqrt(jnp.clip(x_p + t * nz, lo, hi))
        return jnp.sum(kr * w) + jnp.sum(kr_d * (w - s.motor))

    def body(t, _):
        # Central difference rather than a derivative through the clip, whose
        # gradient is zero exactly where the search needs to back out of a
        # saturated motor.
        h = 1e3
        d = (yaw_of(t + h) - yaw_of(t - h)) / (2 * h)
        step = (yaw_target - yaw_of(t)) / jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
        return t + jnp.clip(step, -1e6, 1e6), None

    t, _ = jax.lax.scan(body, 0.0, None, length=iters)
    return jnp.sqrt(jnp.clip(x_p + t * nz, lo, hi))


def motor_command(w_des, p):
    """Invert `motor_setpoint`: the normalized command `u` whose steady state
    is `w_des`.

    `w_c = (w_max - w_min) sqrt(k u^2 + (1-k) u) + w_min`, so with
    `y = ((w_c - w_min)/span)^2` we need `k u^2 + (1-k) u = y`.  Written as
    `2y / ((1-k) + sqrt((1-k)^2 + 4ky))` rather than the usual quadratic
    formula, which divides by `k` and blows up as `k -> 0`.
    """
    span = jnp.maximum(p.w_max - p.w_min, 1e-9)
    y = jnp.clip((w_des - p.w_min[..., None]) / span[..., None], 0.0, 1.0) ** 2
    k = p.k[..., None]
    root = jnp.sqrt(jnp.maximum((1.0 - k) ** 2 + 4.0 * k * y, 0.0))
    return jnp.clip(2.0 * y / jnp.maximum((1.0 - k) + root, 1e-9), 0.0, 1.0)


# --------------------------------------------------------------------------
# Cascade tracking controller
# --------------------------------------------------------------------------
class Gains(NamedTuple):
    """Loop bandwidths, outermost first, in rad/s.

    The separation is forced by the plant, not by taste.  The motors are a 30 ms
    first-order lag, so `1/tau = 33 rad/s` is where they alone contribute 45 deg
    of phase; the rate loop has to stay well under half of that, and each outer
    loop under a third of the one inside it.  Ignoring this is not a tuning
    detail -- at (9, 5, 12, 22) the drone answered a 1.4 m position error by
    rolling past 115 deg in 120 ms and never recovered.

    Yaw is split out because the authority is wildly asymmetric: 821 rad/s^2 in
    roll and 479 in pitch, but only 18.6 in yaw, which comes from the rotor
    drag coefficients rather than from thrust differences.  A yaw gain sized
    like the roll gain only saturates.
    """

    k_pos: float = 2.5                             # position error -> velocity, 1/s
    v_corr_max: float = 3.0                        # cap on that correction, m/s
    k_vel: float = 6.0                             # velocity error -> accel, 1/s
    a_fb_max: float = 25.0                         # cap on the feedback accel, m/s^2
    k_att: tuple = (5.0, 5.0, 2.0)                 # attitude error -> body rate, 1/s
    k_rate: tuple = (16.0, 16.0, 6.0)              # rate error -> angular accel, 1/s
    max_tilt_deg: float = 35.0                     # cap on the commanded tilt
    a_thrust_min: float = 3.0                      # never command less than this, m/s^2
    lookahead_s: float = 0.06
    search_back: int = 8    # path samples to allow going backwards
    search_fwd: int = 90    # and forwards, when re-locating progress


def _vee(S):
    return jnp.stack([S[2, 1], S[0, 2], S[1, 0]])


def _drag_accel(s, p):
    """Body-frame aerodynamic acceleration, everything in `F` except thrust.
    Known exactly here, so feed it forward rather than let the position loop
    discover it -- at 12 m/s it is already 1.2 g."""
    R = quat_to_matrix(s.q)
    v_b = jnp.einsum("...ji,...j->...i", R, s.v)
    vbx, vby, vbz = v_b[0], v_b[1], v_b[2]
    w_sum = jnp.sum(s.motor, -1)
    return jnp.stack([
        -p.k_x * vbx * w_sum - p.k_x2 * vbx * jnp.abs(vbx),
        -p.k_y * vby * w_sum - p.k_y2 * vby * jnp.abs(vby),
        -p.k_v2 * vbz * jnp.abs(vbz),
    ])


# --------------------------------------------------------------------------
# Path tracking
# --------------------------------------------------------------------------
# The controller follows the continuous reference path from `expert_path.py`.
# Two discrete-waypoint schemes were tried first and both failed the same way:
# every geometric corner of the track produced a new special case.  Pursuing
# three fixed points per gate never converged onto the gate axis, and arrived
# at the scoring plane 1.3-2.0 m off against a half-width of 0.8 m.  Switching
# to the gate's axis fixed that and broke the lap instead -- where the lap
# closes the drone leaves gate 2 already on the far side of gate 0's axis, so
# the target index ran a whole lap ahead.  Arming it like the environment does
# fixed that and left a third case: just after gate 0 the drone sits 5 m to the
# side of gate 1 but barely behind its plane, and a carrot advanced by a fixed
# lookahead jumps to the far side of the gate.
#
# A continuous path has none of these, because there is no switch.  The drone
# tracks the nearest point and a carrot a fixed distance further along, and the
# path itself already passes through every gate on its axis in the right order.
#
# What the path does NOT supply is the attitude: `expert_path.attitude_reference`
# is unusable here (its singularities are documented there and in
# docs/deviations.md), and it is not needed -- the cascade derives attitude from
# the acceleration it wants, and at these speeds the attitude loop keeps up.


class TrackCarry(NamedTuple):
    idx: jax.Array       # int32, nearest path sample last step
    z_prev: jax.Array    # (3,) previous commanded body-z, for the rate feedforward


def track_init(ref, p_w) -> TrackCarry:
    return TrackCarry(idx=jnp.argmin(jnp.linalg.norm(ref["p"] - p_w, axis=1)).astype(jnp.int32),
                      z_prev=jnp.array([0.0, 0.0, 1.0]))


def control(carry: TrackCarry, s, p, ref: dict, v_nom: float = 2.0,
            lookahead: float = 0.8, alt_floor: float = 1.8,
            g: Gains = Gains(), search_back: int = 10, search_fwd: int = 140,
            dt: float = 1.0 / 90.0):
    """One step of path tracking.  Returns `(carry, u)`."""
    P, TAN, V = ref["p"], ref["tangent"], ref["v"]
    n = P.shape[0]

    # Progress: nearest sample in a forward window, not a global argmin.  The
    # inverted loop passes 2.7 m directly above itself and a global lookup
    # there can jump a segment and command a reversal.
    cand = (carry.idx + jnp.arange(-search_back, search_fwd)) % n
    i = cand[jnp.argmin(jnp.linalg.norm(P[cand] - s.p, axis=1))]
    j = (i + jnp.int32(lookahead / ref["ds"])) % n

    d = P[j] - s.p
    speed = jnp.minimum(v_nom, V[j])
    v_cmd = speed * d / jnp.maximum(jnp.linalg.norm(d), 1e-6)
    # Floor on altitude.  NED, so up is -z.  Gates sit at 1.35 m and the ground
    # check fires below 0.5 m.
    v_cmd = v_cmd.at[2].add(-jnp.maximum(alt_floor + s.p[2], 0.0) * 2.0)

    a_des = g.k_vel * (v_cmd - s.v)
    R = quat_to_matrix(s.q)
    a_thrust = a_des - GRAVITY_VEC - R @ _drag_accel(s, p)

    # Limit the tilt, not the magnitude: capping the norm lets the horizontal
    # term eat the vertical one, and the vertical term is what holds the drone
    # up.  Never command less than `a_thrust_min` either -- a quadrotor asked
    # for zero thrust has no attitude, and a drone told to stop climbing wants
    # very nearly free fall, which sent the commanded tilt from 11 to 147 deg
    # inside half a second.
    up = jnp.minimum(a_thrust[2], -g.a_thrust_min)
    horiz = a_thrust[:2]
    lim = jnp.abs(up) * jnp.tan(jnp.deg2rad(g.max_tilt_deg))
    horiz = horiz * jnp.minimum(1.0, lim / jnp.maximum(jnp.linalg.norm(horiz), 1e-9))
    a_thrust = jnp.concatenate([horiz, up[None]])

    T = jnp.maximum(jnp.linalg.norm(a_thrust), 1e-3)
    z_B = -a_thrust / T

    # Heading along the path, which keeps the next gate in the camera -- the
    # policy's only view of the world is the gate mask, so a demonstration with
    # the gate out of frame teaches nothing.
    tan = TAN[j]
    h = jnp.array([tan[0], tan[1], 0.0])
    cur = jnp.array([R[0, 0], R[1, 0], 0.0])
    h = jnp.where(jnp.linalg.norm(h) > 0.3, h, cur)
    x_C = h / jnp.maximum(jnp.linalg.norm(h), 1e-6)

    y_B = jnp.cross(z_B, x_C)
    ny = jnp.linalg.norm(y_B)
    y_B = jnp.where(ny > 1e-3, y_B / jnp.maximum(ny, 1e-9), jnp.array([0.0, 1.0, 0.0]))
    R_des = jnp.stack([jnp.cross(y_B, z_B), y_B, z_B], axis=1)

    # Rate feedforward from how fast the commanded thrust axis is turning.
    w_ff = jnp.clip(R.T @ jnp.cross(carry.z_prev, (z_B - carry.z_prev) / dt), -8.0, 8.0)

    e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
    # No integral term on either loop.  It is the obvious answer to `eps_a` and
    # `eps_M_slow`, which hold for about 100 steps, and it loses: the rate loop
    # already sits near the phase margin the 30 ms motor lag leaves it, and an
    # integrator took the 5-lap rate from 5.5% to 0.8% at ki_w = 20.
    M_des = jnp.asarray(g.k_rate) * (-jnp.asarray(g.k_att) * e_R + w_ff - s.omega_b)

    return TrackCarry(idx=i.astype(jnp.int32), z_prev=z_B), motor_command(
        allocate(s, p, T, M_des), p)
