"""Quadcopter dynamics, section II-D of the SkyDreamer paper.

NED throughout: +z is down, so gravity is `+g` on the z axis and thrust is
negative body-z.

The model is the one from Ferede et al. "One Net to Rule Them All"
(arXiv:2504.21586, see tudelft/optimal_quad_control_RL) extended by SkyDreamer
with quadratic drag, angle-of-attack / advance-ratio thrust corrections,
gyroscopic coupling, quaternion attitude, and additive disturbances.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .params import GRAVITY, PROP_RADIUS, DynParams

# Section II-D: RK4 with a 2.2 ms timestep.  Section II-C: the policy runs at
# f_c = 90 Hz.  5 * 2.2 ms = 11.0 ms, which is 90.9 Hz -- the paper's two
# numbers are inconsistent at the 1% level.  See docs/paper_gaps.md.
# The paper prints "a timestep of 2.2 ms" and a 90 Hz control frequency, and
# those disagree: 90 Hz over five substeps is 2.2222 ms, so the printed figure
# is rounded.  Taking 2.2 literally runs the simulator at 90.909 Hz -- 1% fast,
# 172 ms of drift over a 17 s flight -- and leaves it inconsistent with the
# rate penalty, which divides by f_c = 90 Hz.  Derive it from 90 Hz instead.
CONTROL_FREQ = 90.0
SUBSTEPS_PER_STEP = 5
DT_INNER = 1.0 / (CONTROL_FREQ * SUBSTEPS_PER_STEP)
SUBSTEPS = SUBSTEPS_PER_STEP
CONTROL_DT = DT_INNER * SUBSTEPS

# How to read the paper's horizontal advance ratio
#   mu = atan((v_x^B2 + v_y^B2) / (r * w_bar))
# "norm"    -> atan(|v_xy| / (r w_bar)).  Default: dimensionally consistent with
#              the angle-of-attack term right above it, gives ~44% translational
#              lift at 20 m/s.
# "literal" -> atan((v_x^2 + v_y^2) / (r w_bar)).  What the paper literally
#              prints; produces a ~6x thrust gain at 20 m/s, i.e. unflyable.
# Flip this once real flight data is available to identify the term properly.
ADVANCE_RATIO_MODE = "norm"


class State(NamedTuple):
    """Rigid-body state.  Leading dims are free (batch over envs)."""

    p: jax.Array  # (..., 3) position in world frame, NED
    v: jax.Array  # (..., 3) velocity in world frame
    q: jax.Array  # (..., 4) attitude quaternion [w, x, y, z], body->world
    omega_b: jax.Array  # (..., 3) body rates [p, q, r]
    motor: jax.Array  # (..., 4) propeller speeds, rad/s

    @property
    def dim(self) -> int:
        return 17


class Disturbance(NamedTuple):
    eps_a: jax.Array  # (..., 3) m/s^2, world-frame acceleration disturbance
    eps_M: jax.Array  # (..., 3) rad/s^2, angular acceleration disturbance
    eps_u: jax.Array  # (..., 4) additive motor command disturbance


def quat_to_matrix(q: jax.Array) -> jax.Array:
    """Body->world rotation matrix from a [w, x, y, z] quaternion."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def euler_to_quat(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    """psi -> theta -> phi intrinsic sequence, matching the paper's `lambda`."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    return jnp.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def quat_to_euler(q: jax.Array) -> jax.Array:
    """Returns (..., 3) of [roll, pitch, yaw]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jnp.stack([roll, pitch, yaw], axis=-1)


def motor_setpoint(u: jax.Array, eps_u: jax.Array, p: DynParams) -> jax.Array:
    """Steady-state motor response `w_ci` to a normalized command `u_i`."""
    u_tilde = jnp.clip(u + eps_u, 0.0, 1.0)
    k = p.k[..., None]
    span = (p.w_max - p.w_min)[..., None]
    root = jnp.sqrt(jnp.maximum(k * u_tilde**2 + (1.0 - k) * u_tilde, 0.0))
    return span * root + p.w_min[..., None]


def _forces_moments(s: State, u: jax.Array, dist: Disturbance, p: DynParams):
    R = quat_to_matrix(s.q)
    v_b = jnp.einsum("...ji,...j->...i", R, s.v)  # R^T @ v_world
    vbx, vby, vbz = v_b[..., 0], v_b[..., 1], v_b[..., 2]

    w = s.motor
    w_sum = jnp.sum(w, axis=-1)
    w_sq_sum = jnp.sum(w**2, axis=-1)

    # Angle of attack and horizontal advance ratio.  `r * w_bar` with
    # w_bar = sum(w_i) is the paper's normalizer; `r` (undefined in the paper)
    # is the propeller radius.
    denom = jnp.maximum(PROP_RADIUS * w_sum, 1e-6)
    alpha = jnp.arctan(vbz / denom)
    # PAPER-AMBIGUITY: see ADVANCE_RATIO_MODE above and docs/paper_gaps.md.
    v_h2 = vbx**2 + vby**2
    v_h = v_h2 if ADVANCE_RATIO_MODE == "literal" else jnp.sqrt(v_h2)
    mu = jnp.arctan(v_h / denom)

    thrust = -p.k_w * (1.0 + p.k_angle * alpha + p.k_hor * mu) * w_sq_sum
    F = jnp.stack(
        [
            -p.k_x * vbx * w_sum - p.k_x2 * vbx * jnp.abs(vbx),
            -p.k_y * vby * w_sum - p.k_y2 * vby * jnp.abs(vby),
            thrust - p.k_v2 * vbz * jnp.abs(vbz),
        ],
        axis=-1,
    )

    w_c = motor_setpoint(u, dist.eps_u, p)
    dw = (w_c - w) / p.tau[..., None]

    w1, w2, w3, w4 = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    d1, d2, d3, d4 = dw[..., 0], dw[..., 1], dw[..., 2], dw[..., 3]
    pr, qr, rr = s.omega_b[..., 0], s.omega_b[..., 1], s.omega_b[..., 2]

    M = jnp.stack(
        [
            -p.k_p1 * w1**2 - p.k_p2 * w2**2 + p.k_p3 * w3**2 + p.k_p4 * w4**2
            + p.J_x * qr * rr,
            -p.k_q1 * w1**2 + p.k_q2 * w2**2 - p.k_q3 * w3**2 + p.k_q4 * w4**2
            + p.J_y * pr * rr,
            -p.k_r1 * w1 + p.k_r2 * w2 + p.k_r3 * w3 - p.k_r4 * w4
            - p.k_r5 * d1 + p.k_r6 * d2 + p.k_r7 * d3 - p.k_r8 * d4
            + p.J_z * pr * qr,
        ],
        axis=-1,
    )
    return F, M, dw, R


def derivative(s: State, u: jax.Array, dist: Disturbance, p: DynParams) -> State:
    F, M, dw, R = _forces_moments(s, u, dist, p)

    g_vec = jnp.zeros_like(s.v).at[..., 2].set(GRAVITY)
    dv = g_vec + jnp.einsum("...ij,...j->...i", R, F) + dist.eps_a

    omega_quat = jnp.concatenate([jnp.zeros_like(s.omega_b[..., :1]), s.omega_b], -1)
    dq = 0.5 * quat_mul(s.q, omega_quat)

    return State(p=s.v, v=dv, q=dq, omega_b=M + dist.eps_M, motor=dw)


def _axpy(s: State, ds: State, a: float) -> State:
    return jax.tree.map(lambda x, d: x + a * d, s, ds)


def rk4_step(s: State, u: jax.Array, dist: Disturbance, p: DynParams, dt: float) -> State:
    k1 = derivative(s, u, dist, p)
    k2 = derivative(_axpy(s, k1, dt / 2), u, dist, p)
    k3 = derivative(_axpy(s, k2, dt / 2), u, dist, p)
    k4 = derivative(_axpy(s, k3, dt), u, dist, p)
    out = jax.tree.map(
        lambda x, a, b, c, d: x + dt / 6 * (a + 2 * b + 2 * c + d), s, k1, k2, k3, k4
    )
    q = out.q / jnp.linalg.norm(out.q, axis=-1, keepdims=True)
    return out._replace(q=q)


def step(
    s: State,
    u: jax.Array,
    dist: Disturbance,
    p: DynParams,
    substeps: int = SUBSTEPS,
    dt: float = DT_INNER,
) -> State:
    """Advance one control period by `substeps` RK4 steps, zero-order hold on u."""

    def body(carry, _):
        return rk4_step(carry, u, dist, p, dt), None

    out, _ = jax.lax.scan(body, s, None, length=substeps)
    return out


def initial_motor_speeds(frac: jax.Array, p: DynParams) -> jax.Array:
    """Table III initializes `w_i` as a fraction of `w_max`."""
    return frac * p.w_max[..., None]
