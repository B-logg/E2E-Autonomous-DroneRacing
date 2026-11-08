"""Track geometry, gate frames, flight plan logic and reward (paper II-B/C/H).

A track is a cyclic list of gates.  Each gate contributes three coaxial
"planes" the drone must cross in order:

    pre-gate (x = -t_g/2)  ->  gate (x = 0)  ->  post-gate (x = +t_g/2)

which is how the paper turns a zero-thickness gate into a collision volume of
thickness `t_g` with a shaped reward (section II-C).  Only the middle plane is
rendered; pre/post gates are virtual.  Gates flagged `visible=False` are the
paper's "additional invisible gates ... added to enforce the correct flight
maneuvers" -- they gate progress but are never drawn.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PLANES_PER_GATE = 3
N_FLIGHT_PLAN_GATES = 3  # paper: "for each of the next three gates"
FLIGHT_PLAN_DIM = 4 * N_FLIGHT_PLAN_GATES * 2  # 3 deltas + 3 absolutes, (xyz,yaw)


class Track(NamedTuple):
    """Static track description.  All arrays have leading dim `n_gates`."""

    pos: jax.Array  # (N, 3) gate centre in world frame, NED
    yaw: jax.Array  # (N,) gate normal heading; +x_g = intended direction of travel
    inner: jax.Array  # (N,) inner clear size (edge length), m
    outer: jax.Array  # (N,) outer size, m
    visible: jax.Array  # (N,) bool, whether the gate is rendered

    @property
    def n_gates(self) -> int:
        return self.pos.shape[0]


def make_track(gates, inner=1.5, outer=2.7) -> Track:
    """`gates` is a list of (x, y, z, yaw_deg) or (x, y, z, yaw_deg, visible).

    DreamerV3 sets `jax_transfer_guard='disallow'` process-wide
    (embodied/jax/internal.py), which rejects the host-to-device copies this
    one-time setup needs.  Allow them explicitly for this block rather than
    weakening the guard globally -- it would otherwise mask real per-step
    transfers in the training loop."""
    pos, yaw, vis = [], [], []
    for g in gates:
        pos.append(g[:3])
        yaw.append(np.deg2rad(g[3]))
        vis.append(bool(g[4]) if len(g) > 4 else True)
    n = len(gates)
    with jax.transfer_guard("allow"):
        return Track(
            pos=jnp.asarray(np.array(pos, np.float32)),
            yaw=jnp.asarray(np.array(yaw, np.float32)),
            inner=jnp.full((n,), float(inner)),
            outer=jnp.full((n,), float(outer)),
            visible=jnp.asarray(np.array(vis, bool)),
        )


def gate_rotation(yaw: jax.Array) -> jax.Array:
    """World->gate rotation.  Gate x = normal (travel direction), y = right in
    the gate plane, z = down (shared with world, gates are vertical)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    zero, one = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, s, zero], -1),
            jnp.stack([-s, c, zero], -1),
            jnp.stack([zero, zero, one], -1),
        ],
        axis=-2,
    )


def to_gate_frame(p_w: jax.Array, gate_pos: jax.Array, gate_yaw: jax.Array) -> jax.Array:
    """Express a world point in the frame of one gate."""
    R = gate_rotation(gate_yaw)
    return jnp.einsum("...ij,...j->...i", R, p_w - gate_pos)


def plane_pose(track: Track, plane_idx: jax.Array, t_g: float):
    """Centre and yaw of plane `plane_idx`, offset along the gate normal.

    `plane_idx` is monotone across laps, so it must wrap: without the modulo
    JAX silently clamps the out-of-range gather and every lap after the first
    would target the last gate forever."""
    gate = (plane_idx // PLANES_PER_GATE) % track.n_gates
    sub = plane_idx % PLANES_PER_GATE  # 0 = pre, 1 = gate, 2 = post
    yaw = track.yaw[gate]
    normal = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], -1)
    offset = (sub.astype(jnp.float32) - 1.0) * (t_g / 2.0)
    return track.pos[gate] + offset[..., None] * normal, yaw, gate


# --------------------------------------------------------------------------
# Reward (section II-C)
# --------------------------------------------------------------------------

RATE_CLIP = 17.0
CONTROL_FREQ = 90.0
W_PROG, W_GATE = 5.0, 30.0


def rate_penalty(omega_b: jax.Array, f_c: float = CONTROL_FREQ) -> jax.Array:
    """r_rate = (exp(min(||Omega||_1, 17)) - 1) / (2 f_c 1e5).

    Effectively zero below ~13 rad/s total and ~1.33 at the clip, so it only
    punishes rates that would saturate the gyro or smear the camera."""
    l1 = jnp.sum(jnp.abs(omega_b), axis=-1)
    return (jnp.exp(jnp.minimum(l1, RATE_CLIP)) - 1.0) / (2.0 * f_c * 1e5)


# Speed above the cap costs twice the progress it can buy (progress pays W_PROG
# per metre, i.e. W_PROG / f_c per m/s per step), so the reward peaks at the cap.
W_SPEED = 2.0 * W_PROG / CONTROL_FREQ
# Run 7c's: five times the progress.  Run 7b's 8M agent still fell out of the
# split-S at up to 8.5 m/s -- gravity's speed, which twice the progress did not
# make it brake for -- and 30% of its flights ended on the ground there.
W_SPEED_STRICT = 5.0 * W_PROG / CONTROL_FREQ


def speed_penalty(v: jax.Array, v_cap: float, w: float = W_SPEED) -> jax.Array:
    """`w` (W_SPEED) per m/s over `v_cap`, per step; 0 at or below it."""
    return w * jnp.maximum(jnp.linalg.norm(v, axis=-1) - v_cap, 0.0)


def gate_reward(p_g: jax.Array, d_g: float) -> jax.Array:
    """1 at the plane centre, falling linearly to 0 at the effective edge."""
    off = jnp.maximum(jnp.abs(p_g[..., 1]), jnp.abs(p_g[..., 2]))
    return 1.0 - off / d_g


def gate_collision(p_g: jax.Array, d_g: float) -> jax.Array:
    off = jnp.maximum(jnp.abs(p_g[..., 1]), jnp.abs(p_g[..., 2]))
    return off / d_g > 1.0


# NOT FROM THE PAPER.  The moment equation carries gyroscopic coupling
# (J_x q r etc.) that is quadratic in the body rates, and the model has no
# rotational damping at all, so a policy that spins the drone up drives a
# positive feedback loop: 100 rad/s already produces 8900 rad/s^2.  RK4 at
# 2.2 ms eventually diverges to NaN, which then poisons the replay buffer and
# kills the run.  The paper's termination conditions do not cover this.
#
# 50 rad/s is far outside anything physical: the rate penalty clips at
# ||Omega||_1 = 17, a real inverted loop at 13 m/s and 1.35 m radius is about
# 10 rad/s, and the flight controller's gyro saturates well below 50.  A drone
# there has already failed, so ending the episode (reward 0, like any other
# crash) is both safe and faithful.
RATE_DIVERGENCE = 50.0  # rad/s, per axis


def lowest_point(p_w: jax.Array, euler: jax.Array, reach: float) -> jax.Array:
    """Altitude of the airframe's lowest prop tip: the centre's altitude less
    `reach` (centre to prop tip) times the sine of the tilt."""
    cos_tilt = jnp.cos(euler[..., 0]) * jnp.cos(euler[..., 1])
    sin_tilt = jnp.sqrt(jnp.clip(1.0 - cos_tilt ** 2, 0.0, 1.0))
    return -p_w[..., 2] - reach * sin_tilt


def floor_collision(p_w: jax.Array, euler: jax.Array, reach: float, clearance: float) -> jax.Array:
    """The airframe's lowest point within `clearance` of the floor."""
    return lowest_point(p_w, euler, reach) < clearance


# NOT FROM THE PAPER.  Runs 7b-7d lost 14-28% of their flights on the floor
# between G3 and G4: run 7d's 2M crossed G3 still sinking at 2.7 m/s (the
# finishers at 2.0) and hit the floor 0.4 s later.  The airframe could have
# stopped: a TWR-2.4 airframe (5th percentile of training) at 35 deg tilt, full
# throttle after the action delay, stops 1.3 m/s within ~0.1 m.  What it lacks
# is the altitude -- no sensor measures it, and the world model's position is
# off by ~0.3 m -- and a crash pays nothing (II-C), only the lost future, which
# the world model, unsure of the altitude, does not foresee: over run 7d its
# continuation loss rose and its training score fell 134 -> 81 while it learned
# to fly lower.  Section II-C's ground test had been doing this job for 7b: it
# ended any flight descending below 0.5 m.
#
# The penalty is the sink speed over what can still be stopped above
# `FLOOR_BUFFER`: room = the lowest prop tip's height above the crash clearance
# less the buffer, a stopping distance v * FLOOR_LAG + v^2 / (2 FLOOR_DECEL), and
# W_FLOOR per m/s over per step.  Level or climbing flight costs nothing at any
# height, so it does not push the gates up; the takeoff climbs and costs nothing.
#
# Calibrated on run 7d's 2M flights (200, ground truth, after G2; docs/deviations.md):
# buffer and decel were swept for how much the finishers pay against the floor
# deaths.  At (0.5, 12) -- the physics alone, TWR 2.4 -- finishers paid 36.5 and
# the deaths 77.7, and 9 in 10 finishers were charged.  At (0.2, 20) the finishers
# pay 4.1, the deaths 34.1, every death pays more than 9 finishers in 10, and the
# charge starts 0.29 s before the impact (10th percentile 0.19 s) -- 26 steps of
# warning, against a stop that takes ~0.1 s.  0.2 m is also about one standard
# deviation of the world model's altitude (its 3-D error is 0.28 m).
FLOOR_LAG = 0.04      # s, action delay + motor spin-up, fitted to the simulator
FLOOR_DECEL = 20.0    # m/s^2; (TWR 3.4 - 1) g, a median airframe banked 35 deg
FLOOR_BUFFER = 0.2    # m
# Five times the progress a m/s buys (like W_SPEED_STRICT), so diving at G3,
# which is progress toward it, still loses.  7c's global 5x slowed the whole
# lap; this one is zero while level, climbing, or high enough to stop.  Run 7d's
# finishers would pay 1.1 a flight, its floor deaths 9.5 -- a third of a gate.
W_FLOOR = 5.0 * W_PROG / CONTROL_FREQ
# Run 7f's: twice that.  At run 7e's 3M (3 laps, erosion off) the floor crashes
# paid 9.0 a flight and the finishers 2.8 -- the charge does single out the crashes
# (3.2x), but a crash paid a third of one gate (30).  Doubled, a crash pays ~18 on
# top of the altitude term and a finisher ~1.9 a lap, under 1% of a lap's reward.
W_FLOOR_STRONG = 2.0 * W_FLOOR


def safe_descent(room: jax.Array, decel: float = FLOOR_DECEL, lag: float = FLOOR_LAG) -> jax.Array:
    """Fastest sink, m/s, that stops within `room` m: v * lag + v^2 / (2 decel) = room."""
    room = jnp.maximum(room, 0.0)
    return decel * (jnp.sqrt(lag ** 2 + 2.0 * room / decel) - lag)


def floor_penalty(p_w: jax.Array, v_w: jax.Array, euler: jax.Array, reach: float,
                  clearance: float, w: float = W_FLOOR, buffer: float = FLOOR_BUFFER) -> jax.Array:
    """`w` per m/s of sink over `safe_descent`, per step; 0 when level or climbing."""
    room = lowest_point(p_w, euler, reach) - clearance - buffer
    return w * jnp.maximum(v_w[..., 2] - safe_descent(room), 0.0)


def diverged(omega_b: jax.Array, p_w: jax.Array, v_w: jax.Array) -> jax.Array:
    """True once the integration has left the physically meaningful region."""
    spun = jnp.max(jnp.abs(omega_b), axis=-1) > RATE_DIVERGENCE
    finite = (
        jnp.all(jnp.isfinite(omega_b), axis=-1)
        & jnp.all(jnp.isfinite(p_w), axis=-1)
        & jnp.all(jnp.isfinite(v_w), axis=-1)
    )
    return spun | ~finite


def ground_collision(p_w: jax.Array, v_w: jax.Array, euler: jax.Array) -> jax.Array:
    """z_w > -0.5 and (v_z > 1.0 or |phi| > pi/3 or |theta| > pi/3).

    Shaped rather than a plain floor because the real drone launches from a
    raised podium and podium contact forces are not simulated."""
    low = p_w[..., 2] > -0.5
    bad = (
        (v_w[..., 2] > 1.0)
        | (jnp.abs(euler[..., 0]) > jnp.pi / 3)
        | (jnp.abs(euler[..., 1]) > jnp.pi / 3)
    )
    return low & bad


# NOT FROM THE PAPER.  Run 7e's evaluation fails a crossing when any point of the
# airframe passes the real gate's plane outside its opening (hardware.check_flight),
# but training scored and ended crossings on the centre of mass alone, inside
# d_g = 0.45 m: at 3M, 27-42 of each 200 one-lap flights that failed that way were
# clean by the training's own test, so nothing ever told the agent.  The airframe
# reaches 0.08-0.19 m (p10-p90) further out than its centre does at a crossing.
#
# Training sees one state per step, so each footprint point is carried along the
# centre's velocity to the plane.  Against check_flight on run 7e's 3M flights
# (1823 crossings, erosion off, ground truth): error p5..p95 -0.015..+0.009 m,
# pass/fail the same on 99.8% of crossings.
def swept_offset(p_w: jax.Array, v_w: jax.Array, R_wb: jax.Array, points: jax.Array,
                 gate_pos: jax.Array, gate_yaw: jax.Array) -> jax.Array:
    """Largest gate-frame max(|y|, |z|) of the airframe's `points` (body frame,
    (M, 3)) where they cross the gate's plane, the drone moving at `v_w`."""
    R_gw = gate_rotation(gate_yaw)
    loc = (p_w + points @ R_wb.T - gate_pos) @ R_gw.T          # (M, 3) gate frame
    v_g = R_gw @ v_w
    vx = jnp.maximum(v_g[0], 0.5)       # through the plane, never parallel to it
    yz = loc[:, 1:] - loc[:, :1] * (v_g[1:] / vx)
    return jnp.max(jnp.abs(yz))


# NOT FROM THE PAPER.  Between G3 and G4 run 7e flies low as a habit, not only
# when it crashes: at 3M (3 laps, erosion off) half the legs that survived took
# the lowest prop tip under 0.45 m and 16% under 0.25 m, where the ones that hit
# the floor went (p50 0.48 m at G3, then down).  The world model's position is off
# by 0.20-0.25 m, so a flight that low has no margin it can see.  floor_penalty
# charges a sink and nothing else; level flight at 0.2 m is free.
#
# So: `W_ALT` per metre of the lowest prop tip under `ALT_MARGIN`, per step, once
# the flight has been above it (a launch climbs through it for free).  0.45 m keeps
# the centre within G1/G3/G4's window (centre 0.75, CoM window +-0.45) and starts
# the charge 0.31 s (p50; p10 0.17 s) before a floor crash -- 28 steps, beyond the
# actor's 16-step imagination.  0.1 m under costs 0.3 a step, what progress pays
# at 5.4 m/s; run 7e's 3-lap finishers flew 9.0 m-steps under (27 a flight, ~9 a
# lap, against ~250 a lap of reward), its floor crashes 11.2.
ALT_MARGIN = 0.45   # m, lowest prop tip
W_ALT = 3.0         # per metre under, per step


def altitude_penalty(lowest: jax.Array, margin: float = ALT_MARGIN, w: float = W_ALT) -> jax.Array:
    """`w` per metre the lowest point is under `margin`, per step; 0 above it."""
    return w * jnp.maximum(margin - lowest, 0.0)


# NOT FROM THE PAPER.  Runs 7f and 7g lost 4-6% of their flights before they left
# the floor (run_compare.sh, 400 flights each, seed 20000): a prop tip touches once
# the airframe leans past asin(alt / reach), 22 deg at 7 cm.  The crashes lean no
# further than the takeoffs that clear (25 vs 27 deg, p50) -- they lean *while low*:
# they are the low-thrust airframes (TWR 2.8 against 4.0, p50; k_w the strongest
# separator, rank-biserial -0.53), which climb slower on the same lean schedule.
# 14 of the 400 flights crash under both checkpoints: the draw decides it, and the
# policy cannot tell the draw before it is off the floor.  altitude_penalty does
# not reach this: it charges only after the flight has been above its margin.
#
# So: `W_LAUNCH_TIP` per metre of the lowest prop tip under `LAUNCH_TIP_BUFFER`,
# per step, until the airframe is clear of the floor (env._floor_contact's
# `airborne`).  Standing on the floor the tip is at 0.05 m and costs nothing; a
# straight climb costs nothing.  At 2 cm, 7% of 7f's clean takeoffs pay anything,
# every crash pays: its tip spends 0.051 m-steps under (p50; p90 0.090), which at
# 600 costs ~31 -- one gate -- in the 3-5 steps before the touch.
LAUNCH_TIP_BUFFER = 0.02   # m, lowest prop tip, before the airframe is clear
W_LAUNCH_TIP = 600.0       # per metre under, per step


def launch_tip_penalty(lowest: jax.Array, buffer: float = LAUNCH_TIP_BUFFER,
                       w: float = W_LAUNCH_TIP) -> jax.Array:
    """`w` per metre the lowest point is under `buffer`, per step; 0 above it."""
    return w * jnp.maximum(buffer - lowest, 0.0)


# NOT FROM THE PAPER.  Run 7f fixed the floor with a cost that starts before the
# failure (altitude_penalty: ground deaths 30% -> 14% over 3 laps) but not the
# gate, where it only moved the reward's zero and the collision to the airframe:
# per real-gate crossing the airframe still clipped the opening 5.2% (3M) and 6.8%
# (4M) of the time against run 7e's 4.6%, G2 6.3% -> 13.8% (1 lap) as the lap got
# faster (5.04 -> 4.73 s).  At G2 the world model decodes +0.08 m of the true
# +0.13 m lateral offset (corr 0.86): the policy sees the line it takes.
#
# So the same shape at the gate: `W_GATE_MARGIN` per metre of the airframe's worst
# point beyond `GATE_MARGIN`, once, at the real plane.  200 makes a crossing at
# the 0.6 m opening cost -30 against the 0 it paid, one gate's worth.  On 7f's 3M
# flights 22% of crossings pass 0.45 m, by 0.023 m each on average: ~18 a lap
# against ~250 a lap of reward (the altitude term: ~9).  0.45 m is the airframe's
# worst point; its centre is ~0.15 m inside it (p50), so the charge starts with
# the centre ~0.30 m out.
GATE_MARGIN = 0.45      # m, airframe worst point at the real plane
W_GATE_MARGIN = 200.0   # reward per metre beyond, per crossing


def gate_margin_penalty(worst: jax.Array, margin: float = GATE_MARGIN,
                        w: float = W_GATE_MARGIN) -> jax.Array:
    """`w` per metre the airframe's worst point is beyond `margin`; 0 inside it."""
    return w * jnp.maximum(worst - margin, 0.0)


# NOT FROM THE PAPER.  Run 7f's 3M, 400 flights x 3 laps (seed 20000), every real
# crossing scored by hardware.check_flight: a crossing whose centre is within d_g
# (0.45 m) at both the pre- and the post-plane clipped the opening 1 time in 3,080;
# the rest, 17.8%.  The training test is the right one -- the policy fails it.
# Where: the pre-plane is broken by 4% of G1's crossings, 25% of G2's, 12% of G3's
# and G4's, and a crossing that broke it clips 18-49% of the time (0.5-2.5% if
# not).  The breaks hardly depend on the airframe drawn (|rank-biserial| <= 0.16).
# They follow II-C's progress term, which pays the straight-line distance to the
# pre-plane's centre: from G1's post-plane that line meets G2 31 deg off its normal
# and 1.5 m below (the flights: 29 deg p50, 0.47 m beside and 0.38 m below G2's
# axis a metre out), and G2 -> G3 is a vertical drop onto a vertical gate.  Every
# further training run from 7f's 3M (7f to 4M, 7g, 7h) flew G1 -> G2 faster (1.66
# -> 1.48-1.50 s) and lost precision there.
#
# So each gate gets an approach line: its normal, from `APPROACH[g]` m out to the
# pre-plane.  approach_distance is the distance still to fly by way of that line
# -- to its outer end, then along it -- and replaces the straight line in the
# progress term; at APPROACH = t_g / 2 it is II-C's.  It is continuous, so the
# progress it pays sums to start minus end like II-C's: it changes which way is
# "forward", not what a leg pays.  approach_reward is what does change it: crossing
# the line's outer end pays `W_APPROACH` times 1 - offset / d_g, clipped to
# [-1, 1] -- the gate reward's shape, without ending the flight.
# The lengths come from the same flights: G1's U-turn from G4 starts too close to
# fit more than its 0.4 (the turn already comes within 0.16 m of the wall), G3's
# split-S only reaches 0.8 m back on 93% of laps (75% at 1.0 m), G2 and G4 have
# room for 1.0 and 1.2 (no flight's centre went past x = 6.04 m or 3.25 m up).
APPROACH = {"team": (0.4, 1.0, 0.8, 1.2)}   # m from the gate plane, per gate
W_APPROACH = W_GATE / 2


def approach_distance(p_w: jax.Array, track, plane_idx: jax.Array, t_g: jax.Array,
                      approach: jax.Array) -> jax.Array:
    """Distance from `p_w` to the pre-plane of `plane_idx`'s gate by way of the
    gate's approach line, `approach[gate]` m out along its normal: to the line's
    outer end while further out than that, then straight to the line and along it."""
    gate = (plane_idx // PLANES_PER_GATE) % track.n_gates
    half = t_g / 2.0
    a = jnp.maximum(approach[gate], half)
    loc = gate_rotation(track.yaw[gate]) @ (p_w - track.pos[gate])
    s = jnp.clip(loc[0], -a, -half)
    return jnp.sqrt((loc[0] - s) ** 2 + loc[1] ** 2 + loc[2] ** 2) + (-half - s)


def approach_crossing(p_prev: jax.Array, p_next: jax.Array, track, plane_idx: jax.Array,
                      t_g: jax.Array, approach: jax.Array):
    """Whether the step p_prev -> p_next crossed the outer end of the approach line
    of `plane_idx`'s gate, forward, and the offset (max of |y|, |z|) there.  Never
    for a line no longer than the pre-plane's distance (t_g / 2)."""
    gate = (plane_idx // PLANES_PER_GATE) % track.n_gates
    a = approach[gate]
    R = gate_rotation(track.yaw[gate])
    l0 = R @ (p_prev - track.pos[gate])
    l1 = R @ (p_next - track.pos[gate])
    f = jnp.clip((-a - l0[0]) / jnp.where(l1[0] == l0[0], 1.0, l1[0] - l0[0]), 0.0, 1.0)
    yz = l0[1:] + f * (l1[1:] - l0[1:])
    off = jnp.max(jnp.abs(yz))
    crossed = (a > t_g / 2.0) & (l0[0] < -a) & (l1[0] >= -a)
    return crossed, off


def approach_reward(off: jax.Array, d_g: jax.Array, w: float = W_APPROACH) -> jax.Array:
    """`w` times the gate reward's 1 - off / d_g, clipped to [-1, 1]."""
    return w * jnp.clip(1.0 - off / d_g, -1.0, 1.0)


# --------------------------------------------------------------------------
# Flight plan vector (section II-H)
# --------------------------------------------------------------------------


def wrap_pi(a: jax.Array) -> jax.Array:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def flight_plan(track: Track, i: jax.Array) -> jax.Array:
    """24-D vector: for gates i-1..i+2, the three consecutive (dp, dpsi) deltas
    followed by the three absolute (p, psi) of gates i..i+2, all in world frame.

    Contains no drone state -- that is the whole point, it is track knowledge
    only, and it is what lets the policy disambiguate visually identical parts
    of the track."""
    n = track.n_gates
    idx = (jnp.arange(-1, N_FLIGHT_PLAN_GATES) + i) % n  # i-1, i, i+1, i+2
    p = track.pos[idx]
    y = track.yaw[idx]
    dp = p[1:] - p[:-1]
    dy = wrap_pi(y[1:] - y[:-1])
    deltas = jnp.concatenate([dp, dy[:, None]], axis=-1).reshape(-1)
    absol = jnp.concatenate([p[1:], y[1:, None]], axis=-1).reshape(-1)
    return jnp.concatenate([deltas, absol])


# --------------------------------------------------------------------------
# Reference tracks
# --------------------------------------------------------------------------
#
# PROVENANCE.  The paper never publishes gate coordinates.  What follows is a
# reconstruction, and each number is labelled with where it came from:
#
#   [TEXT]  stated numerically in the paper's prose -- authoritative.
#   [FIG]   measured off Figures 4 and 6, which are matplotlib plots with
#           metric axes whose caption reads "The black blocks mark the gate
#           locations with exaggerated thickness".
#   [OURS]  our choice; the paper constrains it only qualitatively.
#
# The [FIG] readings are trustworthy because the same pixel-to-metre
# calibration reproduces three quantities the text states independently:
#   gate outer size          text 2.7 m   measured 2.71 m
#   real-to-virtual gate gap text 2.7 m   measured 2.66-2.70 m
#   ladder flight area       text 6x4 m   measured ~6x4 m
# Residual uncertainty is about +-0.05 m, except gate 1 whose block is clipped
# by the top of the plot, leaving its centre uncertain to about +-0.2 m.
#
# See docs/paper_gaps.md B1/B2.  Replace all of this with surveyed coordinates
# once the real track exists -- nothing else depends on it.
#
# NED: z is NEGATIVE upward.  Gate yaw sets the normal, i.e. the intended
# direction of travel: yaw = 90 deg means the drone flies through it toward +Y.

# [FIG] Both small tracks share the same two real gates: 5.0 m apart along X,
# both in the plane Y = 0, both flown along Y.
GATE_1_X = 3.0       # [FIG] clipped block; centre is +2.8..+3.0
GATE_2_X = -2.0      # [FIG] full block, span 2.71 m, centre -2.01
GATE_Z = -1.35       # [FIG] a 2.7 m outer gate standing on the floor
BIG_Z = -1.25        # [FIG 9] gate blocks span altitude 0.17-2.34 m
LOOP_GAP = 2.7       # [TEXT] "the separation between the actual gate and the
                     # virtual gate of 2.7 m"; [FIG] confirms the loop's top
                     # crossing at altitude 4.0 m = 1.35 + 2.7

INVERTED_LOOP = [
    # Gate 1, real.  Flown toward +Y; Table III starts the drone 2-4 m before
    # it, which matches the slow (dark) start at Y ~ -3.2 in Figure 6.
    (GATE_1_X, 0.0, GATE_Z, 90.0),
    # [OURS] Virtual gate 2.7 m above gate 2, flown toward -Y: "SkyDreamer then
    # flies over the second gate".  The height is [TEXT]+[FIG]; that it is
    # directly overhead rather than offset is our reading of the side view.
    (GATE_2_X, 0.0, GATE_Z - LOOP_GAP, 270.0, False),
    # Gate 2, real.  The split-S: roll inverted over the gate and pull through
    # it, which reverses the direction of travel back to +Y.  Half a loop of
    # diameter 2.7 m has radius 1.35 m -- [TEXT] "an almost perfect circle with
    # a radius of roughly 1.5 m".
    (GATE_2_X, 0.0, GATE_Z, 90.0),
]

LADDER_INVERTED_LOOP = [
    # Same two gates; the difference is the ladder at gate 1.
    (GATE_1_X, 0.0, GATE_Z, 90.0),
    # [OURS] The ladder: "a full 360 degree left turn, and flies back over it",
    # with the drone "remaining mostly within 1 m of the gate" [TEXT].  Two
    # virtual gates beside and above gate 1 force the turn to stay tight.
    (GATE_1_X + 1.0, 1.0, GATE_Z - 1.2, 270.0, False),
    (GATE_1_X - 1.0, 0.0, GATE_Z - 0.6, 90.0, False),
    # ...then the same inverted loop at gate 2.
    (GATE_2_X, 0.0, GATE_Z - LOOP_GAP, 270.0, False),
    (GATE_2_X, 0.0, GATE_Z, 90.0),
]


# --------------------------------------------------------------------------
# Big track -- an out-of-distribution test, NOT a reconstruction
# --------------------------------------------------------------------------
#
# Measured off Figure 9, like the small tracks -- see the note above BIG_TRACK
# below for the calibration and its validation.  The paper's prose gives the
# sequence of maneuvers, which is what the gate *order* and the invisible gates
# follow:
#
#   "the flight begins in front of the top-left gate.  After the first gate,
#    the trajectory continues with a slight left turn, two subsequent gates, a
#    right turn, and a ladder maneuver in which the drone passes through the
#    lower gate, performs a full 360 degree left turn, and flies back over it.
#    From there, SkyDreamer extends the right turn, executing a steep dive into
#    the next gate, and continues through several more gates before performing
#    a tight braking maneuver to initiate a sharp right turn.  After the
#    following gate, the track concludes with a split-S maneuver over the first
#    gate."
#
# Gate size is deliberately left at the training value.  This is a test of
# whether the flight plan generalises to an unseen *layout*; changing the gates
# too would also change what the mask looks like, and a failure could no longer
# be attributed to either one.
#
# A policy trained on the 3-gate inverted loop has never seen headings other
# than +-Y, nor gate separations this large, so expect it to struggle.  That is
# the point: the paper claims the flight plan "potentially enables
# generalization to arbitrary tracks" and never demonstrates it.

# MEASURED, like the small tracks.  The arXiv HTML drops Figure 9's plots --
# they are vector, not raster -- but the PDF has them on page 14 with metric
# axes, so the gates can be read off exactly as for Figures 4 and 6.
#
# Validation, the same kind as before: the measured gate blocks come out
# 2.10 m wide and span altitude 0.17-2.34 m, against the MAVLab gate's stated
# outer size of 2.1 m.  Two independent confirmations that the calibration is
# right.
#
# What the measurement says: a large oval, 8 gates, roughly 23 x 13 m.
#     top straight    X = +6.0, gates at Y = -3.4 and +2.8
#     bottom straight X = -6.0, gates at Y = -3.4 and +2.8
#     left end        one gate at (X +3.7, Y -9.5) -- the start
#     right end       two gates tilted ~45 deg at Y ~ +8.5, where the ladder is
#     middle          one gate at (X 0, Y -0.3) turned 90 deg to the rest
#
# Earlier attempts in this file's history guessed at the layout from the
# caption and from the 3D render and got it wrong twice -- once as a scattered
# polygon, once as a slalom down a single row.  Measuring settled it.
BIG_TRACK = [
    # 1. start gate, at the left end, flown +Y ("the top-left gate")
    (3.7, -9.5, BIG_Z, 90.0),
    # 2-3. "a slight left turn, two subsequent gates" -- the top straight
    (6.0, -3.4, BIG_Z, 90.0),
    (6.0, 2.8, BIG_Z, 90.0),
    # 4-5. "a right turn, and a ladder maneuver in which the drone passes
    #      through the lower gate, performs a full 360 degree left turn, and
    #      flies back over it" -- the two tilted gates at the right end
    (2.6, 8.5, BIG_Z, 135.0),
    (5.5, 11.5, BIG_Z - 2.2, 250.0, False),     # up and around
    (0.5, 10.5, BIG_Z - 1.0, 120.0, False),     # back over it
    # 6. "a steep dive into the next gate"
    (-3.0, 8.9, BIG_Z, 215.0),
    # 7-8. "continues through several more gates" -- the bottom straight, -Y
    (-6.0, 2.8, BIG_Z, 270.0),
    (-6.0, -3.4, BIG_Z, 270.0),
    # 9. "a tight braking maneuver to initiate a sharp right turn.  After the
    #    following gate" -- the middle gate, turned 90 deg to the rest
    (-3.0, -7.5, BIG_Z, 330.0, False),
    (0.0, -0.3, BIG_Z, 0.0),
    # 10. "the track concludes with a split-S maneuver over the first gate"
    (3.7, -9.5, BIG_Z - LOOP_GAP, 270.0, False),
]

def big_track() -> Track:
    """Approximates the paper's Figure 9 layout from its prose, for zero-shot
    testing of a policy trained on the small track.  Same gates as training --
    only the layout is out of distribution."""
    return make_track(BIG_TRACK, inner=1.5, outer=2.7)


def inverted_loop() -> Track:
    """Figure 6 / Table IV "Loop (orange)".  Orange gates: inner 1.5 m, outer
    2.7 m [TEXT]."""
    return make_track(INVERTED_LOOP, inner=1.5, outer=2.7)


def ladder_inverted_loop() -> Track:
    """Figures 4 and 5 / Table IV "Ladder loop (orange)".  This is the track the
    paper's simulation results come from, and section III-B trains it with
    t_g = 0.3 m rather than Table III's 0.8 m."""
    return make_track(LADDER_INVERTED_LOOP, inner=1.5, outer=2.7)


# --------------------------------------------------------------------------
# Our track -- the room the real drone flies in
# --------------------------------------------------------------------------
#
# [MAP] The team's gate_map.yaml (2026-10-29), which the layout preview is drawn
# from: room 8 x 11 x 3.5 m, gate outer 1.5 m, opening 1.2 m.  Gate x/y/z is the
# centre, yaw the passage normal CCW from +X.  G1, G3 and G4 stand on the floor
# (centre 0.75 m); G2 stands on G3, flown the other way -- a 1.5 m-diameter split-S.
#
# The map is right-handed with Z up.  Converted to the simulator's NED: x = Y_map,
# y = X_map, z = -Z_map, yaw = 90 deg - yaw_map.  A plain z flip instead would
# mirror the track, and a policy trained on the mirror turns the wrong way.
TEAM_INNER, TEAM_OUTER = 1.2, 1.5
TEAM_TRACK = [
    # map:  (X,    Y,    Z,    yaw)
    (-1.0, -2.0, -0.75, 0.0),     # G1 (-2.0, -1.0, 0.75,  90)
    (4.0, 0.0, -2.25, 0.0),       # G2 ( 0.0,  4.0, 2.25,  90): on top of G3
    (4.0, 0.0, -0.75, 180.0),     # G3 ( 0.0,  4.0, 0.75, 270): back the other way
    (-1.0, 2.0, -0.75, 180.0),    # G4 ( 2.0, -1.0, 0.75, 270): then a U-turn to G1
]
# Walls and ceiling, NED: (x_lo, x_hi, y_lo, y_hi, ceiling altitude).  Map X -4..4,
# Y -4..7, Z 0..3.5.  The paper's tracks have no room, hence a separate table.
ROOMS = {"team": (-4.0, 7.0, -4.0, 4.0, 3.5)}
# Speed limit, m/s, on tracks that need one (not in the paper).  Run 7 learned to
# reach G2 at ~14 m/s and then hit the ceiling or the far wall: the split-S down
# to G3, 1.5 m below, has a ~0.75 m radius, which TWR ~4 can turn at ~4.8 m/s at
# most, and the room leaves 3 m past G2.  4.0 keeps a margin under that.
V_CAP = {"team": 4.0}
# Floor clearance, m: tracks whose crash test is the airframe reaching the floor
# (`floor_collision`), not section II-C's shaped z_w > -0.5 test (not in the paper).
# That test ends a flight 0.5 m up if it descends or banks hard -- for the paper's
# podium.  Our room's G1, G3 and G4 open 0.15-1.35 m, so it covered the bottom
# third of three openings, and run 7b/7c renders all ended at exactly 0.5 m.
# 0.10 m keeps a margin for ground effect, which the dynamics do not model.
FLOOR = {"team": 0.10}
# Where the real drone starts: on the floor, in front of G1, facing it.  Run 7e's
# evaluation column starts every flight there, not Table III's 0.7-1.3 m below the
# gate -- in our room that is under the floor, and the room clip put all of them at
# 0.3 m, falling, banked up to 20 deg: 14-25% of evaluation flights died there.
#   (distance lo, hi from G1's plane to the centre, m; lateral +-, m; yaw +-, deg;
#    centre altitude at rest, m (est.))
# The team can put it anywhere 1-2 m out.  2.0 m: the shallowest climb to G1's centre
# (19 deg, against 35 at 1 m), the most time for the world model to settle before
# the first gate (its first 20 steps are its worst, 0.36 m), and the only distance
# in that range the agent has flown G1 from -- every start it has had was 2.0-2.7 m
# out.  1.8-2.0 covers placing it a little short.  The climb itself is no limit:
# a TWR-2.4 airframe reaches 0.75 m in 0.37 s.
LAUNCH = {"team": (1.8, 2.0, 0.15, 5.0, 0.05)}
# Lowest mid-air start (the training column), m: G1, G3 and G4's centres are at
# 0.75, and Table III's 0-1.3 m below them is under the floor.  0.6 m keeps a start
# inside the bottom of the opening and clear of the floor penalty's reach.
AIR_START = {"team": 0.6}
# A launched drone that comes back down slower than this settles on the floor
# again; faster is a crash -- section II-C's v_z > 1 m/s.
TOUCHDOWN_SPEED = 1.0
# How thin the 50%-of-the-time mask erosion makes our frame, in pixels of the
# 64 x 64 image per side, on tracks whose frame is not the paper's (not in the
# paper).  Section II-F erodes 1 pixel at 64 x 64 because real masks came out
# thinner, on a frame 0.6 m wide (1.5 m opening, 2.7 m outer).  Ours is 0.15 m:
# 1-2 px beyond 2 m, and a 1-pixel erosion's 3x3 test removes anything under 3 px,
# so every gate further than ~1.25 m vanished -- a frame 2 m before G2 went from
# 110 pixels to 4.  Run 7e's 3M, scored with that erosion off: 45.0 -> 65.5% one lap,
# 6.0 -> 36.5% three laps (physical, 200 flights).  The same *fraction* of the frame
# as the paper's 1 pixel of 0.6 m is 0.25 px of 0.15 m, applied in the renderer
# (render.render_mask's `shrink_px`), where a quarter pixel can be resolved.
ERODE_PX = {"team": 0.25}


def team_track() -> Track:
    """Our 4-gate room track; real gates, no virtual ones."""
    return make_track(TEAM_TRACK, inner=TEAM_INNER, outer=TEAM_OUTER)


def left_room(p_w: jax.Array, room) -> jax.Array:
    """True outside the walls or above the ceiling (the floor is ground_collision's)."""
    x_lo, x_hi, y_lo, y_hi, ceiling = room
    x, y = p_w[..., 0], p_w[..., 1]
    return (x < x_lo) | (x > x_hi) | (y < y_lo) | (y > y_hi) | (-p_w[..., 2] > ceiling)
