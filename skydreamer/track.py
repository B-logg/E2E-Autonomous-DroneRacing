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


def gate_reward(p_g: jax.Array, d_g: float) -> jax.Array:
    """1 at the plane centre, falling linearly to 0 at the effective edge."""
    off = jnp.maximum(jnp.abs(p_g[..., 1]), jnp.abs(p_g[..., 2]))
    return 1.0 - off / d_g


def gate_collision(p_g: jax.Array, d_g: float) -> jax.Array:
    off = jnp.maximum(jnp.abs(p_g[..., 1]), jnp.abs(p_g[..., 2]))
    return off / d_g > 1.0


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


def inverted_loop() -> Track:
    """Figure 6 / Table IV "Loop (orange)".  Orange gates: inner 1.5 m, outer
    2.7 m [TEXT]."""
    return make_track(INVERTED_LOOP, inner=1.5, outer=2.7)


def ladder_inverted_loop() -> Track:
    """Figures 4 and 5 / Table IV "Ladder loop (orange)".  This is the track the
    paper's simulation results come from, and section III-B trains it with
    t_g = 0.3 m rather than Table III's 0.8 m."""
    return make_track(LADDER_INVERTED_LOOP, inner=1.5, outer=2.7)
