"""Quadcopter dynamic parameters and domain randomization.

Everything here is transcribed from the SkyDreamer paper (arXiv:2510.14783):
  - `NOMINAL` is Table II verbatim.
  - `TRAIN` / `EVAL` are Table III verbatim.

Paper ambiguities are marked with `PAPER-AMBIGUITY` and collected in
docs/paper_gaps.md.  Do not silently "fix" them here without updating that file.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------
# Table II: quadcopter dynamical parameters
# --------------------------------------------------------------------------

NOMINAL: dict[str, float] = {
    # thrust / drag
    "k_w": 1.55e-6,
    "k_x": 5.37e-5,
    "k_y": 5.37e-5,
    "k_x2": 4.10e-3,
    "k_y2": 1.51e-2,
    "k_v2": 0.00,
    # thrust corrections (angle of attack, horizontal advance ratio)
    # PAPER-AMBIGUITY: Table II calls this `k_angle`, eq. (F) calls it `k_alpha`.
    # They are the same coefficient.
    "k_angle": 3.145,
    "k_hor": 7.245,
    # inertia coupling / gyroscopic
    "J_x": -0.89,
    "J_y": 0.96,
    "J_z": -0.34,
    # motor response
    "w_min": 341.75,
    "w_max": 3100.00,
    "k": 0.50,
    "tau": 0.03,
    # roll moments
    "k_p1": 4.99e-5,
    "k_p2": 3.78e-5,
    "k_p3": 4.82e-5,
    "k_p4": 3.83e-5,
    # pitch moments
    "k_q1": 2.05e-5,
    "k_q2": 2.46e-5,
    "k_q3": 2.02e-5,
    "k_q4": 2.57e-5,
    # yaw moments (k_r1-k_r4 share a value, as do k_r5-k_r8)
    "k_r1": 3.38e-3,
    "k_r2": 3.38e-3,
    "k_r3": 3.38e-3,
    "k_r4": 3.38e-3,
    "k_r5": 3.24e-4,
    "k_r6": 3.24e-4,
    "k_r7": 3.24e-4,
    "k_r8": 3.24e-4,
}

# Order matters: this is the layout of the `d` vector the world model decodes.
PARAM_NAMES: tuple[str, ...] = tuple(NOMINAL.keys())

# Not in SkyDreamer's Table II, but given by the source it defers to for the
# dynamics.  MonoRace (Bahnam et al., arXiv:2601.15222) defines `r` as "the
# propeller radius" and puts the value in a footnote: 0.0485775 m, estimated
# rather than measured, and not randomized.  Note it is not the geometric
# radius of the 5.1" props (0.06477 m) but 0.75 of it -- the 75% blade station,
# the conventional reference section in blade element theory.
PROP_RADIUS = 0.0485775  # m [MonoRace footnote 4]

GRAVITY = 9.81

# `w_min` and `w_max` are randomized +-20%, every other parameter +-30% (train)
# or +-20% (eval).  See Table III.
_MOTOR_LIMIT_PARAMS = ("w_min", "w_max")


class DynParams(NamedTuple):
    """Per-drone dynamic parameters.  Every field is a scalar or a batch of
    scalars, so this pytree vmaps cleanly over parallel environments."""

    k_w: jax.Array
    k_x: jax.Array
    k_y: jax.Array
    k_x2: jax.Array
    k_y2: jax.Array
    k_v2: jax.Array
    k_angle: jax.Array
    k_hor: jax.Array
    J_x: jax.Array
    J_y: jax.Array
    J_z: jax.Array
    w_min: jax.Array
    w_max: jax.Array
    k: jax.Array
    tau: jax.Array
    k_p1: jax.Array
    k_p2: jax.Array
    k_p3: jax.Array
    k_p4: jax.Array
    k_q1: jax.Array
    k_q2: jax.Array
    k_q3: jax.Array
    k_q4: jax.Array
    k_r1: jax.Array
    k_r2: jax.Array
    k_r3: jax.Array
    k_r4: jax.Array
    k_r5: jax.Array
    k_r6: jax.Array
    k_r7: jax.Array
    k_r8: jax.Array

    @classmethod
    def nominal(cls, batch: int | None = None) -> "DynParams":
        def make(v):
            return jnp.full((batch,), v) if batch else jnp.asarray(v)

        return cls(**{k: make(v) for k, v in NOMINAL.items()})

    def as_vector(self) -> jax.Array:
        """Stack into the `d` vector, each entry normalized to [-1, 1] against
        its own widest training range, so every decoder target has the same
        scale regardless of whether it was randomized +-20% or +-30%."""
        cols = []
        for name in PARAM_NAMES:
            nom = NOMINAL[name]
            val = getattr(self, name)
            if nom == 0.0:
                # k_v2 is nominally zero, so a relative deviation is undefined.
                cols.append(jnp.zeros_like(val))
                continue
            frac = TRAIN.motor_limit_frac if name in _MOTOR_LIMIT_PARAMS else TRAIN.param_frac
            cols.append((val / nom - 1.0) / frac)
        return jnp.stack(cols, axis=-1)


# --------------------------------------------------------------------------
# Table III: randomization, and initial states
# --------------------------------------------------------------------------


class RandomizationConfig(NamedTuple):
    """Table III.  Angles in radians, everything else in SI."""

    # camera extrinsics (roll, pitch, yaw of body->camera)
    cam_roll: tuple[float, float]
    cam_pitch: tuple[float, float]
    cam_yaw: tuple[float, float]
    # multiplicative parameter randomization
    motor_limit_frac: float  # w_min, w_max
    param_frac: float  # everything else in `d`
    # disturbances
    eps_a_slow: float  # m/s^2, resampled with p = 1/100 per step
    eps_M_slow: float  # rad/s^2, resampled with p = 1/100 per step
    eps_M_fast: float  # rad/s^2, resampled every step
    eps_u_fast: float  # normalized motor command, resampled every step
    # gate geometry used for reward and collision
    d_g: float  # effective gate half-size, m
    t_g: float  # pre-gate..post-gate total thickness, m
    # initial state, expressed in the frame of the first gate
    init_x_g: tuple[float, float]
    init_y_g: tuple[float, float]
    init_z_g: tuple[float, float]
    init_rate: tuple[float, float]  # p, q, r
    init_motor: tuple[float, float]  # fraction of w_max
    init_att: tuple[float, float]  # roll, pitch, and yaw relative to gate


_DEG = jnp.pi / 180.0

TRAIN = RandomizationConfig(
    cam_roll=(-5 * _DEG, 5 * _DEG),
    cam_pitch=(45 * _DEG, 55 * _DEG),
    cam_yaw=(-5 * _DEG, 5 * _DEG),
    motor_limit_frac=0.20,
    param_frac=0.30,
    eps_a_slow=3.0,
    eps_M_slow=3.0,
    eps_M_fast=125.0,
    eps_u_fast=0.2,
    d_g=0.8,
    t_g=0.8,
    init_x_g=(-4.0, -2.0),
    init_y_g=(-1.0, 1.0),
    init_z_g=(0.0, 1.3),
    init_rate=(-0.1, 0.1),
    init_motor=(0.25, 0.5),
    init_att=(-jnp.pi / 9, jnp.pi / 9),
)

EVAL = TRAIN._replace(
    param_frac=0.20,
    eps_a_slow=2.0,
    eps_M_slow=2.0,
    eps_M_fast=100.0,
    d_g=1.0,
    init_z_g=(0.7, 1.3),
)

# Section III-D: the big-track run additionally perturbs w_max by +-300 rad/s
# resampled every 10 steps, to cover imperfect actuator modelling.
BIG_TRACK_W_MAX_DISTURBANCE = 300.0
BIG_TRACK_W_MAX_RESAMPLE_EVERY = 10


def _uniform(key: jax.Array, lo: float, hi: float, shape=()) -> jax.Array:
    return jax.random.uniform(key, shape, minval=lo, maxval=hi)


def sample_params(key: jax.Array, cfg: RandomizationConfig) -> DynParams:
    """Sample one drone's dynamic parameters per Table III."""
    keys = jax.random.split(key, len(PARAM_NAMES))
    out = {}
    for k, name in zip(keys, PARAM_NAMES):
        frac = (
            cfg.motor_limit_frac if name in _MOTOR_LIMIT_PARAMS else cfg.param_frac
        )
        out[name] = NOMINAL[name] * _uniform(k, 1.0 - frac, 1.0 + frac)
    return DynParams(**out)


def sample_extrinsics(key: jax.Array, cfg: RandomizationConfig) -> jax.Array:
    """Sample camera extrinsics `c_e = [phi_c, theta_c, psi_c]` (rad)."""
    kr, kp, ky = jax.random.split(key, 3)
    return jnp.stack(
        [
            _uniform(kr, *cfg.cam_roll),
            _uniform(kp, *cfg.cam_pitch),
            _uniform(ky, *cfg.cam_yaw),
        ]
    )
