"""CTBA plant parameters (collective thrust + roll/pitch angle + yaw rate, flown
through Betaflight's ANGLE mode): nominal values identified from our manual-flight logs,
and the per-episode randomization around them.

Every nominal value cites where it came from.  `[LOG]` = identified from the
2026-11-15 Blackbox logs (hierarchy_control/sysid, drone_geometry/flight_sysid.md);
`[EST]` = an estimate the logs could not pin down; `[SD]` = SkyDreamer's value,
kept because nothing better exists yet.

Units: SI, except thrust-map inputs (throttle 0..1, volts).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
CONTROL_FREQ = 100.0          # policy rate, Hz.  Betaflight: attitude estimate 100 Hz, PID 4 kHz.
SUBSTEPS = 10                 # plant integrated at 1 kHz
CONTROL_DT = 1.0 / CONTROL_FREQ
DT = CONTROL_DT / SUBSTEPS

# ---------------------------------------------------------------------------
# Action scaling: the policy's [-1, 1]^4 -> (throttle, roll, pitch, yaw rate).
# Roll/pitch are Betaflight ANGLE-mode stick deflections: target angle =
# ANGLE_LIMIT * deflection, exactly as in the manual flights.  Yaw stays a rate;
# Betaflight's yaw rate curve must be LINEAR with this full-stick value (README).
# ---------------------------------------------------------------------------
ANGLE_LIMIT = 55.0 * 3.141592653589793 / 180.0   # [LOG] fitted 55.0-55.8 deg; the FC's angle_limit MUST be 55
YAW_RATE_MAX = 4.0            # rad/s at full yaw stick (229 deg/s)

# ANGLE-mode level loop [LOG] (pid.c pidLevel): rate setpoint = LEVEL_GAIN * (target - FC attitude),
# per Euler axis.  Fitted 4.81-4.95 1/s on LOG3-7 (R2 0.98-0.998); levelPID P = 50 -> 5.0.
LEVEL_GAIN = 4.9

# FC attitude estimate [LOG] (imu.c, Mahony): 100 Hz, gyro + accelerometer averaged over the
# period, accelerometer used only while 0.9 g < |acc| < 1.1 g, proportional gain dcm_kp.
# Reproduced on LOG3-7: kp 0.20-0.25 best; FC CLI imu_dcm_kp = 2500 -> 0.25 (2026-11-16).  Without the 0.9-1.1 g band the
# level-loop fit drops from R2 0.99 to 0.78-0.92, so the band is real.
EST_KP = 0.25
EST_ACC_BAND = (0.9, 1.1)

# ---------------------------------------------------------------------------
# Thrust map [LOG]: specific thrust (m/s^2) along -body z vs collective throttle u
#   a = g * scale * (K1 w + K2 w^2),  w = u * V / V_REF
# fitted on 9 punch / throttle-chop events plus 5155 hover samples (BETA = 1 fixed).
# ---------------------------------------------------------------------------
K1, K2, V_REF = 2.009, 3.951, 22.2
THRUST_TAU = 0.017            # [LOG] 22.6 ms fitted, less the FC's 25 Hz acc filter (6.4 ms)
THRUST_DELAY = 0.010          # [LOG] fitted pure delay; tau + delay = 33 ms is the robust part
CLIMB_LOSS = 0.084 * GRAVITY  # [LOG] 0.084 g per m/s climbed: inflow + body drag, 1/s
ACC_FILTER_TAU = 1.0 / (2 * 3.141592653589793 * 25.0)   # FC acc_lpf_hz = 25 Hz

# Battery [LOG]: open-circuit voltage over the flights 25.1 -> 22.2 V; internal
# resistance 18-35 mOhm; hover current ~10 A; current ~ thrust^1.5 (momentum theory).
V_OC = 23.4
R_INT = 0.025
I_HOVER = 10.0

# ---------------------------------------------------------------------------
# Closed rate loop [LOG]: Betaflight PID setpoint -> filtered gyro, 1-10 Hz,
# coherence-weighted.  Roll and pitch are second order; yaw first order.
# Identified in ANGLE mode, i.e. the mode we deploy in (feedforward included as flown).
# ---------------------------------------------------------------------------
RATE_FN = (18.3, 11.0)        # Hz, roll, pitch
RATE_ZETA = (0.49, 0.51)
YAW_TAU = 0.0216              # s
RATE_GAIN = (1.0, 1.0, 1.0)   # fitted 1.03 / 1.11 / 1.07 in angle mode; 1.0 nominal, randomized
# [EST] angular-acceleration limits (the logs only reached 47 / 44 / 33 rad/s^2):
# hover-thrust headroom x arm / inertia estimated from the CAD layout.
ALPHA_MAX = (420.0, 220.0, 30.0)

# ---------------------------------------------------------------------------
# Drag [SD]: not identifiable without velocity; SkyDreamer's body drag, scaled
# to specific force.  Linear term ~ rotor drag at hover, quadratic ~ body.
# ---------------------------------------------------------------------------
DRAG_LIN = (0.35, 0.45)       # 1/s, body x, y
DRAG_QUAD = (4.1e-3, 1.51e-2)  # 1/m

# ---------------------------------------------------------------------------
# Sensors [LOG]
# ---------------------------------------------------------------------------
DEG = 3.141592653589793 / 180.0
GYRO_QUANT = 1.0 * DEG        # MSP_RAW_IMU reports integer deg/s
ACC_QUANT = 1.0 / 512.0       # MSP_RAW_IMU acc: 512 LSB/g
GYRO_SIGMA = (1.0 * DEG, 4.0 * DEG)   # measured 1.5-2.5 deg/s above 15 Hz in flight
ACC_SIGMA = (0.02, 0.10)              # g; measured 0.03-0.08
ACC_BIAS = 0.02                       # g, per axis
ATT_QUANT = 0.1 * DEG         # MSP_ATTITUDE reports decidegrees
GYRO_BIAS_EST = 0.5 * DEG     # residual gyro bias in the FC estimate, per axis [LOG9 raw: -0.19/+0.01/+0.39 deg/s]
MOUNT_TILT = 1.5 * DEG        # IMU vs airframe, roll/pitch [LOG9 at rest: -0.9 / +0.5 deg incl. the floor]
EST_INIT_ERR = 2.0 * DEG      # mid-air starts: FC estimate off by up to this per axis

# Delays, in control steps (10 ms)
ACTION_DELAY = (1, 2)          # Jetson -> MSP -> Betaflight RC path [EST]
IMAGE_DELAY = (2, 5)           # exposure -> GateNet mask [EST]; SkyDreamer: 33 ms
ACTION_DELAY_MAX = ACTION_DELAY[1]
IMAGE_DELAY_MAX = IMAGE_DELAY[1]


class CtbaParams(NamedTuple):
    """One airframe + battery + sensor + FC-estimator draw.  Scalars (or batches of scalars)."""
    thrust_scale: jax.Array
    thrust_tau: jax.Array
    thrust_delay: jax.Array
    climb_loss: jax.Array
    v_oc: jax.Array
    r_int: jax.Array
    i_hover: jax.Array
    fn_roll: jax.Array
    fn_pitch: jax.Array
    zeta_roll: jax.Array
    zeta_pitch: jax.Array
    yaw_tau: jax.Array
    gain_p: jax.Array
    gain_q: jax.Array
    gain_r: jax.Array
    alpha_p: jax.Array
    alpha_q: jax.Array
    alpha_r: jax.Array
    drag_x: jax.Array
    drag_y: jax.Array
    drag_x2: jax.Array
    drag_y2: jax.Array
    gyro_sigma: jax.Array
    acc_sigma: jax.Array
    level_gain: jax.Array
    est_kp: jax.Array
    gbias_x: jax.Array      # rad/s, gyro bias seen by the FC estimator
    gbias_y: jax.Array
    mount_roll: jax.Array   # rad, IMU frame vs airframe
    mount_pitch: jax.Array


NOMINAL = dict(
    thrust_scale=1.0, thrust_tau=THRUST_TAU, thrust_delay=THRUST_DELAY, climb_loss=CLIMB_LOSS,
    v_oc=V_OC, r_int=R_INT, i_hover=I_HOVER,
    fn_roll=RATE_FN[0], fn_pitch=RATE_FN[1], zeta_roll=RATE_ZETA[0], zeta_pitch=RATE_ZETA[1],
    yaw_tau=YAW_TAU, gain_p=RATE_GAIN[0], gain_q=RATE_GAIN[1], gain_r=RATE_GAIN[2],
    alpha_p=ALPHA_MAX[0], alpha_q=ALPHA_MAX[1], alpha_r=ALPHA_MAX[2],
    drag_x=DRAG_LIN[0], drag_y=DRAG_LIN[1], drag_x2=DRAG_QUAD[0], drag_y2=DRAG_QUAD[1],
    gyro_sigma=2.0 * DEG, acc_sigma=0.05,
    level_gain=LEVEL_GAIN, est_kp=EST_KP, gbias_x=0.0, gbias_y=0.0, mount_roll=0.0, mount_pitch=0.0,
)

# (lo, hi) absolute ranges for training.  Evaluation uses the same ranges: the
# real values are uncertain by about this much, so there is no narrower "truth".
RANGES = dict(
    thrust_scale=(0.8, 1.2),             # max thrust; covers the beta ambiguity and mass +-
    thrust_tau=(0.012, 0.030),
    thrust_delay=(0.005, 0.015),
    climb_loss=(0.5 * CLIMB_LOSS, 1.5 * CLIMB_LOSS),
    v_oc=(21.6, 25.2),                   # 3.6-4.2 V/cell
    r_int=(0.018, 0.040),
    i_hover=(9.0, 11.0),
    fn_roll=(0.8 * RATE_FN[0], 1.25 * RATE_FN[0]),  # same mode as identified: fit spread only
    fn_pitch=(0.8 * RATE_FN[1], 1.25 * RATE_FN[1]),
    zeta_roll=(0.35, 0.8),
    zeta_pitch=(0.35, 0.8),
    yaw_tau=(0.6 * YAW_TAU, 1.6 * YAW_TAU),
    gain_p=(0.9, 1.15), gain_q=(0.9, 1.15), gain_r=(0.9, 1.15),
    alpha_p=(0.6 * ALPHA_MAX[0], 1.5 * ALPHA_MAX[0]),
    alpha_q=(0.6 * ALPHA_MAX[1], 1.5 * ALPHA_MAX[1]),
    alpha_r=(0.6 * ALPHA_MAX[2], 1.5 * ALPHA_MAX[2]),
    drag_x=(0.5 * DRAG_LIN[0], 1.5 * DRAG_LIN[0]),
    drag_y=(0.5 * DRAG_LIN[1], 1.5 * DRAG_LIN[1]),
    drag_x2=(0.5 * DRAG_QUAD[0], 1.5 * DRAG_QUAD[0]),
    drag_y2=(0.5 * DRAG_QUAD[1], 1.5 * DRAG_QUAD[1]),
    gyro_sigma=GYRO_SIGMA,
    acc_sigma=ACC_SIGMA,
    level_gain=(4.6, 5.2),               # fitted 4.81-4.95
    est_kp=(0.22, 0.28),                 # read from the FC (0.25); +-10% for filter-model mismatch
    gbias_x=(-GYRO_BIAS_EST, GYRO_BIAS_EST),
    gbias_y=(-GYRO_BIAS_EST, GYRO_BIAS_EST),
    mount_roll=(-MOUNT_TILT, MOUNT_TILT),
    mount_pitch=(-MOUNT_TILT, MOUNT_TILT),
)
PARAM_NAMES = tuple(CtbaParams._fields)


def nominal(batch: int | None = None) -> CtbaParams:
    mk = (lambda v: jnp.full((batch,), v, jnp.float32)) if batch else (lambda v: jnp.asarray(v, jnp.float32))
    return CtbaParams(**{k: mk(NOMINAL[k]) for k in PARAM_NAMES})


def sample(key: jax.Array) -> CtbaParams:
    ks = jax.random.split(key, len(PARAM_NAMES))
    return CtbaParams(**{n: jax.random.uniform(k, minval=RANGES[n][0], maxval=RANGES[n][1])
                         for k, n in zip(ks, PARAM_NAMES)})


def as_vector(p: CtbaParams) -> jax.Array:
    """Decoder target: every parameter mapped to [-1, 1] over its range."""
    cols = []
    for n in PARAM_NAMES:
        lo, hi = RANGES[n]
        cols.append(2.0 * (getattr(p, n) - lo) / (hi - lo) - 1.0)
    return jnp.stack(cols, -1)


def thrust_static(u, v, scale):
    """Static specific thrust (m/s^2) at throttle u and battery voltage v."""
    w = u * v / V_REF
    return GRAVITY * scale * (K1 * w + K2 * w * w)


def throttle_for(a, v, scale):
    """Inverse of `thrust_static`: throttle giving specific thrust `a` at voltage v."""
    c = jnp.maximum(a, 0.0) / (GRAVITY * scale)
    w = (-K1 + jnp.sqrt(K1 * K1 + 4.0 * K2 * c)) / (2.0 * K2)
    return jnp.clip(w * V_REF / jnp.maximum(v, 1.0), 0.0, 1.0)


def battery_voltage(p: CtbaParams, a_thrust):
    """Loaded voltage: open circuit less the IR drop at the current thrust."""
    i = p.i_hover * jnp.power(jnp.maximum(a_thrust, 0.0) / GRAVITY, 1.5)
    return p.v_oc - p.r_int * i
