"""The drone under CTBA in Betaflight ANGLE mode: a rigid body, Betaflight's
attitude estimator, its level loop, its closed rate loop, and the identified
thrust map.

NED world, FRD body (x forward, y right, z down), as skydreamer.dynamics.
The policy never commands rates or motors.  Per control period (10 ms):

    roll/pitch target --(level loop: LEVEL_GAIN * (target - FC estimate))--+
    yaw-rate target ---------------------------------------------------------+--> rate setpoint
    rate setpoint --(closed loop: 2nd order roll/pitch, 1st order yaw, accel-limited)--> omega
    throttle --(delay, PT1)--> specific thrust a_T --(climb loss)--> force

integrated at 1 kHz (semi-implicit Euler; the fastest pole is 2 pi 25.6 Hz, so
omega_n dt <= 0.16).  At the end of the period the FC estimate is updated once,
as Betaflight's 100 Hz attitude task does: Mahony filter on the period's mean
gyro and accelerometer, in the IMU frame (tilted from the airframe by the
mounting error), the accelerometer used only while its norm is 0.9-1.1 g.  The
level loop therefore steers the *estimate*; where the estimate is wrong
(sustained acceleration, gyro bias, mounting) the airframe is off by as much.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skydreamer.dynamics import euler_to_quat, quat_mul, quat_to_euler, quat_to_matrix

from . import params as P


class State(NamedTuple):
    # field names p, v, q, omega_b match skydreamer.dynamics.State, so skydreamer's
    # floor contact and rendering helpers work on this state unchanged.
    p: jax.Array        # (3,) world position, NED
    v: jax.Array        # (3,) world velocity
    q: jax.Array        # (4,) body->world [w, x, y, z]
    omega_b: jax.Array  # (3,) body rates
    omega_d: jax.Array  # (3,) body angular acceleration (rate-loop state)
    a_t: jax.Array      # () specific thrust, m/s^2
    acc_f: jax.Array    # (3,) FC-filtered specific force, body FRD, g
    thr_hist: jax.Array  # (3,) throttle held in the previous two periods and this one
    q_est: jax.Array    # (4,) Betaflight's estimate of the IMU frame's attitude, body->world


class Disturbance(NamedTuple):
    eps_a: jax.Array    # (3,) world acceleration, m/s^2
    eps_m: jax.Array    # (3,) body angular acceleration, rad/s^2


class ImuNoise(NamedTuple):
    """What corrupts the FC's own IMU reading over one period (estimator input)."""
    acc_bias: jax.Array   # (3,) g
    acc: jax.Array        # (3,) g, this period's noise on the mean
    gyro: jax.Array       # (3,) rad/s, this period's noise on the mean

    @staticmethod
    def zero():
        z = jnp.zeros(3)
        return ImuNoise(z, z, z)


def mount_matrix(p: P.CtbaParams) -> jax.Array:
    """IMU frame -> airframe rotation (small roll/pitch mounting error)."""
    return quat_to_matrix(euler_to_quat(p.mount_roll, p.mount_pitch, 0.0))


def attitude_rp(q) -> jax.Array:
    """(roll, pitch) as Betaflight computes them (imu.c: atan2 / asin of the rotation matrix's
    third row), which are skydreamer's ZYX Euler angles."""
    return quat_to_euler(q)[:2]


def estimate(q_est, gyro_s, acc_s, kp, dt=P.CONTROL_DT):
    """One Betaflight Mahony update (imu.c imuMahonyAHRSupdate, ki = 0), FRD/NED.
    gyro_s rad/s and acc_s g in the IMU frame, period means."""
    m2 = jnp.sum(acc_s * acc_s)
    lo, hi = P.EST_ACC_BAND
    use = (m2 > lo * lo) & (m2 < hi * hi)
    up_meas = acc_s / jnp.sqrt(jnp.maximum(m2, 1e-9))       # specific force points "up"
    up_est = -quat_to_matrix(q_est)[2, :]                    # world up (-z NED) in the IMU frame
    e = jnp.cross(up_meas, up_est)
    w = gyro_s + jnp.where(use, kp, 0.0) * e
    q = q_est + dt * 0.5 * quat_mul(q_est, jnp.concatenate([jnp.zeros(1), w]))
    return q / jnp.linalg.norm(q)


def rate_setpoint(s: State, att_cmd, yaw_rate_cmd, p: P.CtbaParams) -> jax.Array:
    """pid.c pidLevel (ANGLE mode): roll/pitch rate setpoint from the angle error against
    the FC's estimate; yaw passes through as a rate."""
    tgt = jnp.clip(att_cmd, -P.ANGLE_LIMIT, P.ANGLE_LIMIT)
    rp = p.level_gain * (tgt - attitude_rp(s.q_est))
    return jnp.concatenate([rp, jnp.reshape(yaw_rate_cmd, (1,))])


def body_specific_force(s: State, p: P.CtbaParams) -> jax.Array:
    """Aerodynamic + thrust specific force in the body frame (m/s^2), i.e. what
    an ideal accelerometer reads (at hover: (0, 0, -g))."""
    R = quat_to_matrix(s.q)
    vb = R.T @ s.v
    # climbing along the thrust axis (vb_z < 0) loses thrust; clipped so a fast
    # fall cannot invent thrust the fit never saw
    a_eff = jnp.maximum(s.a_t + p.climb_loss * jnp.clip(vb[2], -25.0, 3.0), 0.0)
    fx = -p.drag_x * vb[0] - p.drag_x2 * vb[0] * jnp.abs(vb[0])
    fy = -p.drag_y * vb[1] - p.drag_y2 * vb[1] * jnp.abs(vb[1])
    return jnp.stack([fx, fy, -a_eff])


def _substep(s: State, i, cmd, p: P.CtbaParams, d: Disturbance) -> State:
    # --- thrust: which period's throttle is acting at this substep, after the delay
    t_back = (i + 1) * P.DT - p.thrust_delay          # time since this period began, delayed
    idx = jnp.where(t_back >= 0.0, 2, jnp.where(t_back >= -P.CONTROL_DT, 1, 0))
    u = s.thr_hist[idx]
    v_bat = P.battery_voltage(p, s.a_t)
    a_tgt = P.thrust_static(u, v_bat, p.thrust_scale)
    a_t = s.a_t + P.DT / p.thrust_tau * (a_tgt - s.a_t)

    # --- closed rate loop
    w, wd = s.omega_b, s.omega_d
    wn = 2 * jnp.pi * jnp.stack([p.fn_roll, p.fn_pitch])
    zeta = jnp.stack([p.zeta_roll, p.zeta_pitch])
    gain = jnp.stack([p.gain_p, p.gain_q, p.gain_r])
    ref = gain * cmd
    wdd = wn ** 2 * (ref[:2] - w[:2]) - 2 * zeta * wn * wd[:2]
    alpha = jnp.stack([p.alpha_p, p.alpha_q, p.alpha_r])
    wd_rp = wd[:2] + P.DT * wdd
    wd_y = (ref[2] - w[2]) / p.yaw_tau
    wd_new = jnp.clip(jnp.concatenate([wd_rp, wd_y[None]]), -alpha, alpha) + d.eps_m
    w_new = w + P.DT * wd_new

    # --- translation
    s_mid = s._replace(a_t=a_t)
    f_b = body_specific_force(s_mid, p)
    R = quat_to_matrix(s.q)
    acc = R @ f_b + jnp.array([0.0, 0.0, P.GRAVITY]) + d.eps_a
    v_new = s.v + P.DT * acc
    p_new = s.p + P.DT * v_new

    # --- attitude (rates at the end of the substep)
    dq = 0.5 * quat_mul(s.q, jnp.concatenate([jnp.zeros(1), w_new]))
    q_new = s.q + P.DT * dq
    q_new = q_new / jnp.linalg.norm(q_new)

    # --- FC accelerometer filter (what Betaflight reports), in g
    acc_f = s.acc_f + P.DT / P.ACC_FILTER_TAU * (f_b / P.GRAVITY - s.acc_f)
    return State(p=p_new, v=v_new, q=q_new, omega_b=w_new, omega_d=wd_new, a_t=a_t,
                 acc_f=acc_f, thr_hist=s.thr_hist, q_est=s.q_est)


def step_rates(s: State, throttle, rate_sp, p: P.CtbaParams, d: Disturbance) -> State:
    """One control period with a rate setpoint held (the plant below the level loop);
    the FC estimate is not touched."""
    s = s._replace(thr_hist=jnp.concatenate([s.thr_hist[1:], jnp.reshape(throttle, (1,))]))

    def body(c, i):
        c = _substep(c, i, rate_sp, p, d)
        return c, (c.omega_b, c.acc_f)

    out, (w, a) = jax.lax.scan(body, s, jnp.arange(P.SUBSTEPS))
    return out, jnp.mean(w, 0), jnp.mean(a, 0)


def imu_reading(omega_b, acc_f, p: P.CtbaParams, n: ImuNoise):
    """Airframe rates / specific force -> what the IMU reports (its own frame)."""
    Rm = mount_matrix(p)
    gyro = Rm.T @ omega_b + n.gyro
    acc = Rm.T @ acc_f + n.acc_bias + n.acc
    return gyro, acc


def step(s: State, throttle, att_cmd, yaw_rate_cmd, p: P.CtbaParams, d: Disturbance,
         n: ImuNoise | None = None, grounded=False) -> State:
    """Advance one control period with (throttle, roll/pitch target, yaw-rate target) held.
    `grounded`: resting on the floor, the accelerometer reads 1 g up (the floor carries
    the weight), which the estimator sees too."""
    n = ImuNoise.zero() if n is None else n
    sp = rate_setpoint(s, att_cmd, yaw_rate_cmd, p)
    s, w_mean, a_mean = step_rates(s, throttle, sp, p, d)
    rest = quat_to_matrix(s.q).T @ jnp.array([0.0, 0.0, -1.0])
    a_mean = jnp.where(grounded, rest, a_mean)
    w_mean = jnp.where(grounded, 0.0, w_mean)
    gyro_s, acc_s = imu_reading(w_mean, a_mean, p, n)
    gyro_s = gyro_s + jnp.stack([p.gbias_x, p.gbias_y, 0.0])
    return s._replace(q_est=estimate(s.q_est, gyro_s, acc_s, p.est_kp))


def converged_estimate(q, p: P.CtbaParams, err=None) -> jax.Array:
    """The FC estimate after settling at rest: the IMU frame's attitude, i.e. the airframe's
    rotated by the mounting error, plus an optional (roll, pitch) error."""
    qi = quat_mul(q, euler_to_quat(p.mount_roll, p.mount_pitch, 0.0))
    if err is not None:
        rpy = quat_to_euler(qi)
        qi = euler_to_quat(rpy[0] + err[0], rpy[1] + err[1], rpy[2])
    return qi


def initial(p_w, q, omega, a_t, throttle, q_est=None) -> State:
    """A state at rest relative to its own thrust: filters and histories settled."""
    f_b = jnp.array([0.0, 0.0, -a_t]) / P.GRAVITY
    return State(p=p_w, v=jnp.zeros(3), q=q, omega_b=omega, omega_d=jnp.zeros(3),
                 a_t=jnp.asarray(a_t, jnp.float32), acc_f=f_b,
                 thr_hist=jnp.full((3,), throttle, jnp.float32),
                 q_est=q if q_est is None else q_est)


def measure(s: State, p: P.CtbaParams, acc_bias, key):
    """(gyro rad/s, acc g, FC roll/pitch rad) as Betaflight hands them over MSP
    (MSP_RAW_IMU in the IMU frame, MSP_ATTITUDE): noisy, quantized."""
    k1, k2 = jax.random.split(key)
    Rm = mount_matrix(p)
    gyro = Rm.T @ s.omega_b + p.gyro_sigma * jax.random.normal(k1, (3,))
    gyro = jnp.round(gyro / P.GYRO_QUANT) * P.GYRO_QUANT
    acc = Rm.T @ s.acc_f + acc_bias + p.acc_sigma * jax.random.normal(k2, (3,))
    acc = jnp.round(acc / P.ACC_QUANT) * P.ACC_QUANT
    att = jnp.round(attitude_rp(s.q_est) / P.ATT_QUANT) * P.ATT_QUANT
    return gyro, acc, att
