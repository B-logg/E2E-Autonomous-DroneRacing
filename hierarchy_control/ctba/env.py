"""The racing environment under CTBA.

The same informed POMDP as skydreamer.env -- same track, gates, rewards,
termination, floor launch, mask rendering and augmentation -- with three
changes:

  * action    (collective throttle, roll angle, pitch angle, yaw rate) for
              Betaflight's ANGLE mode, i.e. its stick inputs, not motor commands;
  * plant     ctba.plant: Betaflight's attitude estimate, level loop and rate
              loop as identified, and the identified thrust map;
  * observation  mask + gyro + accelerometer + the FC's roll/pitch estimate (as
              Betaflight reports them over MSP) + flight plan.  No RPM: our ESC
              (BLHeli_S) does not send it.

Policy rate 100 Hz.  Action delay 1-2 steps and image delay 2-5 steps, drawn
per episode (ctba.params).  Pure-functional JAX; `reset`/`step` jit and vmap.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skydreamer import augment, track as trk
from skydreamer import env as SE
from skydreamer.dynamics import euler_to_quat, quat_to_euler, quat_to_matrix
from skydreamer.params import TRAIN as SD_TRAIN, EVAL as SD_EVAL, sample_extrinsics

from . import params as P
from . import plant
from .tasks import CtbaConfig

PACK_MASK = SE.PACK_MASK
N_DYN = len(P.PARAM_NAMES)

# Table III's disturbances, scaled for a closed-loop plant: the rate loop rejects
# moment disturbances, so what is left is a small residual angular acceleration.
EPS_A_SLOW = (SD_TRAIN.eps_a_slow, SD_EVAL.eps_a_slow)   # m/s^2, world (3.0 / 2.0)
EPS_M_SLOW = 2.0          # rad/s^2, held ~1 s
EPS_M_FAST = 10.0         # rad/s^2, every step
RESAMPLE_P = 1.0 / P.CONTROL_FREQ   # slow terms: once a second on average


class EnvState(NamedTuple):
    key: jax.Array
    s: plant.State
    params: P.CtbaParams
    acc_bias: jax.Array      # (3,) g
    act_delay: jax.Array     # int32, control steps
    img_delay: jax.Array     # int32, control steps
    c_e: jax.Array
    d_g: jax.Array
    t_g: jax.Array
    eps_a_amp: jax.Array
    eps_a: jax.Array
    eps_m_slow: jax.Array
    rs_s: jax.Array
    erode: jax.Array
    plane: jax.Array
    plane0: jax.Array
    armed: jax.Array
    fp_index: jax.Array
    fp_done: jax.Array
    prev_dist: jax.Array
    step_count: jax.Array
    done: jax.Array
    action_buf: jax.Array    # (ACTION_DELAY_MAX, 4) in physical units (thr, roll, pitch, yaw rate)
    mask_buf: jax.Array      # (IMAGE_DELAY_MAX + 1, H, W)
    term_cause: jax.Array
    grounded: jax.Array
    airborne: jax.Array
    alt_ok: jax.Array
    approached: jax.Array


CMD_SCALE = (P.ANGLE_LIMIT, P.ANGLE_LIMIT, P.YAW_RATE_MAX)


def action_to_command(a: jax.Array) -> jax.Array:
    """Policy action in [-1, 1]^4 -> (throttle 0..1, roll rad, pitch rad, yaw rate rad/s).
    a1, a2 are exactly Betaflight's ANGLE-mode stick deflections (FRD signs: a2 > 0 nose up)."""
    a = jnp.clip(a, -1.0, 1.0)
    thr = 0.5 * (a[..., :1] + 1.0)
    rest = a[..., 1:] * jnp.asarray(CMD_SCALE)
    return jnp.concatenate([thr, rest], -1)


def command_to_action(c: jax.Array) -> jax.Array:
    """Inverse of `action_to_command` (demonstrations)."""
    a0 = 2.0 * c[..., :1] - 1.0
    a1 = c[..., 1:] / jnp.asarray(CMD_SCALE)
    return jnp.clip(jnp.concatenate([a0, a1], -1), -1.0, 1.0)


def _mix(a, b, use_train):
    return jnp.where(use_train, a, b)


def reset(key: jax.Array, ccfg: CtbaConfig):
    cfg = ccfg.base
    k = jax.random.split(key, 16)
    use_train = jax.random.uniform(k[0]) < cfg.train_fraction

    d_g = _mix(cfg.d_g[0], cfg.d_g[1], use_train)
    t_g = jnp.asarray(cfg.t_g if cfg.t_g is not None else SD_TRAIN.t_g, jnp.float32)
    params = P.sample(k[1])
    c_e = sample_extrinsics(k[2], SD_TRAIN)
    acc_bias = jax.random.uniform(k[3], (3,), minval=-P.ACC_BIAS, maxval=P.ACC_BIAS)
    act_delay = jax.random.randint(k[4], (), P.ACTION_DELAY[0], P.ACTION_DELAY[1] + 1)
    img_delay = jax.random.randint(k[5], (), P.IMAGE_DELAY[0], P.IMAGE_DELAY[1] + 1)

    n = cfg.track.n_gates
    start_gate = jnp.where(use_train, jax.random.randint(k[6], (), 0, n), 0)
    gate_pos = cfg.track.pos[start_gate]
    gate_yaw = cfg.track.yaw[start_gate]
    R_wg = trk.gate_rotation(gate_yaw).T

    # --- Table III mid-air start (training column)
    ks = jax.random.split(k[7], 6)
    x_g = jax.random.uniform(ks[0], minval=SD_TRAIN.init_x_g[0], maxval=SD_TRAIN.init_x_g[1])
    y_g = jax.random.uniform(ks[1], minval=SD_TRAIN.init_y_g[0], maxval=SD_TRAIN.init_y_g[1])
    z_lo = _mix(SD_TRAIN.init_z_g[0], SD_EVAL.init_z_g[0], use_train)
    z_hi = jnp.asarray(SD_TRAIN.init_z_g[1])
    if cfg.air_start_alt is not None:
        z_hi = jnp.minimum(z_hi, -gate_pos[2] - cfg.air_start_alt)
        z_lo = jnp.minimum(z_lo, z_hi)
    z_g = jax.random.uniform(ks[2], minval=z_lo, maxval=z_hi)
    rates = jax.random.uniform(ks[3], (3,), minval=SD_TRAIN.init_rate[0], maxval=SD_TRAIN.init_rate[1])
    att = jax.random.uniform(ks[5], (3,), minval=SD_TRAIN.init_att[0], maxval=SD_TRAIN.init_att[1])
    # thrust: 0.8-1.3 g, so it starts somewhere near a hover it must find itself
    a_t = P.GRAVITY * jax.random.uniform(ks[4], minval=0.8, maxval=1.3)

    p_w = gate_pos + R_wg @ jnp.stack([x_g, y_g, z_g])
    if cfg.room is not None:
        x_lo, x_hi, y_lo, y_hi, ceiling = cfg.room
        m = SE.ROOM_MARGIN
        p_w = jnp.clip(p_w, jnp.array([x_lo + m, y_lo + m, -ceiling + m]),
                       jnp.array([x_hi - m, y_hi - m, -m]))
    q = euler_to_quat(att[0], att[1], gate_yaw + att[2])

    # --- the real start: on the floor before the start gate, motors idle
    launch = jnp.array(False)
    if cfg.launch is not None:
        d_lo, d_hi, y_half, yaw_half, rest = cfg.launch
        kl = jax.random.split(k[8], 3)
        d = jax.random.uniform(kl[0], minval=d_lo, maxval=d_hi)
        y_l = jax.random.uniform(kl[1], minval=-y_half, maxval=y_half)
        yaw_l = jax.random.uniform(kl[2], minval=-yaw_half, maxval=yaw_half) * jnp.pi / 180.0
        p_l = (gate_pos + R_wg @ jnp.stack([-d, y_l, 0.0])).at[2].set(-rest)
        launch = ~use_train
        p_w = jnp.where(launch, p_l, p_w)
        q = jnp.where(launch, euler_to_quat(0.0, 0.0, gate_yaw + yaw_l), q)
        rates = jnp.where(launch, 0.0, rates)
        a_t = jnp.where(launch, 0.0, a_t)

    v_bat = P.battery_voltage(params, a_t)
    thr0 = P.throttle_for(a_t, v_bat, params.thrust_scale)
    # FC estimate: settled on the floor (the IMU frame's attitude), or off by up to
    # EST_INIT_ERR in a mid-air start
    est_err = jax.random.uniform(k[14], (2,), minval=-P.EST_INIT_ERR, maxval=P.EST_INIT_ERR)
    est_err = jnp.where(launch, 0.0, est_err)
    s = plant.initial(p_w, q, rates, a_t, thr0, q_est=plant.converged_estimate(q, params, est_err))
    s = s._replace(acc_f=jnp.where(launch, jnp.array([0.0, 0.0, -1.0]), s.acc_f))

    plane = start_gate * trk.PLANES_PER_GATE
    target, target_yaw, _ = trk.plane_pose(cfg.track, plane, t_g)
    armed0 = (trk.gate_rotation(target_yaw) @ (p_w - target))[0] < 0.0
    eps_a_amp = _mix(EPS_A_SLOW[0], EPS_A_SLOW[1], use_train)
    cmd0 = jnp.concatenate([thr0[None], jnp.zeros(3)])

    state = EnvState(
        key=k[9], s=s, params=params, acc_bias=acc_bias, act_delay=act_delay, img_delay=img_delay,
        c_e=c_e, d_g=d_g, t_g=t_g, eps_a_amp=eps_a_amp,
        eps_a=jax.random.uniform(k[10], (3,), minval=-eps_a_amp, maxval=eps_a_amp),
        eps_m_slow=jax.random.uniform(k[11], (3,), minval=-EPS_M_SLOW, maxval=EPS_M_SLOW),
        rs_s=jax.random.uniform(k[12], minval=augment.ROLLING_SHUTTER_RANGE[0],
                                maxval=augment.ROLLING_SHUTTER_RANGE[1]),
        erode=jax.random.uniform(k[13]) < cfg.erosion_prob,
        plane=plane, plane0=plane, armed=armed0, fp_index=start_gate, fp_done=jnp.array(False),
        prev_dist=SE._progress_distance(cfg, p_w, plane, t_g),
        step_count=jnp.array(0, jnp.int32), done=jnp.array(False),
        action_buf=jnp.broadcast_to(cmd0, (P.ACTION_DELAY_MAX, 4)),
        mask_buf=jnp.zeros((P.IMAGE_DELAY_MAX + 1, cfg.image_size, cfg.image_size)),
        term_cause=jnp.array(0, jnp.int32), grounded=launch, airborne=~launch,
        alt_ok=jnp.array(False), approached=jnp.array(False),
    )
    mask = SE._render_augmented(state, cfg)
    state = state._replace(mask_buf=jnp.broadcast_to(mask, state.mask_buf.shape))
    return state, observe(state, ccfg)


def observe(state: EnvState, ccfg: CtbaConfig) -> dict:
    cfg = ccfg.base
    key = jax.random.fold_in(state.key, state.step_count)
    gyro, acc, att_fc = plant.measure(state.s, state.params, state.acc_bias, key)
    att_err = trk.wrap_pi(plant.attitude_rp(state.s.q_est) - plant.attitude_rp(state.s.q))
    fp = trk.flight_plan(cfg.track, state.fp_index)
    p_g, v_g, gate_yaw = SE._gate_relative(state, cfg)
    euler = quat_to_euler(state.s.q)
    idx = P.IMAGE_DELAY_MAX - state.img_delay          # img_delay steps old
    mask = state.mask_buf[idx]
    delays = jnp.stack([
        (state.act_delay - P.ACTION_DELAY[0]) / max(P.ACTION_DELAY[1] - P.ACTION_DELAY[0], 1),
        (state.img_delay - P.IMAGE_DELAY[0]) / max(P.IMAGE_DELAY[1] - P.IMAGE_DELAY[0], 1),
    ]).astype(jnp.float32) * 2.0 - 1.0
    return {
        # --- observation: what the policy has on the drone
        "mask": SE.pack_mask(mask) if PACK_MASK else mask[..., None].astype(jnp.uint8) * 255,
        "rates": gyro,
        "acc": acc,
        "att": att_fc,
        "flight_plan": fp,
        # --- information: decoder targets, training only
        "info_p_w": state.s.p,
        "info_p_g": p_g,
        "info_v_w": state.s.v,
        "info_v_g": v_g,
        "info_att": jnp.concatenate([euler, trk.wrap_pi(euler[2:3] - gate_yaw)]),
        "info_rates": state.s.omega_b,
        "info_att_err": att_err,
        "info_thrust": jnp.reshape(state.s.a_t / P.GRAVITY, (1,)),
        "info_acc": state.s.acc_f,
        "info_cam": state.c_e,
        "info_dyn": P.as_vector(state.params),
        "info_delay": delays,
        "info_meas_rates": gyro,
        "info_meas_acc": acc,
        "info_meas_att": att_fc,
        "info_flight_plan": fp,
        # --- diagnostics (stripped before the agent and the replay)
        "log/diverged": (state.term_cause == 3).astype(jnp.float32),
        "log/hit_gate": (state.term_cause == 1).astype(jnp.float32),
        "log/hit_ground": ((state.term_cause == 2) | (state.term_cause == 4)).astype(jnp.float32),
        "log/rate_l1": jnp.sum(jnp.abs(state.s.omega_b)).astype(jnp.float32),
        "log/speed": jnp.linalg.norm(state.s.v).astype(jnp.float32),
        "log/att_err_deg": (jnp.max(jnp.abs(att_err)) * 180.0 / jnp.pi).astype(jnp.float32),
    }


OBS_DIMS = {
    "rates": 3, "acc": 3, "att": 2, "flight_plan": trk.FLIGHT_PLAN_DIM,
    "info_p_w": 3, "info_p_g": 3, "info_v_w": 3, "info_v_g": 3, "info_att": 4, "info_rates": 3,
    "info_att_err": 2,
    "info_thrust": 1, "info_acc": 3, "info_cam": 3, "info_dyn": N_DYN, "info_delay": 2,
    "info_meas_rates": 3, "info_meas_acc": 3, "info_meas_att": 2,
    "info_flight_plan": trk.FLIGHT_PLAN_DIM,
}
LOG_KEYS = ("diverged", "hit_gate", "hit_ground", "rate_l1", "speed", "att_err_deg")


def step(state: EnvState, action: jax.Array, ccfg: CtbaConfig):
    cfg = ccfg.base
    key, *k = jax.random.split(state.key, 8)
    cmd = action_to_command(action)

    # --- action delay: buf holds the last ACTION_DELAY_MAX commands, oldest first
    buf = state.action_buf
    exec_cmd = buf[P.ACTION_DELAY_MAX - state.act_delay]
    action_buf = jnp.concatenate([buf[1:], cmd[None]], 0)

    # --- disturbances
    resample_a = jax.random.uniform(k[0]) < RESAMPLE_P
    resample_m = jax.random.uniform(k[1]) < RESAMPLE_P
    eps_a = jnp.where(resample_a, jax.random.uniform(k[2], (3,), minval=-state.eps_a_amp,
                                                     maxval=state.eps_a_amp), state.eps_a)
    eps_m_slow = jnp.where(resample_m, jax.random.uniform(k[3], (3,), minval=-EPS_M_SLOW,
                                                          maxval=EPS_M_SLOW), state.eps_m_slow)
    eps_m = eps_m_slow + jax.random.uniform(k[4], (3,), minval=-EPS_M_FAST, maxval=EPS_M_FAST)
    dist = plant.Disturbance(eps_a=eps_a, eps_m=eps_m)

    kn = jax.random.split(k[6], 3)
    noise = plant.ImuNoise(acc_bias=state.acc_bias,
                           acc=state.params.acc_sigma * jax.random.normal(kn[0], (3,)),
                           gyro=0.5 * state.params.gyro_sigma * jax.random.normal(kn[1], (3,)))
    s_next = plant.step(state.s, exec_cmd[0], exec_cmd[1:3], exec_cmd[3], state.params, dist,
                        noise, grounded=state.grounded)
    grounded, airborne, launch_crash = state.grounded, state.airborne, jnp.array(False)
    if cfg.launch is not None:
        s_next, grounded, airborne, launch_crash = SE._floor_contact(state, s_next, cfg)
        # resting on the floor: the floor carries the weight, the accelerometer reads 1 g up
        R = quat_to_matrix(s_next.q)
        rest_f = R.T @ jnp.array([0.0, 0.0, -1.0])
        s_next = s_next._replace(
            acc_f=jnp.where(grounded, rest_f, s_next.acc_f),
            omega_d=jnp.where(grounded, 0.0, s_next.omega_d))

    # --- gates, reward, termination: skydreamer.env.step verbatim in substance
    target, _, _ = trk.plane_pose(cfg.track, state.plane, state.t_g)
    gate = (state.plane // trk.PLANES_PER_GATE) % cfg.track.n_gates
    yaw = cfg.track.yaw[gate]
    p_plane = trk.gate_rotation(yaw) @ (s_next.p - target)

    armed = state.armed | (p_plane[0] < 0.0)
    passed = armed & (p_plane[0] > 0.0)
    r_cross = trk.gate_reward(p_plane, state.d_g)
    struck = trk.gate_collision(p_plane, state.d_g)
    if cfg.swept_points is not None:
        half = cfg.track.inner[gate] / 2
        worst = trk.swept_offset(s_next.p, s_next.v, quat_to_matrix(s_next.q),
                                 jnp.asarray(cfg.swept_points, jnp.float32),
                                 cfg.track.pos[gate], yaw)
        real = (state.plane % trk.PLANES_PER_GATE) == 1
        r_cross = jnp.where(real, 1.0 - worst / half, r_cross)
        struck = jnp.where(real, worst > half, struck)
    r_gate = jnp.where(passed, r_cross, 0.0)
    hit_gate = passed & struck

    dist_now = SE._progress_distance(cfg, s_next.p, state.plane, state.t_g)
    active = (state.plane % trk.PLANES_PER_GATE) == 0
    r_prog = jnp.where(active, state.prev_dist - dist_now, 0.0)

    euler = quat_to_euler(s_next.q)
    r_pen = trk.rate_penalty(s_next.omega_b, P.CONTROL_FREQ)
    if cfg.v_cap is not None:
        r_pen = r_pen + trk.speed_penalty(s_next.v, cfg.v_cap, cfg.w_speed)
    if cfg.w_floor:
        r_pen = r_pen + trk.floor_penalty(s_next.p, s_next.v, euler, *cfg.floor, cfg.w_floor)
    alt_ok = state.alt_ok
    if cfg.w_alt:
        lowest = trk.lowest_point(s_next.p, euler, cfg.floor[0])
        alt_ok = alt_ok | (lowest >= cfg.alt_margin)
        r_pen = r_pen + jnp.where(alt_ok, trk.altitude_penalty(lowest, cfg.alt_margin, cfg.w_alt), 0.0)
    reward = trk.W_PROG * r_prog - r_pen + trk.W_GATE * r_gate

    hit_ground = trk.floor_collision(s_next.p, euler, *cfg.floor)
    if cfg.launch is not None:
        hit_ground = jnp.where(airborne, hit_ground, launch_crash)
    blew_up = trk.diverged(s_next.omega_b, s_next.p, s_next.v)
    hit_wall = trk.left_room(s_next.p, cfg.room) if cfg.room is not None else jnp.array(False)
    terminal = hit_gate | hit_ground | blew_up | hit_wall
    term_cause = jnp.where(hit_gate, 1, jnp.where(hit_ground, 2, jnp.where(
        blew_up, 3, jnp.where(hit_wall, 4, 0)))).astype(jnp.int32)
    reward = jnp.where(terminal, 0.0, reward)

    plane = state.plane + passed.astype(jnp.int32)
    sub = state.plane % trk.PLANES_PER_GATE
    p_inc = jnp.array([1.0 / 3.0, 0.5, 1.0])[sub]
    do_inc = passed & (~state.fp_done) & (jax.random.uniform(k[5]) < p_inc)
    fp_index = (state.fp_index + do_inc.astype(jnp.int32)) % cfg.track.n_gates
    fp_done = (state.fp_done | do_inc) & ~(passed & (sub == 2))

    next_target, next_yaw, _ = trk.plane_pose(cfg.track, plane, state.t_g)
    prev_dist = jnp.where(passed, SE._progress_distance(cfg, s_next.p, plane, state.t_g), dist_now)
    p_next = trk.gate_rotation(next_yaw) @ (s_next.p - next_target)
    armed = jnp.where(passed, p_next[0] < 0.0, armed)

    new = state._replace(
        key=key, s=s_next, eps_a=eps_a, eps_m_slow=eps_m_slow,
        erode=augment.resample_erosion(kn[2], state.erode, cfg.erosion_prob),
        plane=plane, armed=armed, fp_index=fp_index, fp_done=fp_done, prev_dist=prev_dist,
        step_count=state.step_count + 1, done=terminal, action_buf=action_buf,
        term_cause=term_cause, grounded=grounded, airborne=airborne, alt_ok=alt_ok)

    mask = SE._render_augmented(new, cfg)
    new = new._replace(mask_buf=jnp.concatenate([new.mask_buf[1:], mask[None]], 0))
    truncated = new.step_count >= cfg.max_steps
    return new, observe(new, ccfg), reward, terminal, truncated


def gates_passed(state: EnvState) -> jax.Array:
    return (state.plane - state.plane0) // trk.PLANES_PER_GATE


def batched(ccfg: CtbaConfig):
    reset_b = jax.jit(jax.vmap(lambda k: reset(k, ccfg)))
    step_b = jax.jit(jax.vmap(lambda s, a: step(s, a, ccfg)))
    return reset_b, step_b
