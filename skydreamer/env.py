"""The SkyDreamer informed-POMDP racing environment.

Pure-functional JAX: `reset` and `step` are jittable and vmap over a batch of
environments, so a whole rollout batch advances without leaving the accelerator.

Observation `o_t`   -> keys without a prefix; what the policy actually sees.
Information `i_t`   -> keys prefixed `info_`; decoder targets only.

Per the paper, i_t = {o_t, o_t^+} \\ {X}: everything except the image, because
the image's content is already implied by the privileged state and decoding it
would slow training down for nothing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import augment, track as trk
from .dynamics import (
    Disturbance,
    State,
    euler_to_quat,
    initial_motor_speeds,
    quat_to_euler,
    quat_to_matrix,
    step as dyn_step,
)
from .params import (
    EVAL,
    NOMINAL,
    TRAIN,
    DynParams,
    sample_extrinsics,
    sample_params,
)
from .render import fov_valid, render_mask
from .track import Track

IMAGE_SIZE = 64

# Section III-A: "we simulate an image delay of 33 ms and an action delay of
# 11 ms to match deployment conditions".  At an 11.0 ms control period that is
# 3 and 1 steps respectively.
IMAGE_DELAY_STEPS = 3
ACTION_DELAY_STEPS = 1

# Section III-A: 70% of episodes use the training column of Table III and may
# start in front of any gate; 30% use the evaluation column and always start at
# the start gate.
TRAIN_FRACTION = 0.7

# Evaluation flights in the paper run 2000 steps (~22 s).
MAX_STEPS = 2000

# How far inside the walls, ceiling and floor a start position is pulled
# (`EnvConfig.room`): about one airframe plus a little.
ROOM_MARGIN = 0.3

# NOT FROM THE PAPER -- but the paper insists the two exist separately.  The
# observation carries Omega-hat and omega-hat, "the body rates measured by the
# IMU" and "the propeller angular velocities measured by the ESC", while the
# privileged information carries Omega and omega, which "represent the
# ground-truth rates, not the measured ones".  Decoding both is only meaningful
# if they differ; at zero noise they are the same array, and the training log
# showed exactly that -- info_rates and info_meas_rates agreed to two decimals
# for 17M steps, so half the decoder was reconstructing the other half.
#
# The magnitude is ours.  On a 5in racer the gyro is dominated by airframe
# vibration, not by the MEMS floor: an ICM-42688-class part contributes about
# 0.05 deg/s over a 45 Hz band, while post-filter vibration noise runs 1-10
# deg/s RMS and varies with prop balance, frame stiffness and IMU mounting.
# That is a per-airframe quantity, which is precisely what Table III randomizes
# everything else for -- so the sigma is drawn per episode rather than fixed,
# and the range includes zero so the paper-literal setting stays inside the
# training distribution.
GYRO_NOISE_RANGE = (0.1, 0.5)  # rad/s (5.7 - 29 deg/s), sigma drawn per episode
# Bidirectional DShot RPM telemetry quantises and jitters at roughly 1-2% of
# the reading.  Fixed rather than randomized: it is a property of the protocol,
# not of the airframe.
RPM_NOISE_FRAC = 0.01  # of w_max

# The mask is binary but stored one byte per pixel, and DreamerV3 keeps the
# replay buffer uncompressed in RAM: 4096 of the 4596 bytes/step carry 4096
# *bits* of information.  Packing them 8-to-a-byte turns the paper's 10e6-step
# buffer from 46 GB into 10 GB.  The agent unpacks before the encoder (see
# `packbits` in patches/informed_dreamer.patch), so the network sees
# bit-identical input -- this is storage, not modelling.
PACK_MASK = True


class EnvConfig(NamedTuple):
    track: Track
    max_steps: int = MAX_STEPS
    train_fraction: float = TRAIN_FRACTION
    image_size: int = IMAGE_SIZE
    gyro_noise_range: tuple[float, float] = GYRO_NOISE_RANGE
    rpm_noise_frac: float = RPM_NOISE_FRAC
    # Section III-D perturbs w_max by +-300 rad/s, resampled every 10 steps,
    # "to account for imperfect actuator response modeling".  Big track only.
    w_max_disturbance: float = 0.0
    w_max_resample_every: int = 10
    # Table III sets t_g = 0.8 m, but section III-B trains the ladder inverted
    # loop with "a smaller tunnel size t_g = 0.3 m, to further demonstrate
    # SkyDreamer's ability to execute tight maneuvers" -- and that is the run
    # the paper's simulation figures come from.  None = use Table III.
    t_g: float | None = None
    # Camera position along body x, metres ahead of the centre of mass.  The
    # paper's camera sits at the CoM; ours is ~0.10 m forward.  0 = paper.
    cam_offset: float = 0.0
    # (train, eval) gate window d_g in metres, replacing Table III's (0.8, 1.0).
    d_g: tuple[float, float] | None = None
    # Real camera (horizontal, vertical) field of view in degrees.  The policy's
    # image is always on the nominal ~104 degree square K; a narrower camera leaves
    # blank borders, applied here after augmentation.  None = fills K (the paper).
    fov: tuple[float, float] | None = None
    # Walls and ceiling, (x_lo, x_hi, y_lo, y_hi, ceiling altitude) in NED metres;
    # leaving them ends the episode like a crash.  Start positions are pulled in
    # to ROOM_MARGIN inside.  None = open field (the paper).
    room: tuple[float, float, float, float, float] | None = None
    # Speed limit in m/s, enforced by a reward penalty (track.speed_penalty).
    # None = none (the paper).
    v_cap: float | None = None
    # Its penalty per m/s over, per step.
    w_speed: float = trk.W_SPEED
    # (reach, clearance) in m: crash when the airframe comes within `clearance` of
    # the floor (track.floor_collision).  None = section II-C's shaped test.
    floor: tuple[float, float] | None = None
    # The evaluation column's start on the floor in front of the start gate
    # (track.LAUNCH): (distance lo, hi, lateral half-range, yaw half-range deg,
    # resting centre altitude).  Needs `floor`.  None = Table III's mid-air start.
    launch: tuple[float, float, float, float, float] | None = None
    # Lowest altitude of a mid-air start, m; None = only the room clip.
    air_start_alt: float | None = None
    # track.floor_penalty's weight; 0 = off.  Needs `floor`.
    w_floor: float = 0.0
    # Chance that the 1-pixel erosion (augment.erode1) is on, drawn at reset and
    # re-drawn every ~EROSION_HOLD_STEPS.  Paper II-F: 0.5.
    erosion_prob: float = augment.EROSION_PROB
    # How the erosion thins the mask: None = section II-F's 1-pixel erosion of the
    # finished 64 x 64 mask (augment.erode1); a number = every frame edge that many
    # pixels thinner, in the renderer (track.ERODE_PX: our 0.15 m frame).
    erode_px: float | None = None
    # Body-frame points covering the airframe, (M, 3) (hardware.footprint): if set,
    # the real gate plane (the middle one) is scored and its collision tested on the
    # furthest of them against the opening (track.swept_offset), as the physical
    # evaluation does, rather than on the centre of mass against d_g.  The virtual
    # pre/post planes keep d_g.  None = the paper.
    swept_points: np.ndarray | None = None
    # track.altitude_penalty: the margin in m for the lowest prop tip, and the weight
    # (0 = off).  Needs `floor` (for the reach).
    alt_margin: float = trk.ALT_MARGIN
    w_alt: float = 0.0
    # track.gate_margin_penalty at the real plane, on the airframe's worst point
    # (0 = off).  Needs `swept_points`.
    gate_margin: float = trk.GATE_MARGIN
    w_gate_margin: float = 0.0
    # track.launch_tip_penalty: the lowest prop tip under `launch_tip` m before the
    # airframe is clear of the floor (0 = off).  Needs `launch` and `floor`.
    launch_tip: float = trk.LAUNCH_TIP_BUFFER
    w_launch_tip: float = 0.0


class EnvState(NamedTuple):
    key: jax.Array
    s: State
    params: DynParams
    c_e: jax.Array  # (3,) camera extrinsics
    d_g: jax.Array  # effective gate half-size for this episode
    t_g: jax.Array  # pre..post gate thickness for this episode
    bounds: jax.Array  # (4,) [eps_a, eps_M_slow, eps_M_fast, eps_u] amplitudes
    eps_a: jax.Array  # (3,) held slow acceleration disturbance
    eps_M_slow: jax.Array  # (3,) held slow moment disturbance
    rs_s: jax.Array  # rolling-shutter strength for this episode
    gyro_sigma: jax.Array  # this airframe's gyro noise level
    w_max_offset: jax.Array  # III-D actuator disturbance, held for N steps
    erode: jax.Array  # bool, current erosion level
    plane: jax.Array  # int32, monotone index into the gate planes (wraps per lap)
    plane0: jax.Array  # int32, plane the episode started on
    # True once the drone has been *behind* the current target plane.  A gate
    # is a window, not a half-space, and `p_plane[0] > 0` alone cannot say so.
    # On the inverted loop that broke the lap outright: gates 0 and 2 both face
    # +y with gate 2 behind, so finishing gate 2 at y = +0.4 already satisfies
    # gate 0's `y > -0.4` and the drone was scored as crossing gate 0's plane
    # 4.5 m off centre -- an instant gate strike, every lap, for any policy.
    # It is why no evaluation episode ever passed more than 2 gates and why a
    # 5-lap success was unreachable by construction.
    armed: jax.Array  # bool
    fp_index: jax.Array  # int32, flight-plan gate index (lags `plane` randomly)
    fp_done: jax.Array  # bool, whether fp_index already advanced for this gate
    prev_dist: jax.Array  # distance to the current target plane, previous step
    step_count: jax.Array
    done: jax.Array
    action_buf: jax.Array  # (ACTION_DELAY_STEPS, 4)
    mask_buf: jax.Array  # (IMAGE_DELAY_STEPS + 1, H, W)
    # Why the episode ended, for diagnostics only -- nothing in the training
    # loop reads it.  `done` is one bool, which is all the agent needs but
    # leaves "it crashed" unanswerable: flying into a gate, mushing into the
    # floor and tumbling out of control are different failures with different
    # fixes.  0 = still flying, 1 = gate, 2 = ground, 3 = divergence,
    # 4 = wall or ceiling (`EnvConfig.room`).
    term_cause: jax.Array
    # `EnvConfig.launch`: resting on the floor (the floor holds it up), and whether
    # it has climbed clear since -- until then the floor is something it stands
    # on, not `floor`'s crash clearance.  False / True for a mid-air start.
    grounded: jax.Array
    airborne: jax.Array
    # Whether the lowest prop tip has been above `EnvConfig.alt_margin` this
    # episode; track.altitude_penalty only charges once it has, so a launch climbs
    # through the margin for free.
    alt_ok: jax.Array


def _mix(a: float, b: float, use_train: jax.Array) -> jax.Array:
    return jnp.where(use_train, a, b)


def _episode_bounds(use_train: jax.Array):
    """Pick each Table III quantity from the train or eval column."""
    return dict(
        eps_a=_mix(TRAIN.eps_a_slow, EVAL.eps_a_slow, use_train),
        eps_M_slow=_mix(TRAIN.eps_M_slow, EVAL.eps_M_slow, use_train),
        eps_M_fast=_mix(TRAIN.eps_M_fast, EVAL.eps_M_fast, use_train),
        eps_u=_mix(TRAIN.eps_u_fast, EVAL.eps_u_fast, use_train),
        d_g=_mix(TRAIN.d_g, EVAL.d_g, use_train),
        t_g=_mix(TRAIN.t_g, EVAL.t_g, use_train),
        z_lo=_mix(TRAIN.init_z_g[0], EVAL.init_z_g[0], use_train),
    )


def _sample_params_mixed(key: jax.Array, use_train: jax.Array) -> DynParams:
    """Table III randomizes `d` by +-30% (train) or +-20% (eval), except the
    motor limits which are always +-20%.  Sample once in [-1, 1] and scale, so
    the two columns stay on the same underlying draw."""
    p_train = sample_params(key, TRAIN)
    p_eval = sample_params(key, EVAL)
    return jax.tree.map(lambda a, b: jnp.where(use_train, a, b), p_train, p_eval)


def reset(key: jax.Array, cfg: EnvConfig) -> tuple[EnvState, dict]:
    k = jax.random.split(key, 12)
    use_train = jax.random.uniform(k[0]) < cfg.train_fraction
    b = _episode_bounds(use_train)

    if cfg.t_g is not None:
        b["t_g"] = jnp.asarray(float(cfg.t_g))
    if cfg.d_g is not None:
        b["d_g"] = _mix(cfg.d_g[0], cfg.d_g[1], use_train)

    params = _sample_params_mixed(k[1], use_train)
    c_e = sample_extrinsics(k[2], TRAIN)  # identical ranges in both columns

    n = cfg.track.n_gates
    start_gate = jnp.where(use_train, jax.random.randint(k[3], (), 0, n), 0)

    gate_pos = cfg.track.pos[start_gate]
    gate_yaw = cfg.track.yaw[start_gate]
    R_wg = trk.gate_rotation(gate_yaw).T  # gate -> world

    ks = jax.random.split(k[4], 6)
    x_g = jax.random.uniform(ks[0], minval=TRAIN.init_x_g[0], maxval=TRAIN.init_x_g[1])
    y_g = jax.random.uniform(ks[1], minval=TRAIN.init_y_g[0], maxval=TRAIN.init_y_g[1])
    z_lo, z_hi = b["z_lo"], TRAIN.init_z_g[1]
    if cfg.air_start_alt is not None:
        # Table III's 0-1.3 m below a gate whose centre is 0.75 m up is under the
        # floor; draw from the part of the range that is at least this high.
        z_hi = jnp.minimum(z_hi, -gate_pos[2] - cfg.air_start_alt)
        z_lo = jnp.minimum(z_lo, z_hi)
    z_g = jax.random.uniform(ks[2], minval=z_lo, maxval=z_hi)
    rates = jax.random.uniform(ks[3], (3,), minval=TRAIN.init_rate[0], maxval=TRAIN.init_rate[1])
    motor_frac = jax.random.uniform(
        ks[4], (4,), minval=TRAIN.init_motor[0], maxval=TRAIN.init_motor[1]
    )
    att = jax.random.uniform(ks[5], (3,), minval=TRAIN.init_att[0], maxval=TRAIN.init_att[1])

    p_w = gate_pos + R_wg @ jnp.stack([x_g, y_g, z_g])
    if cfg.room is not None:
        # Table III starts 2-4 m before the gate, which in a small room can be
        # through the wall; the floor bound keeps a low gate's start off the ground.
        x_lo, x_hi, y_lo, y_hi, ceiling = cfg.room
        m = ROOM_MARGIN
        p_w = jnp.clip(p_w, jnp.array([x_lo + m, y_lo + m, -ceiling + m]),
                       jnp.array([x_hi - m, y_hi - m, -m]))
    q = euler_to_quat(att[0], att[1], gate_yaw + att[2])
    motor = initial_motor_speeds(motor_frac, params)

    launch = jnp.array(False)
    if cfg.launch is not None:
        # The real start: at rest on the floor, level, motors at idle, facing the
        # start gate from `d` m out.  Its own key, so the other draws are unchanged.
        d_lo, d_hi, y_half, yaw_half, rest = cfg.launch
        kl = jax.random.split(jax.random.fold_in(k[4], 1), 3)
        d = jax.random.uniform(kl[0], minval=d_lo, maxval=d_hi)
        y_l = jax.random.uniform(kl[1], minval=-y_half, maxval=y_half)
        yaw_l = jax.random.uniform(kl[2], minval=-yaw_half, maxval=yaw_half) * jnp.pi / 180.0
        p_l = (gate_pos + R_wg @ jnp.stack([-d, y_l, 0.0])).at[2].set(-rest)
        launch = ~use_train
        p_w = jnp.where(launch, p_l, p_w)
        q = jnp.where(launch, euler_to_quat(0.0, 0.0, gate_yaw + yaw_l), q)
        rates = jnp.where(launch, 0.0, rates)
        motor = jnp.where(launch, params.w_min * jnp.ones(4), motor)

    s = State(
        p=p_w,
        v=jnp.zeros(3),
        q=q,
        omega_b=rates,
        motor=motor,
    )

    plane = start_gate * trk.PLANES_PER_GATE
    target, target_yaw, _ = trk.plane_pose(cfg.track, plane, b["t_g"])
    armed0 = (trk.gate_rotation(target_yaw) @ (p_w - target))[0] < 0.0

    state = EnvState(
        key=k[5],
        s=s,
        params=params,
        c_e=c_e,
        d_g=b["d_g"],
        t_g=b["t_g"],
        bounds=jnp.stack([b["eps_a"], b["eps_M_slow"], b["eps_M_fast"], b["eps_u"]]),
        eps_a=jax.random.uniform(k[6], (3,), minval=-b["eps_a"], maxval=b["eps_a"]),
        eps_M_slow=jax.random.uniform(k[7], (3,), minval=-b["eps_M_slow"], maxval=b["eps_M_slow"]),
        rs_s=jax.random.uniform(
            k[8], minval=augment.ROLLING_SHUTTER_RANGE[0], maxval=augment.ROLLING_SHUTTER_RANGE[1]
        ),
        erode=jax.random.uniform(k[9]) < cfg.erosion_prob,
        gyro_sigma=jax.random.uniform(
            k[10], minval=cfg.gyro_noise_range[0], maxval=cfg.gyro_noise_range[1]
        ),
        w_max_offset=jax.random.uniform(
            k[11], minval=-cfg.w_max_disturbance, maxval=cfg.w_max_disturbance
        ),
        plane=plane,
        plane0=plane,
        armed=armed0,
        fp_index=start_gate,
        fp_done=jnp.array(False),
        prev_dist=jnp.linalg.norm(p_w - target),
        step_count=jnp.array(0, jnp.int32),
        done=jnp.array(False),
        action_buf=jnp.zeros((ACTION_DELAY_STEPS, 4)),
        # D + 1 entries: the newest frame is pushed before the observation is
        # read, so index 0 is exactly D steps old.  Sizing this D would give a
        # D-1 step delay -- 22 ms instead of the paper's 33 ms.
        mask_buf=jnp.zeros((IMAGE_DELAY_STEPS + 1, cfg.image_size, cfg.image_size)),
        term_cause=jnp.array(0, jnp.int32),
        grounded=launch,
        airborne=~launch,
        alt_ok=jnp.array(False),
    )
    # Prime the image buffer so the first observations are consistent rather
    # than blank; the real drone likewise has valid frames before it launches.
    mask = _render_augmented(state, cfg)
    state = state._replace(mask_buf=jnp.broadcast_to(mask, state.mask_buf.shape))
    return state, observe(state, cfg)


def _render_augmented(state: EnvState, cfg: EnvConfig) -> jax.Array:
    p_cam = state.s.p
    if cfg.cam_offset:
        p_cam = p_cam + quat_to_matrix(state.s.q)[..., :, 0] * cfg.cam_offset
    shrink = 0.0 if cfg.erode_px is None else jnp.where(state.erode, cfg.erode_px, 0.0)
    mask = render_mask(
        p_cam, state.s.q, state.c_e, cfg.track, cfg.image_size, cfg.image_size, shrink
    )
    erode = state.erode if cfg.erode_px is None else jnp.array(False)
    out = augment.apply(mask, state.s.omega_b, state.c_e, state.rs_s, erode)
    if cfg.fov is not None:
        out = out * fov_valid(cfg.image_size, cfg.image_size, cfg.fov)
    return out


def _gate_relative(state: EnvState, cfg: EnvConfig):
    """Position and velocity in the frame of the gate currently being flown."""
    gate = (state.plane // trk.PLANES_PER_GATE) % cfg.track.n_gates
    yaw = cfg.track.yaw[gate]
    R = trk.gate_rotation(yaw)
    p_g = R @ (state.s.p - cfg.track.pos[gate])
    v_g = R @ state.s.v
    return p_g, v_g, yaw


def pack_mask(mask: jax.Array) -> jax.Array:
    """(H, W) in {0,1} -> (H, W//8, 1) uint8, eight pixels per byte."""
    h, w = mask.shape
    bits = (mask > 0.5).astype(jnp.uint8).reshape(h, w // 8, 8)
    weights = (1 << jnp.arange(7, -1, -1)).astype(jnp.uint32)
    return (bits.astype(jnp.uint32) * weights).sum(-1).astype(jnp.uint8)[..., None]


def unpack_mask(packed: jax.Array) -> jax.Array:
    """Inverse of `pack_mask`, for tests and visualisation."""
    shifts = jnp.arange(7, -1, -1, dtype=jnp.uint8)
    bits = ((packed[..., None] >> shifts) & 1).astype(jnp.uint8)
    bits = jnp.moveaxis(bits, -2, -1)
    h, w8, c = packed.shape
    return bits.reshape(h, w8 * 8, c)[..., 0].astype(jnp.float32)


def observe(state: EnvState, cfg: EnvConfig) -> dict:
    k1, k2 = jax.random.split(jax.random.fold_in(state.key, state.step_count), 2)

    omega_meas = state.s.omega_b + state.gyro_sigma * jax.random.normal(k1, (3,))
    rpm_meas = state.s.motor + (
        cfg.rpm_noise_frac * NOMINAL["w_max"] * jax.random.normal(k2, (4,))
    )
    rpm_norm = rpm_meas / NOMINAL["w_max"]

    fp = trk.flight_plan(cfg.track, state.fp_index)
    p_g, v_g, gate_yaw = _gate_relative(state, cfg)
    euler = quat_to_euler(state.s.q)

    return {
        # --- observation o_t: what the policy sees at deployment ---
        "mask": (
            pack_mask(state.mask_buf[0]) if PACK_MASK
            else state.mask_buf[0][..., None].astype(jnp.uint8) * 255
        ),
        "rates": omega_meas,
        "rpm": rpm_norm,
        "flight_plan": fp,
        # --- information i_t: decoder targets, training only ---
        "info_p_w": state.s.p,
        "info_p_g": p_g,
        "info_v_w": state.s.v,
        "info_v_g": v_g,
        "info_att": jnp.concatenate(
            [euler, trk.wrap_pi(euler[2:3] - gate_yaw)]
        ),  # [phi_w, theta_w, psi_w, psi_g]
        "info_rates": state.s.omega_b,
        "info_rpm": state.s.motor / NOMINAL["w_max"],
        "info_cam": state.c_e,
        "info_dyn": state.params.as_vector(),
        "info_meas_rates": omega_meas,
        "info_meas_rpm": rpm_norm,
        "info_flight_plan": fp,
        # --- diagnostics only.  embodied strips `log/*` before the agent sees
        # it (core/driver.py) and before it reaches the replay buffer
        # (core/replay.py); run/train.py aggregates it per episode instead.  So
        # these are free: they change nothing about the model, and they turn a
        # 60-hour run from blind into one where the failure mode is visible
        # while it happens rather than only at the evaluation afterwards.
        "log/diverged": (state.term_cause == 3).astype(jnp.float32),
        "log/hit_gate": (state.term_cause == 1).astype(jnp.float32),
        # walls and ceiling count here too: a new log key would change the
        # observation space the existing checkpoints were built with
        "log/hit_ground": ((state.term_cause == 2) | (state.term_cause == 4)).astype(jnp.float32),
        "log/rate_l1": jnp.sum(jnp.abs(state.s.omega_b)).astype(jnp.float32),
        "log/speed": jnp.linalg.norm(state.s.v).astype(jnp.float32),
    }


def step(state: EnvState, action: jax.Array, cfg: EnvConfig):
    # One key per consumer.  Reusing a key across two draws makes those draws
    # perfectly correlated, which silently couples unrelated randomness.
    key, *k = jax.random.split(state.key, 10)
    action = jnp.clip(action, 0.0, 1.0)

    # --- action delay: what leaves the policy now executes ACTION_DELAY steps later
    u_exec = state.action_buf[0]
    action_buf = jnp.concatenate([state.action_buf[1:], action[None]], axis=0)

    # --- disturbances (Table III).  Slow terms resample with p = 1/100 per
    # step ("1 Hz"), fast terms every step ("90 Hz").
    a_amp, m_slow_amp, m_fast_amp, u_amp = state.bounds
    resample_a = jax.random.uniform(k[0]) < 0.01
    resample_m = jax.random.uniform(k[1]) < 0.01
    eps_a = jnp.where(
        resample_a, jax.random.uniform(k[2], (3,), minval=-a_amp, maxval=a_amp), state.eps_a
    )
    eps_M_slow = jnp.where(
        resample_m,
        jax.random.uniform(k[3], (3,), minval=-m_slow_amp, maxval=m_slow_amp),
        state.eps_M_slow,
    )
    eps_M = eps_M_slow + jax.random.uniform(k[4], (3,), minval=-m_fast_amp, maxval=m_fast_amp)
    eps_u = jax.random.uniform(k[5], (4,), minval=-u_amp, maxval=u_amp)
    dist = Disturbance(eps_a=eps_a, eps_M=eps_M, eps_u=eps_u)

    # III-D: "introduce disturbances of +-300 rad/s to w_max, randomly
    # resampled every 10 timesteps, to account for imperfect actuator response
    # modeling".  Zero amplitude on the small tracks, so this is a no-op there.
    resample_w = (state.step_count % cfg.w_max_resample_every) == 0
    w_max_offset = jnp.where(
        resample_w,
        jax.random.uniform(
            k[8], minval=-cfg.w_max_disturbance, maxval=cfg.w_max_disturbance
        ),
        state.w_max_offset,
    )
    params = state.params._replace(w_max=state.params.w_max + w_max_offset)

    s_next = dyn_step(state.s, u_exec, dist, params)
    grounded, airborne, launch_crash = state.grounded, state.airborne, jnp.array(False)
    if cfg.launch is not None:
        s_next, grounded, airborne, launch_crash = _floor_contact(state, s_next, cfg)

    # --- plane crossing, reward, termination
    target, _, _ = trk.plane_pose(cfg.track, state.plane, state.t_g)
    gate = (state.plane // trk.PLANES_PER_GATE) % cfg.track.n_gates
    yaw = cfg.track.yaw[gate]
    p_plane = trk.gate_rotation(yaw) @ (s_next.p - target)

    armed = state.armed | (p_plane[0] < 0.0)
    passed = armed & (p_plane[0] > 0.0)
    r_cross = trk.gate_reward(p_plane, state.d_g)
    struck = trk.gate_collision(p_plane, state.d_g)
    if cfg.swept_points is not None:
        # The real plane: the airframe against the opening, as hardware.check_flight.
        half = cfg.track.inner[gate] / 2
        worst = trk.swept_offset(s_next.p, s_next.v, quat_to_matrix(s_next.q),
                                 jnp.asarray(cfg.swept_points, jnp.float32),
                                 cfg.track.pos[gate], yaw)
        real = (state.plane % trk.PLANES_PER_GATE) == 1
        r_real = 1.0 - worst / half
        if cfg.w_gate_margin:
            r_real = r_real - trk.gate_margin_penalty(worst, cfg.gate_margin, cfg.w_gate_margin) / trk.W_GATE
        r_cross = jnp.where(real, r_real, r_cross)
        struck = jnp.where(real, worst > half, struck)
    r_gate = jnp.where(passed, r_cross, 0.0)
    hit_gate = passed & struck

    dist_now = jnp.linalg.norm(s_next.p - target)
    # II-C: progress is only shaped toward the *pre*-gate; it is zeroed once the
    # drone is inside the pre..post volume (sub-plane 1 or 2).
    active = (state.plane % trk.PLANES_PER_GATE) == 0
    r_prog = jnp.where(active, state.prev_dist - dist_now, 0.0)

    euler = quat_to_euler(s_next.q)
    r_rate = trk.rate_penalty(s_next.omega_b)
    if cfg.v_cap is not None:
        r_rate = r_rate + trk.speed_penalty(s_next.v, cfg.v_cap, cfg.w_speed)
    if cfg.w_floor:
        r_rate = r_rate + trk.floor_penalty(s_next.p, s_next.v, euler, *cfg.floor, cfg.w_floor)
    alt_ok = state.alt_ok
    if cfg.w_alt:
        lowest = trk.lowest_point(s_next.p, euler, cfg.floor[0])
        alt_ok = alt_ok | (lowest >= cfg.alt_margin)
        r_rate = r_rate + jnp.where(
            alt_ok, trk.altitude_penalty(lowest, cfg.alt_margin, cfg.w_alt), 0.0)
    if cfg.w_launch_tip and cfg.launch is not None:
        tip = trk.lowest_point(s_next.p, euler, cfg.floor[0])
        r_rate = r_rate + jnp.where(
            airborne, 0.0, trk.launch_tip_penalty(tip, cfg.launch_tip, cfg.w_launch_tip))
    reward = trk.W_PROG * r_prog - r_rate + trk.W_GATE * r_gate

    if cfg.floor is not None:
        hit_ground = trk.floor_collision(s_next.p, euler, *cfg.floor)
        if cfg.launch is not None:
            hit_ground = jnp.where(airborne, hit_ground, launch_crash)
    else:
        hit_ground = trk.ground_collision(s_next.p, s_next.v, euler)
    blew_up = trk.diverged(s_next.omega_b, s_next.p, s_next.v)
    hit_wall = (trk.left_room(s_next.p, cfg.room) if cfg.room is not None
                else jnp.array(False))
    terminal = hit_gate | hit_ground | blew_up | hit_wall
    # Diagnostics only; checked in this order so a gate strike that also drives
    # the drone into the floor is reported as the gate strike that caused it.
    term_cause = jnp.where(
        hit_gate, 1, jnp.where(hit_ground, 2, jnp.where(blew_up, 3, jnp.where(hit_wall, 4, 0)))
    ).astype(jnp.int32)
    reward = jnp.where(terminal, 0.0, reward)  # II-C: terminal reward is zero

    plane = state.plane + passed.astype(jnp.int32)

    # II-H: during training the flight-plan index increments at a *random* point
    # between the pre- and post-gate, so the policy never learns to rely on it
    # firing exactly at the gate.  Uniform over the three crossings.
    sub = state.plane % trk.PLANES_PER_GATE
    p_inc = jnp.array([1.0 / 3.0, 0.5, 1.0])[sub]
    do_inc = passed & (~state.fp_done) & (jax.random.uniform(k[6]) < p_inc)
    fp_index = (state.fp_index + do_inc.astype(jnp.int32)) % cfg.track.n_gates
    fp_done = (state.fp_done | do_inc) & ~(passed & (sub == 2))

    next_target, next_yaw, _ = trk.plane_pose(cfg.track, plane, state.t_g)
    prev_dist = jnp.where(passed, jnp.linalg.norm(s_next.p - next_target), dist_now)
    p_next = trk.gate_rotation(next_yaw) @ (s_next.p - next_target)
    armed = jnp.where(passed, p_next[0] < 0.0, armed)

    new = state._replace(
        key=key,
        s=s_next,
        eps_a=eps_a,
        eps_M_slow=eps_M_slow,
        erode=augment.resample_erosion(k[7], state.erode, cfg.erosion_prob),
        w_max_offset=w_max_offset,
        plane=plane,
        armed=armed,
        fp_index=fp_index,
        fp_done=fp_done,
        prev_dist=prev_dist,
        step_count=state.step_count + 1,
        done=terminal,
        action_buf=action_buf,
        term_cause=term_cause,
        grounded=grounded,
        airborne=airborne,
        alt_ok=alt_ok,
    )

    # --- image delay: the policy sees a frame IMAGE_DELAY_STEPS old
    mask = _render_augmented(new, cfg)
    new = new._replace(mask_buf=jnp.concatenate([new.mask_buf[1:], mask[None]], axis=0))

    truncated = new.step_count >= cfg.max_steps
    return new, observe(new, cfg), reward, terminal, truncated


def _floor_contact(state: EnvState, s_next: State, cfg: EnvConfig):
    """The floor under a launched drone, until it first climbs clear of it.

    Below its resting height, level enough that no prop tip touches and slower
    than track.TOUCHDOWN_SPEED, the drone rests: the floor holds it, level, where
    it was (it does not slide or turn on the floor), motors still spinning.  A
    prop tip on the floor, or coming down faster, is a crash.  Clear means above
    `floor`'s clearance plus the airframe's reach, where `floor_collision` cannot
    fire at any tilt, and from then on that is the floor's only rule.

    Returns (state, grounded, airborne, crashed)."""
    reach, clearance = cfg.floor
    rest = cfg.launch[4]
    alt = -s_next.p[2]
    euler = quat_to_euler(s_next.q)
    tip = trk.lowest_point(s_next.p, euler, reach) < 0.0
    down = alt <= rest
    settle = ~state.airborne & down & ~tip & (s_next.v[2] <= trk.TOUCHDOWN_SPEED)
    p_rest = jnp.where(state.grounded, state.s.p, s_next.p).at[2].set(-rest)
    q_rest = jnp.where(state.grounded, state.s.q, euler_to_quat(0.0, 0.0, euler[2]))
    rested = s_next._replace(p=p_rest, v=jnp.zeros(3), q=q_rest, omega_b=jnp.zeros(3))
    s = jax.tree.map(lambda a, b: jnp.where(settle, a, b), rested, s_next)
    airborne = state.airborne | (alt > clearance + reach)
    crashed = ~airborne & ~settle & (tip | down)
    return s, settle, airborne, crashed


def gates_passed(state: EnvState) -> jax.Array:
    """How many gates have been fully completed since the episode started."""
    return (state.plane - state.plane0) // trk.PLANES_PER_GATE


def batched(cfg: EnvConfig):
    """Build jitted, vmapped `reset`/`step` for a fixed config.

    `cfg` is closed over rather than passed as an argument: it mixes traced
    arrays (the track) with values used as array shapes (`image_size`), so it
    is neither a clean pytree nor hashable as a static arg."""
    reset_b = jax.jit(jax.vmap(lambda k: reset(k, cfg)))
    step_b = jax.jit(jax.vmap(lambda s, a: step(s, a, cfg)))
    return reset_b, step_b
