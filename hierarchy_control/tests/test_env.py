"""The CTBA environment: shapes, determinism, physics sanity, demonstrations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ctba import env as E, expert as X, params as P, plant, tasks

TASK = "c1_team44up"


@pytest.fixture(scope="module")
def cfg():
    return tasks.make_config(TASK)


def test_every_stage_builds():
    for st in tasks.STAGES:
        tasks.make_config(f"{st}_team44up")


def test_obs_matches_declared_space(cfg):
    from ctba.embodied import Ctba
    _, obs = E.reset(jax.random.key(0), cfg)
    space = Ctba(TASK).obs_space
    for k, v in obs.items():
        assert k in space, k
        assert tuple(np.shape(v)) == tuple(space[k].shape), (k, np.shape(v), space[k].shape)
    for k in space:
        if k not in ("reward", "is_first", "is_last", "is_terminal"):
            assert k in obs, k


def test_deterministic(cfg):
    a = jnp.array([-0.4, 0.1, -0.1, 0.0])
    outs = []
    for _ in range(2):
        st, _ = E.reset(jax.random.key(3), cfg)
        for _ in range(5):
            st, o, r, *_ = E.step(st, a, cfg)
        outs.append((np.asarray(st.s.p), np.asarray(o["rates"]), float(r)))
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    np.testing.assert_array_equal(outs[0][1], outs[1][1])


def test_action_mapping_roundtrip():
    a = jnp.array([[-1.0, 0.5, -0.25, 1.0], [0.3, 0.0, 0.0, -1.0]])
    np.testing.assert_allclose(E.command_to_action(E.action_to_command(a)), a, atol=1e-6)
    c = E.action_to_command(jnp.array([1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_allclose(c, [1.0, P.ANGLE_LIMIT, P.ANGLE_LIMIT, P.YAW_RATE_MAX], rtol=1e-6)


def test_nominal_hover_holds():
    """At the identified hover throttle, level target: 1 s later within 0.3 m."""
    p = P.nominal()
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    thr = P.throttle_for(P.GRAVITY, P.battery_voltage(p, P.GRAVITY), 1.0)
    s = plant.initial(jnp.array([0.0, 0.0, -1.0]), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), P.GRAVITY, thr)
    for _ in range(100):
        s = plant.step(s, thr, jnp.zeros(2), 0.0, p, d)
    assert abs(float(s.p[2]) + 1.0) < 0.3
    np.testing.assert_allclose(np.asarray(s.acc_f), [0, 0, -1.0], atol=0.03)


def test_rate_step_response():
    """Below the level loop, roll step to 2 rad/s: about 90% within 40 ms (fn 18 Hz, zeta 0.5),
    pitch slower."""
    p = P.nominal()
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    s = plant.initial(jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), P.GRAVITY, 0.3)
    for _ in range(4):
        s = plant.step_rates(s, 0.3, jnp.array([2.0, 2.0, 2.0]), p, d)[0]
    w = np.asarray(s.omega_b)
    assert w[0] > 1.8 and w[1] < w[0] and 0.5 < w[2] < 1.9, w


def _hover_state(p):
    thr = P.throttle_for(P.GRAVITY, P.battery_voltage(p, P.GRAVITY), 1.0)
    s = plant.initial(jnp.array([0.0, 0.0, -2.0]), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), P.GRAVITY, thr)
    return s, thr


def test_angle_step_response():
    """ANGLE mode: 20 deg roll target.  Level gain 4.9/s -> ~0.2 s time constant: a third of
    the way at 0.1 s, within 1.5 deg at 1 s, no overshoot beyond 2 deg.  Estimator correction
    off (kp 0): with it on, the estimate is pulled (next test) and the airframe ends ~2 deg past."""
    from skydreamer.dynamics import quat_to_euler
    p = P.nominal()._replace(est_kp=jnp.float32(0.0))
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    s, thr = _hover_state(p)
    tgt = jnp.array([jnp.radians(20.0), 0.0])
    roll = []
    for _ in range(100):
        s = plant.step(s, thr, tgt, 0.0, p, d)
        roll.append(float(np.degrees(quat_to_euler(s.q)[0])))
    roll = np.array(roll)
    assert 4.0 < roll[9] < 12.0, roll[9]
    assert abs(roll[-1] - 20.0) < 1.5 and roll.max() < 22.0, (roll[-1], roll.max())


def test_estimator_recovers_and_is_pulled():
    """(a) a 3 deg initial estimate error at hover decays (kp 0.25: ~4 s time constant);
    (b) holding 25 deg roll at 1/cos(25) g (|acc| 1.10 g, inside the band) pulls the estimate
    toward level: the error grows -- the physical mechanism the simulator must contain."""
    p = P.nominal()
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    s, thr = _hover_state(p)
    s = s._replace(q_est=plant.converged_estimate(s.q, p, jnp.radians(jnp.array([3.0, 0.0]))))
    step = jax.jit(lambda s, thr, t: plant.step(s, thr, t, 0.0, p, d))
    for _ in range(1000):
        s = step(s, thr, jnp.zeros(2))
    err = np.degrees(np.asarray(plant.attitude_rp(s.q_est) - plant.attitude_rp(s.q)))
    assert np.abs(err).max() < 0.6, err

    s, _ = _hover_state(p)
    a = P.GRAVITY / np.cos(np.radians(24.0))
    thr = P.throttle_for(a, P.battery_voltage(p, a), 1.0)
    for _ in range(500):
        s = step(s, thr, jnp.array([jnp.radians(24.0), 0.0]))
    err = np.degrees(np.asarray(plant.attitude_rp(s.q_est) - plant.attitude_rp(s.q)))
    print("estimate error after 5 s at 24 deg roll:", err)
    assert err[0] < -1.0, err      # FC believes it is less tilted than it is


def test_floor_launch_rests_at_idle(cfg):
    c = cfg._replace(base=cfg.base._replace(train_fraction=0.0))
    st, _ = E.reset(jax.random.key(1), c)
    assert bool(st.grounded)
    a = jnp.array([-1.0, 0.0, 0.0, 0.0])   # throttle 0
    for _ in range(50):
        st, o, r, term, _ = E.step(st, a, c)
    assert bool(st.grounded) and not bool(term)
    np.testing.assert_allclose(np.asarray(o["acc"]), [0, 0, -1.0], atol=0.1)


def test_demonstrations_fly_the_track(cfg):
    ref = X.reference(cfg.base.track, 2.5)
    g, cause = X.screener(cfg, ref, 2.5, 0.8, 0.6, X.Gains())(jax.random.split(jax.random.key(0), 32))
    g, cause = np.asarray(g), np.asarray(cause)
    print("expert: gates", g, "causes", np.bincount(cause, minlength=5))
    assert np.mean((g >= 8) & (cause == 0)) > 0.4


def test_camera_frame_rate(cfg):
    """A new mask only arrives at the camera rate: over 60 steps of flight the
    newest buffered frame changes at most 60 / period + 1 times, and at least once."""
    st, _ = E.reset(jax.random.key(5), cfg)
    period = float(st.cam_period)
    lo, hi = cfg.camera.fps
    assert 100 / hi - 1e-4 <= period <= 100 / lo + 1e-4
    a = jnp.array([-0.3, 0.2, 0.1, 0.3])
    changes, last = 0, np.asarray(st.mask_buf[-1])
    for _ in range(60):
        st, *_ = E.step(st, a, cfg)
        cur = np.asarray(st.mask_buf[-1])
        changes += int(not np.array_equal(cur, last))
        last = cur
    assert changes <= 60 / period + 1, (changes, period)
    assert changes >= 1


def test_camera_field_and_launch_view(cfg):
    """The policy image spans the Pi Cam v2 full field (62.2 x 48.8 deg) at the 30 deg mount,
    and from the floor launch the start gate is in view."""
    from ctba import camera as CAM
    spec = cfg.camera
    assert spec.name == "picam_v2_1640x1232" and spec.tilt_deg == 30.0
    rays = np.asarray(CAM.pixel_rays(spec, 64, 64))
    h_edge = np.degrees(np.arctan(np.abs(rays[..., 1]).max()))
    v_edge = np.degrees(np.arctan(np.abs(rays[..., 2]).max()))
    assert abs(2 * h_edge - 62.2) < 2.0 and abs(2 * v_edge - 48.8) < 2.0, (h_edge, v_edge)
    c = cfg._replace(base=cfg.base._replace(train_fraction=0.0))
    st, o = E.reset(jax.random.key(2), c)
    assert float(np.asarray(st.mask_buf[-1]).mean()) > 0.02
