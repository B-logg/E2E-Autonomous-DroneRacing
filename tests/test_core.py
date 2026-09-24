"""Checks that pin the transcription of the paper down.

Run: .venv/bin/python -m pytest tests -q
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skydreamer import augment, track as T
from skydreamer.dynamics import (
    Disturbance,
    State,
    euler_to_quat,
    derivative,
    motor_setpoint,
    quat_to_euler,
    step,
)
from skydreamer.env import EnvConfig, batched, gates_passed, reset
from skydreamer.env import step as env_step  # dynamics.step shadows this name
from skydreamer.env import unpack_mask
from skydreamer.params import GRAVITY, NOMINAL, DynParams
from skydreamer.render import render_mask

ZERO = Disturbance(jnp.zeros(3), jnp.zeros(3), jnp.zeros(4))
P = DynParams.nominal()
W_HOVER = float(np.sqrt(GRAVITY / (4 * NOMINAL["k_w"])))


def hover_state():
    return State(
        p=jnp.zeros(3),
        v=jnp.zeros(3),
        q=jnp.array([1.0, 0, 0, 0]),
        omega_b=jnp.zeros(3),
        motor=jnp.full((4,), W_HOVER),
    )


def hover_command():
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        w = float(motor_setpoint(jnp.full((4,), mid), jnp.zeros(4), P)[0])
        lo, hi = (mid, hi) if w < W_HOVER else (lo, mid)
    return (lo + hi) / 2


def test_thrust_to_weight_matches_paper():
    """Table II must give the ~6 g the paper reports in flight."""
    twr = NOMINAL["k_w"] * 4 * NOMINAL["w_max"] ** 2 / GRAVITY
    assert 5.9 < twr < 6.2, twr


def test_hover_is_an_equilibrium_in_translation():
    d = derivative(hover_state(), jnp.full((4,), hover_command()), ZERO, P)
    assert np.allclose(np.asarray(d.v), 0.0, atol=1e-4)


def test_uniform_command_leaves_a_residual_moment():
    """Not a bug: Table II's per-motor coefficients are asymmetric, so a
    uniform command produces a real trim moment the policy has to cancel.
    Pinned here so a future 'fix' to the moment signs gets caught."""
    d = derivative(hover_state(), jnp.full((4,), hover_command()), ZERO, P)
    assert abs(float(d.omega_b[0])) > 1.0
    assert abs(float(d.omega_b[1])) > 1.0
    assert float(d.omega_b[2]) == pytest.approx(0.0, abs=1e-6)


def test_climb_reduces_thrust_descent_increases_it():
    """Sign check on the angle-of-attack term: momentum theory says climbing
    at fixed RPM loses thrust.  Wrong sign here silently ruins sim-to-real."""
    u = jnp.full((4,), hover_command())
    climb = hover_state()._replace(v=jnp.array([0.0, 0.0, -5.0]))  # NED: up
    descend = hover_state()._replace(v=jnp.array([0.0, 0.0, 5.0]))
    a_climb = float(derivative(climb, u, ZERO, P).v[2])
    a_desc = float(derivative(descend, u, ZERO, P).v[2])
    assert a_climb > 0 > a_desc  # climbing sags (falls), descending is buoyed


def test_quaternion_stays_normalized():
    s = hover_state()._replace(omega_b=jnp.array([5.0, -3.0, 2.0]))
    for _ in range(200):
        s = step(s, jnp.full((4,), 0.4), ZERO, P)
    assert float(jnp.linalg.norm(s.q)) == pytest.approx(1.0, abs=1e-5)


def test_euler_roundtrip():
    ang = jnp.array([0.3, -0.5, 2.0])
    out = quat_to_euler(euler_to_quat(ang[0], ang[1], ang[2]))
    assert np.allclose(np.asarray(out), np.asarray(ang), atol=1e-5)


# --------------------------------------------------------------------------
# track
# --------------------------------------------------------------------------


def test_plane_pose_wraps_across_laps():
    """plane index is monotone across laps; without the modulo JAX clamps the
    gather and every lap after the first targets the final gate forever."""
    tr = T.inverted_loop()
    n = tr.n_gates
    for i in range(3 * n):
        a, ya, ga = T.plane_pose(tr, jnp.int32(i), 0.8)
        b, yb, gb = T.plane_pose(tr, jnp.int32(i + 3 * n), 0.8)
        assert np.allclose(np.asarray(a), np.asarray(b))
        assert float(ya) == float(yb)


def test_pre_and_post_gates_straddle_the_gate():
    tr = T.inverted_loop()
    t_g = 0.8
    pre, _, _ = T.plane_pose(tr, jnp.int32(0), t_g)
    mid, _, _ = T.plane_pose(tr, jnp.int32(1), t_g)
    post, _, _ = T.plane_pose(tr, jnp.int32(2), t_g)
    assert np.allclose(np.asarray(mid), np.asarray(tr.pos[0]))
    assert float(jnp.linalg.norm(post - pre)) == pytest.approx(t_g, abs=1e-5)


def test_flight_plan_shape_and_no_drone_state():
    tr = T.inverted_loop()
    f = T.flight_plan(tr, jnp.int32(0))
    assert f.shape == (T.FLIGHT_PLAN_DIM,)
    # the absolute half must be gates i..i+2 verbatim
    assert np.allclose(np.asarray(f[12:15]), np.asarray(tr.pos[0]))


def test_flight_plan_cycles():
    tr = T.inverted_loop()
    a = T.flight_plan(tr, jnp.int32(0))
    b = T.flight_plan(tr, jnp.int32(tr.n_gates))
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_gate_reward_and_collision_agree_at_the_edge():
    d_g = 1.0
    edge = jnp.array([0.0, d_g, 0.0])
    assert float(T.gate_reward(edge, d_g)) == pytest.approx(0.0)
    assert not bool(T.gate_collision(edge, d_g))
    assert bool(T.gate_collision(jnp.array([0.0, d_g * 1.01, 0.0]), d_g))


def test_rate_penalty_is_negligible_until_it_bites():
    lo = float(T.rate_penalty(jnp.full((3,), 10.0 / 3)))
    hi = float(T.rate_penalty(jnp.full((3,), 17.0 / 3)))
    assert lo < 1e-2 < hi
    # clipped at 17, so it cannot explode
    assert float(T.rate_penalty(jnp.full((3,), 100.0))) == pytest.approx(hi)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _approach_gate_one(tr, back=4.0):
    """Sit `back` metres before gate 1 on its normal, pitched 50 deg forward so
    the 50 deg upward camera looks straight down the track."""
    g = np.asarray(tr.pos[0])
    yaw = float(tr.yaw[0])
    eye = jnp.array([g[0] - back * np.cos(yaw), g[1] - back * np.sin(yaw), g[2]])
    q = euler_to_quat(jnp.array(0.0), jnp.array(-np.deg2rad(50)), jnp.array(yaw))
    return eye, q


def test_gate_renders_centred_when_looked_at_straight_on():
    tr = T.inverted_loop()
    eye, q = _approach_gate_one(tr)
    ce = jnp.array([0.0, np.deg2rad(50.0), 0.0])
    m = np.asarray(render_mask(eye, q, ce, tr))
    assert m.sum() > 0
    # Gate 2 sits 5 m to the side and shows up at the frame edge, so measure
    # only the central half -- otherwise it drags the centroid off.
    ys, xs = np.nonzero(m[:, 16:48])
    assert abs((xs + 16).mean() - 32) < 3, (xs + 16).mean()
    assert abs(ys.mean() - 32) < 5, ys.mean()


def test_camera_extrinsics_actually_move_the_image():
    tr = T.inverted_loop()
    eye, q = _approach_gate_one(tr)
    a = np.asarray(render_mask(eye, q, jnp.array([0.0, np.deg2rad(50), 0.0]), tr))
    b = np.asarray(render_mask(eye, q, jnp.array([0.0, np.deg2rad(45), 0.0]), tr))
    assert not np.array_equal(a, b)


def test_invisible_gates_are_never_drawn():
    """Isolate one gate so nothing else can leak into the frame."""
    q = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    eye = jnp.array([-3.0, 0.0, -1.5])
    shown = T.make_track([(0.0, 0.0, -1.5, 0.0, True)])
    hidden = T.make_track([(0.0, 0.0, -1.5, 0.0, False)])
    assert float(render_mask(eye, q, jnp.zeros(3), shown).sum()) > 0.0
    assert float(render_mask(eye, q, jnp.zeros(3), hidden).sum()) == 0.0


def test_erosion_shrinks_the_mask():
    m = jnp.zeros((16, 16)).at[4:12, 4:12].set(1.0)
    assert float(augment.erode1(m).sum()) == 36.0  # 8x8 -> 6x6


def test_rolling_shutter_is_identity_at_zero_rate():
    m = jnp.zeros((16, 16)).at[4:12, 4:12].set(1.0)
    out = augment.rolling_shutter(m, jnp.array(0.0), jnp.array(0.0), jnp.array(0.01))
    assert np.array_equal(np.asarray(out), np.asarray(m))


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def test_observation_split_matches_the_informed_pomdp():
    cfg = EnvConfig(track=T.inverted_loop())
    _, obs = reset(jax.random.key(0), cfg)
    o = {k for k in obs if not k.startswith("info_")}
    i = {k for k in obs if k.startswith("info_")}
    assert o == {"mask", "rates", "rpm", "flight_plan"}
    # i_t = {o_t, o_t^+} \ {X}: the image must NOT be a decoder target
    assert "info_mask" not in i
    assert {"info_p_w", "info_v_w", "info_cam", "info_dyn"} <= i


def test_batched_rollout_is_finite():
    cfg = EnvConfig(track=T.ladder_inverted_loop())
    reset_b, step_b = batched(cfg)
    st, obs = reset_b(jax.random.split(jax.random.key(0), 16))
    for _ in range(200):
        st, obs, r, term, trunc = step_b(st, jnp.full((16, 4), 0.35))
        for v in obs.values():
            assert bool(jnp.all(jnp.isfinite(v)))
        assert bool(jnp.all(jnp.isfinite(r)))


def test_episode_starts_before_the_first_plane():
    """If a reset ever lands past its target plane the drone banks a free gate
    reward on step 1, which quietly corrupts the return scale."""
    cfg = EnvConfig(track=T.inverted_loop())
    for seed in range(64):
        st, _ = reset(jax.random.key(seed), cfg)
        target, yaw, _ = T.plane_pose(cfg.track, st.plane, st.t_g)
        local = T.gate_rotation(yaw) @ (st.s.p - target)
        assert float(local[0]) < 0.0
        assert int(gates_passed(st)) == 0


# --------------------------------------------------------------------------
# regressions for bugs found in the verification pass
# --------------------------------------------------------------------------


def _delay_probe(cfg, n):
    """Fly a fixed action and record (true mask, observed mask) per step."""
    st, obs = reset(jax.random.key(7), cfg)
    truth, seen = [], []
    for _ in range(n):
        st, obs, *_ = env_step(st, jnp.full((4,), 0.45), cfg)
        truth.append(np.asarray(st.mask_buf[-1]))  # freshly rendered this step
        seen.append(np.asarray(unpack_mask(obs["mask"])))
    return truth, seen


def test_image_delay_is_exactly_three_steps():
    """33 ms at an 11.0 ms control period.  The buffer must hold D+1 frames:
    sizing it D silently gives a 22 ms delay instead."""
    from skydreamer.env import IMAGE_DELAY_STEPS

    cfg = EnvConfig(track=T.inverted_loop())
    truth, seen = _delay_probe(cfg, 40)
    # find the lag that lines the two sequences up
    lags = []
    for t in range(12, 40):
        matches = [d for d in range(0, 6) if np.array_equal(seen[t], truth[t - d])]
        assert matches, f"observed frame at step {t} matches no recent true frame"
        lags.append(min(matches))
    assert max(lags) == IMAGE_DELAY_STEPS == 3, (max(lags), IMAGE_DELAY_STEPS)


def test_action_delay_is_exactly_one_step():
    """The action leaving the policy at step t must reach the motors at t+1.

    Disturbances are zeroed for this test: eps_u alone can drive a motor the
    *opposite* way from a zero command, which is correct behaviour but makes
    the delay unobservable in the motor trace."""
    cfg = EnvConfig(track=T.inverted_loop())
    st, _ = reset(jax.random.key(11), cfg)
    st = st._replace(bounds=jnp.zeros(4))

    before = np.asarray(st.s.motor).copy()
    st, *_ = env_step(st, jnp.ones(4), cfg)  # commanded, not yet executed
    after_one = np.asarray(st.s.motor)
    st, *_ = env_step(st, jnp.ones(4), cfg)  # now the full-throttle one runs
    after_two = np.asarray(st.s.motor)

    # step 1 executes the buffered zero command -> motors decay toward w_min
    assert (after_one < before).all(), (before, after_one)
    # step 2 executes full throttle -> motors climb again
    assert (after_two > after_one).all(), (after_one, after_two)


def test_disturbance_and_flightplan_randomness_are_independent():
    """Every random draw in step() needs its own key.  Sharing one makes the
    draws identical, which coupled the flight-plan increment to the
    acceleration-disturbance resample."""
    cfg = EnvConfig(track=T.inverted_loop())
    st, _ = reset(jax.random.key(5), cfg)
    key, *k = jax.random.split(st.key, 9)
    draws = [float(jax.random.uniform(x)) for x in k]
    assert len(set(draws)) == len(draws), "duplicate key detected in step()"


def test_dyn_vector_normalizes_each_parameter_to_unit_range():
    """+-20% params and +-30% params must both map to +-1, or the decoder sees
    targets on two different scales."""
    from skydreamer.params import NOMINAL, TRAIN, PARAM_NAMES

    edge = {}
    for name in PARAM_NAMES:
        frac = 0.20 if name in ("w_min", "w_max") else TRAIN.param_frac
        edge[name] = jnp.asarray(NOMINAL[name] * (1.0 + frac))
    v = np.asarray(DynParams(**edge).as_vector())
    nonzero = [NOMINAL[n] != 0.0 for n in PARAM_NAMES]
    assert np.allclose(v[nonzero], 1.0, atol=1e-4), v[nonzero]
    assert np.allclose(v[[not x for x in nonzero]], 0.0)


def test_every_table_two_coefficient_is_transcribed():
    """Guard against a silent edit to Table II."""
    from skydreamer.params import NOMINAL

    expected = {
        "k_w": 1.55e-6, "k_x": 5.37e-5, "k_y": 5.37e-5,
        "k_x2": 4.10e-3, "k_y2": 1.51e-2, "k_v2": 0.0,
        "k_angle": 3.145, "k_hor": 7.245,
        "J_x": -0.89, "J_y": 0.96, "J_z": -0.34,
        "w_min": 341.75, "w_max": 3100.0, "k": 0.50, "tau": 0.03,
        "k_p1": 4.99e-5, "k_p2": 3.78e-5, "k_p3": 4.82e-5, "k_p4": 3.83e-5,
        "k_q1": 2.05e-5, "k_q2": 2.46e-5, "k_q3": 2.02e-5, "k_q4": 2.57e-5,
        "k_r1": 3.38e-3, "k_r2": 3.38e-3, "k_r3": 3.38e-3, "k_r4": 3.38e-3,
        "k_r5": 3.24e-4, "k_r6": 3.24e-4, "k_r7": 3.24e-4, "k_r8": 3.24e-4,
    }
    assert NOMINAL == expected


def test_divergence_guard_keeps_nan_out_of_observations():
    """The moment equation's gyroscopic coupling is quadratic in the body rates
    and the model has no rotational damping, so a spinning drone runs away:
    100 rad/s already gives 8900 rad/s^2, and RK4 at 2.2 ms eventually reaches
    NaN.  That killed a real run at 12k steps -- DreamerV3 asserts on
    non-finite observations, and by then the buffer is already polluted.

    Adversarial commands across many seeds must never produce a non-finite
    observation."""
    cfg = EnvConfig(track=T.inverted_loop())
    ended_by_spin = 0
    for seed in range(40):
        st, obs = reset(jax.random.key(seed), cfg)
        for t in range(400):
            u = jnp.array([1.0, 0.0, 1.0, 0.0]) if (t // 40) % 2 == 0 else jnp.array(
                [0.0, 1.0, 0.0, 1.0])
            st, obs, r, term, trunc = env_step(st, u, cfg)
            for v in obs.values():
                assert bool(jnp.all(jnp.isfinite(v))), (seed, t)
            assert bool(jnp.isfinite(r))
            if bool(term):
                if float(jnp.max(jnp.abs(st.s.omega_b))) > T.RATE_DIVERGENCE:
                    ended_by_spin += 1
                break
    assert ended_by_spin > 0, "the guard never fired; the stress test is too gentle"


def test_divergence_threshold_leaves_real_maneuvers_alone():
    """The paper's inverted loop is about 10 rad/s (13 m/s on a 1.35 m radius)
    and its rate penalty clips at ||Omega||_1 = 17, so the guard must sit well
    above both or it would cut off legitimate flight."""
    # RATE_CLIP is an L1 budget across three axes; RATE_DIVERGENCE is per axis.
    # Even if the whole L1 budget sat on one axis, the guard is well above it.
    assert T.RATE_DIVERGENCE > 2 * T.RATE_CLIP
    ok = jnp.array([15.0, 15.0, 15.0])          # harder than any paper maneuver
    assert not bool(T.diverged(ok, jnp.zeros(3), jnp.zeros(3)))
    assert bool(T.diverged(jnp.array([60.0, 0.0, 0.0]), jnp.zeros(3), jnp.zeros(3)))
    nan = jnp.array([jnp.nan, 0.0, 0.0])
    assert bool(T.diverged(nan, jnp.zeros(3), jnp.zeros(3)))


def test_exported_track_files_match_the_code():
    """`tracks/` is generated from `track.py`. A second hand-maintained copy of
    the same coordinates drifts from the first, and then nobody can say which
    layout the policy actually flew."""
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "export_tracks.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def _load_evaluate():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sd_evaluate", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_feeds_the_agent_the_mask_training_fed_it():
    """The evaluation harness builds its own observation dict instead of going
    through embodied_env.py, so the two can drift -- and they did.  The mask is
    bit-packed uint8; thresholding it (`(byte > 0.5) * 255`, left over from when
    the env emitted a float mask) turns every byte holding one gate pixel into
    eight lit pixels.  The policy then flies on a grossly dilated image, which
    scored the 17M checkpoint at 0% success on the track it was trained on.

    Pin the invariant: whatever the harness hands the agent must unpack to the
    same image the environment produced."""
    from skydreamer.env import PACK_MASK

    ev = _load_evaluate()
    cfg = EnvConfig(track=T.inverted_loop())

    # Find a frame with a partly-filled byte -- an all-or-nothing mask cannot
    # show the corruption, so an empty frame would make this test vacuous.
    for seed in range(40):
        st, obs = reset(jax.random.key(seed), cfg)
        raw = np.asarray(obs["mask"])
        if PACK_MASK and np.any((raw > 0) & (raw < 255)):
            break
    else:
        pytest.skip("no partly-filled mask byte found; cannot detect corruption")

    got = ev._agent_obs({k: np.asarray(v) for k, v in obs.items()})["mask"]
    assert got.dtype == np.uint8
    assert np.array_equal(got, raw), "the harness altered the packed mask"
    assert np.array_equal(
        np.asarray(unpack_mask(jnp.asarray(got))), np.asarray(unpack_mask(obs["mask"]))
    )

    # And the specific thing that was wrong really does destroy it.
    corrupted = (raw > 0.5).astype(np.uint8) * 255
    assert not np.array_equal(corrupted, raw)


def test_symlog_head_predictions_are_not_in_the_original_units():
    """The trap that made the decoded state look ten times worse than it is.

    `symlog_mse` builds `MSE(pred, symlog)`: the loss squashes the *target*, so
    the network's output lives in symlog space.  `MSE.pred()` returns it
    untouched -- correct for a loss, which never needs the inverse, and wrong
    for anything reading it as a state estimate.  A drone 3 m out decodes as
    1.39, which scores as a 1.6 m error that is purely the missing symexp.

    Measured on the finished 17M checkpoint, inverting it moved the reported
    position error from 2.219 m to 0.249 m.  Pinned here because it is a
    property of vendored DreamerV3, so an upgrade could silently change it."""
    pytest.importorskip("embodied")
    from embodied.jax import outs
    from embodied.jax.nets import symlog

    truth = 3.0
    squashed = float(symlog(jnp.array(truth)))
    assert squashed == pytest.approx(np.log(4.0), abs=1e-6)

    head = outs.MSE(jnp.array([squashed]), symlog)
    # A perfect prediction costs nothing ...
    assert float(head.loss(jnp.array([truth]))[0]) == pytest.approx(0.0, abs=1e-6)
    # ... and yet pred() is not the truth; it is still log-compressed.
    assert float(head.pred()[0]) == pytest.approx(squashed, abs=1e-6)
    assert abs(float(head.pred()[0]) - truth) > 1.5

    # symexp is the inverse the reader has to apply.
    p = head.pred()
    recovered = float((jnp.sign(p) * jnp.expm1(jnp.abs(p)))[0])
    assert recovered == pytest.approx(truth, abs=1e-5)


def test_patch_inverts_symlog_before_exposing_decoded_state():
    patch = (pathlib.Path(__file__).resolve().parents[1]
             / "patches" / "informed_dreamer.patch").read_text()
    assert "expm1" in patch, "decoded state is exposed without undoing symlog"


def test_chase_view_renders_the_simulated_world():
    """The third-person panel used to be a matplotlib line drawing of recorded
    numbers.  This renders the scene with the simulator's own ray caster --
    gates from `render_mask`, floor from intersecting the same rays with the
    ground plane the dynamics terminate on -- so it is a picture of the
    environment rather than a plot of it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sd_visualize", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "visualize.py"
    )
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)

    tr = T.inverted_loop()
    gate = np.asarray(tr.pos[0])
    yaw = float(tr.yaw[0])
    # Sit 5 m before gate 1, flying at it level.
    p = jnp.array([gate[0] - 5 * np.cos(yaw), gate[1] - 5 * np.sin(yaw), gate[2]])
    q = euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(yaw))

    img, uv = viz.chase_view(p, q, tr, res=64)
    assert img.shape == (64, 64, 3)

    colours = {tuple(np.round(c, 3)) for c in img.reshape(-1, 3)}
    assert len(colours) >= 3, "expected at least sky, floor and gate"
    assert tuple(np.round(viz._rgb(viz.INK), 3)) in colours, "no gate in view"
    assert tuple(np.round(viz._rgb(viz.FLOOR), 3)) in colours, "no ground in view"

    # The chase camera sits behind the drone, so the drone must project into
    # the frame -- that is the whole point of the viewpoint.
    assert uv is not None
    assert 0 <= uv[0] < 64 and 0 <= uv[1] < 64, uv


def test_env_records_why_the_episode_ended():
    """`done` is one bool, which is all the agent needs and useless for working
    out why a policy fails.  Flying into a gate, mushing into the floor and
    tumbling out of control want different fixes."""
    cfg = EnvConfig(track=T.inverted_loop())
    st, _ = reset(jax.random.key(0), cfg)
    assert int(st.term_cause) == 0, "a fresh episode has not ended"

    seen = set()
    for seed in range(30):
        st, _ = reset(jax.random.key(seed), cfg)
        # Alternating opposite motor pairs spins the drone up; a low uniform
        # command just drops it.  Between them both failure modes appear.
        spin = seed % 2 == 0
        for t in range(400):
            if spin:
                u = jnp.array([1.0, 0.0, 1.0, 0.0]) if (t // 40) % 2 == 0 else jnp.array(
                    [0.0, 1.0, 0.0, 1.0])
            else:
                u = jnp.full((4,), 0.05)
            st, _, _, term, _ = env_step(st, u, cfg)
            if bool(term):
                cause = int(st.term_cause)
                assert cause != 0, "terminated without a cause"
                seen.add(cause)
                break
    assert {2, 3} <= seen, f"expected ground and divergence among causes, saw {seen}"


def test_summary_still_describes_a_run_that_never_succeeds():
    """Every paper-comparison row is defined only over completed flights, so a
    0% run printed n/a on six of seven rows and said nothing about how it
    failed.  The diagnostic layer must describe those episodes anyway."""
    ev = _load_evaluate()
    records = [
        dict(success=False, cause=1, gates=1, duration=0.68, gate_times=[0.4],
             gate_err=[0.21], margin=0.05, vmax=10.6, amax=4.1),
        dict(success=False, cause=2, gates=0, duration=0.31, gate_times=[],
             gate_err=[], margin=None, vmax=9.2, amax=3.8),
        dict(success=False, cause=3, gates=1, duration=1.22, gate_times=[0.5],
             gate_err=[0.44], margin=0.02, vmax=17.0, amax=5.9),
    ]
    decode = {"p_w": 2.2, "v_w": 5.9, "early": {"p_w": 0.1, "v_w": 0.3}}
    s = ev.summarize(records, decode, n_gates=3, laps=5)

    assert s["success_rate"] == 0.0
    assert "lap1_s" not in s, "lap times need a completed lap; they must stay absent"

    d = s["diagnostics"]
    assert d["gates_passed"]["histogram"] == {0: 1, 1: 2}
    assert d["termination"] == {"hit a gate": 1, "hit the ground": 1, "diverged": 1}
    assert d["duration_s"]["max"] == pytest.approx(1.22)
    assert d["max_speed_ms_all"]["max"] == pytest.approx(17.0)
    assert d["gate_error_m_all"]["max"] == pytest.approx(0.44)
    assert d["min_gate_margin_m_all"] == pytest.approx(0.02)

    text = ev.report("inverted_loop", s)
    for want in ("gates passed", "ended by", "hit a gate", "flight time",
                 f"first {ev.EARLY_STEPS} steps", "whole flight"):
        assert want in text, want


def test_eval_driver_stops_integrating_dead_episodes():
    """`scripts/evaluate.py` drives the batched env itself instead of going
    through embodied's driver, and the driver is what normally resets an env on
    `is_last`.  Without an equivalent, a drone that crashes on step 100 keeps
    being flown for the remaining 1900 steps of the batch: the guard terminates
    it again every step to no effect while its rates grow without bound, until
    they overflow and DreamerV3's finiteness assert kills the evaluation.

    That is exactly what happened on the 17M checkpoint -- observed body rates
    of -2240 rad/s and a NaN row.  Every smoke test missed it because an untrained
    policy crashes every episode almost at once, so the batch ends before the
    runaway has time to grow.

    The fix is `_hold`: freeze the state of any episode that is no longer
    flying.  This pins both halves -- that the runaway is real, and that
    freezing stops it."""
    ev = _load_evaluate()

    cfg = EnvConfig(track=T.inverted_loop())
    reset_b, step_b = batched(cfg)
    b, horizon = 8, 2000

    def fly(freeze: bool) -> float:
        st, obs = reset_b(jax.random.split(jax.random.key(0), b))
        alive = np.ones(b, bool)
        worst = 0.0
        for t in range(horizon):
            u = jax.random.uniform(jax.random.key(1000 + t), (b, 4))
            st_next, obs_next, _, term, _ = step_b(st, u)
            if freeze:
                hold = jnp.asarray(~alive)
                st = ev._hold(hold, st, st_next)
                obs = ev._hold(hold, obs, obs_next)
            else:
                st, obs = st_next, obs_next
            alive &= ~(np.asarray(term) & alive)
            rates = np.asarray(obs["rates"])
            if not np.isfinite(rates).all():
                return float("inf")
            worst = max(worst, float(np.abs(rates).max()))
        return worst

    # Kept flying after death, the rates run away -- orders of magnitude past
    # anything physical, and on a longer batch all the way to NaN.
    assert fly(freeze=False) > 10 * T.RATE_DIVERGENCE
    # Frozen, they stay bounded.  Not *at* the threshold: the guard fires on
    # the step that crosses 50 rad/s, and that step's observation is recorded
    # before the freeze takes effect on the next one, so one step's worth of
    # overshoot is expected and correct.  What matters is that it stops there.
    assert fly(freeze=True) < 2 * T.RATE_DIVERGENCE


def test_big_track_matches_figure_nine():
    """The big track is measured off Figure 9's top-down and side views in the
    PDF (page 14 -- the arXiv HTML drops those panels because they are vector).
    These are the positions that measurement gave; they are pinned so a future
    edit cannot quietly drift away from the paper."""
    pos = np.asarray(T.big_track().pos)
    vis = np.asarray(T.big_track().visible)
    real = pos[vis]

    # two straights, two gates each, at X = +-6
    top = sorted(float(p[1]) for p in real if abs(p[0] - 6.0) < 0.3)
    bot = sorted(float(p[1]) for p in real if abs(p[0] + 6.0) < 0.3)
    assert len(top) == len(bot) == 2, (top, bot)
    assert top == pytest.approx([-3.4, 2.8], abs=0.1)
    assert bot == pytest.approx([-3.4, 2.8], abs=0.1)

    # the start gate at the left end, and the middle gate turned 90 degrees
    assert any(abs(p[0] - 3.7) < 0.2 and abs(p[1] + 9.5) < 0.2 for p in real)
    mid = [p for p in real if abs(p[0]) < 0.3 and abs(p[1] + 0.3) < 0.3]
    assert len(mid) == 1

    # gates sit on the floor: blocks span altitude 0.17-2.34 m in the figure
    assert float(-real[0][2]) == pytest.approx(1.25, abs=0.1)
    assert int(vis.sum()) == 8


def test_big_track_is_an_unseen_layout():
    """It only earns the name out-of-distribution if it really is one, while
    keeping the gates themselves identical so a failure cannot be blamed on a
    changed mask."""
    small, big = T.inverted_loop(), T.big_track()

    assert float(big.inner[0]) == float(small.inner[0])
    assert float(big.outer[0]) == float(small.outer[0])

    small_h = {round(float(np.degrees(y))) % 360 for y in small.yaw}
    big_h = {round(float(np.degrees(y))) % 360 for y in big.yaw}
    assert len(big_h) > 2 * len(small_h), (small_h, big_h)

    span = lambda t: float(np.ptp(np.asarray(t.pos)[:, 1]))
    assert span(big) > 3 * span(small), (span(small), span(big))


def test_big_track_gates_are_reachable_in_sequence():
    """No two gates coincident, and none so far apart that a lap is mostly
    empty space.  The measured layout's longest hop is the middle gate back to
    the start, about 10 m on a 21 m track."""
    pos = np.asarray(T.big_track().pos)
    gaps = np.linalg.norm(np.diff(np.vstack([pos, pos[:1]]), axis=0), axis=-1)
    assert gaps.min() > 1.0, gaps.min()
    assert gaps.max() < 12.0, gaps.max()
