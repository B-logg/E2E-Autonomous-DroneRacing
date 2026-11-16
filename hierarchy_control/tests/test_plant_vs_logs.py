"""The plant, driven by what the real drone was commanded, reproduces what it did.

Replays the manual-flight Blackbox logs (drone_geometry/flight_sysid.md) through
ctba.plant at the nominal parameters:
  * estimator:  Betaflight's Mahony filter on the logged IMU reproduces the attitude the
                FC flew on (its ANGLE-mode setpoints follow from it, R2 > 0.97), and the
                simulator's JAX filter is the same filter;
  * ANGLE chain: logged sticks + throttle in, simulated attitude and body rates vs the
                FC's (reproduced) attitude and the logged gyro;
  * rate loop:  logged Betaflight setpoints in, simulated body rates vs logged gyro;
  * thrust:     logged throttle and battery in, simulated filtered specific force vs
                the FC's accelerometer, on the punch / throttle-chop windows.
Skipped when the logs are not on this machine (the training server).
"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "manual_flight_logs" / "LOGS"
pytestmark = pytest.mark.skipif(not (LOGS / "LOG00006.BFL.csv").exists(), reason="flight logs not present")
sys.path.insert(0, str(ROOT / "sysid"))


def _plant_rates(sp_dps):
    import jax
    import jax.numpy as jnp
    from ctba import params as P, plant

    p = P.nominal()
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    s = plant.initial(jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.radians(jnp.asarray(sp_dps[0])),
                      P.GRAVITY, 0.3)
    step = jax.jit(lambda s, c: plant.step_rates(s, 0.3, c, p, d)[0])

    def body(s, c):
        s = step(s, c)
        return s, s.omega_b

    _, w = jax.lax.scan(body, s, jnp.radians(jnp.asarray(sp_dps, jnp.float32)))
    return np.degrees(np.asarray(w))


def test_rate_loop_reproduces_logged_gyro():
    import common as C
    import fit_rate as R

    r2 = []
    for n in (6, 7):
        L = C.prep(n); fs = L["fs"]; k = int(round(fs / 100))
        air = L["armed"] & (C.lp(L["thr"], 2, fs) > 0.15)
        e = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
        a, b = max(zip(e[::2], e[1::2]), key=lambda ab: ab[1] - ab[0])   # longest airborne run
        sp = L["sp"][a:b:k, :3]; g = L["gyro"][a:b:k]
        sim = _plant_rates(sp)
        # the sim's output is the rate at the end of each 10 ms period; the log sample is at its start
        for ax in (0, 1):
            y, yh = g[1:, ax], sim[:-1, ax]
            r2.append(1 - np.var(y - yh) / np.var(y))
    print("rate-loop R2 (LOG6 roll, pitch, LOG7 roll, pitch):", np.round(r2, 3))
    assert min(r2) > 0.9


def test_thrust_reproduces_logged_accelerometer():
    import jax
    import jax.numpy as jnp
    import common as C
    import fit_thrust_dyn as F
    from ctba import params as P, plant

    rms = []
    for n in (4, 5, 7):
        L = C.prep(n); fs = L["fs"]; k = int(round(fs / 100))
        for a, b in F.events(n, L):
            thr = L["thr"][a:b:k]; az = L["acc"][a:b:k, 2]
            V = L["vbat"][a:b:k]; I = L["amp"][a:b:k]
            p = P.nominal()._replace(v_oc=jnp.float32(np.median(V + P.R_INT * I)))
            d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
            a0 = float(P.thrust_static(thr[0], V[0], 1.0))
            s = plant.initial(jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), a0, thr[0])

            def body(s, u):
                s = plant.step_rates(s, u, jnp.zeros(3), p, d)[0]
                return s, -s.acc_f[2]

            _, sim = jax.lax.scan(body, s, jnp.asarray(thr, jnp.float32))
            sim = np.asarray(sim)
            rms.append(np.sqrt(np.mean((sim[:-1] - az[1:]) ** 2)))
    print("thrust replay rms per event (g):", np.round(rms, 2))
    assert np.median(rms) < 0.45


def test_hover_throttle_matches_logs():
    from ctba import params as P
    # LOG3 24.36 V -> 0.281, LOG7 22.37 V -> 0.317 (loaded voltages at hover)
    for v, u in ((24.36, 0.281), (22.99, 0.307), (22.37, 0.317)):
        got = float(P.throttle_for(P.GRAVITY, v, 1.0))
        assert abs(got - u) < 0.015, (v, u, got)


def test_fc_estimator_reproduces_angle_mode_setpoints():
    import fc_attitude as F

    r2 = [f["r2"] for n in (3, 4, 5, 6, 7) for f in F.run(n, 0.25, (1, 1, 1), (1, 1))[3]]
    fits = F.run(6, 0.25, (1, 1, 1), (1, 1))[3]
    print("level-loop R2 per log/axis:", np.round(r2, 3), "LOG6 gain/limit:",
          [(round(f["gain"], 2), round(f["limit"], 1)) for f in fits])
    assert min(r2) > 0.97
    for f in fits:
        assert abs(f["gain"] - 4.9) < 0.3 and abs(f["limit"] - 55.0) < 1.5


def test_jax_estimator_is_betaflights():
    import common as C
    import fc_attitude as F
    import fit_angle_chain as A

    L = C.prep(6)
    _, q = F.mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=L["fs"])
    bf = F.euler_bf(q)
    jx = A.jax_estimate(L)[: len(bf)]
    # Betaflight pitch is nose-down positive, FRD pitch nose-up positive
    np.testing.assert_allclose(jx * [1, -1], bf, atol=0.5)


def test_angle_chain_reproduces_flights():
    import common as C
    import fc_attitude as F
    import fit_angle_chain as A

    for n in (3, 7):
        L = C.prep(n)
        _, q = F.mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=L["fs"])
        att = F.euler_bf(q)
        a, b = max(A.segments(L), key=lambda ab: ab[1] - ab[0])
        k = int(round(L["fs"] / F.ATT_HZ))
        ref = att[np.clip(np.arange(a, b, k) // k, 0, len(att) - 1)]
        idx, est, w = A.replay(L, a, b, ref, [1.0, -1.0])
        m = min(len(est), len(ref))
        y, yh = ref[1:m], est[: m - 1]
        tgt = (L["rc"][idx, :2] / 500.0 * 55.0)[: m - 1]
        g = (L["gyro"][idx, :2] * A.FLU_TO_FRD[:2])[1:m]
        for ax in (0, 1):
            r_att, r_tgt = A.r2(y[:, ax], yh[:, ax]), A.r2(y[:, ax], tgt[:, ax])
            r_w = A.r2(g[:, ax], w[: m - 1, ax])
            print(f"LOG{n} axis {ax}: attitude R2 {r_att:.3f} (target-only {r_tgt:.3f}), rates R2 {r_w:.3f}")
            assert r_att > 0.98 and r_att > r_tgt + 0.1 and r_w > 0.9
