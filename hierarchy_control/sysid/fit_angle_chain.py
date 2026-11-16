"""ANGLE-mode chain, end to end: the logged sticks (target angles) and throttle in,
the simulated drone (ctba.plant: FC estimator + level loop + rate loop + thrust)
out, compared with the FC attitude reproduced from the logged IMU (fc_attitude.py).

Also checks that ctba.plant.estimate (JAX, FRD) is the same filter as
fc_attitude.mahony (numpy, Betaflight's board frame) and finds the sign mapping
between Betaflight's sticks / angles and the simulator's FRD angles.

Betaflight board frame as logged: x forward, y left, z up  ->  FRD = (x, -y, -z).

    python fit_angle_chain.py      -> results_angle_chain.json
"""
import json
from pathlib import Path

import numpy as np

import common as C
import fc_attitude as F

HERE = Path(__file__).resolve().parent
FLU_TO_FRD = np.array([1.0, -1.0, -1.0])
LOGS = (3, 4, 5, 6, 7)


def jax_estimate(L, kp=0.25):
    """Run ctba.plant.estimate on the logged IMU (converted to FRD), 100 Hz."""
    import jax
    import jax.numpy as jnp
    from ctba import plant

    fs = L["fs"]; k = int(round(fs / F.ATT_HZ)); n = len(L["t"]) // k
    g = np.radians(L["gyro"][: n * k].reshape(n, k, 3).mean(1)) * FLU_TO_FRD
    a = L["acc"][: n * k].reshape(n, k, 3).mean(1) * FLU_TO_FRD
    arm = L["armed"][: n * k].reshape(n, k)[:, -1]
    kps = np.where(arm, kp, 10 * kp)

    def body(q, x):
        q = plant.estimate(q, x[0], x[1], x[2])
        return q, plant.attitude_rp(q)

    _, rp = jax.lax.scan(body, jnp.array([1.0, 0, 0, 0]),
                         (jnp.asarray(g, jnp.float32), jnp.asarray(a, jnp.float32), jnp.asarray(kps, jnp.float32)))
    return np.degrees(np.asarray(rp))


def segments(L, min_s=8.0):
    """Airborne runs (throttle up, not near a crash attitude), at least `min_s` long."""
    fs = L["fs"]
    air = L["armed"] & (C.lp(L["thr"], 2, fs) > 0.15)
    e = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if (b - a) / fs >= min_s]


def replay(L, a, b, att_ref, sign, p=None):
    """Sticks + throttle of [a, b) into the simulator, starting from the reproduced FC attitude."""
    import jax
    import jax.numpy as jnp
    from skydreamer.dynamics import euler_to_quat
    from ctba import params as P, plant

    fs = L["fs"]; k = int(round(fs / 100))
    idx = np.arange(a, b, k)
    stick = L["rc"][idx, :3] / 500.0
    tgt = np.radians(P.ANGLE_LIMIT * 180 / np.pi * stick[:, :2] * np.asarray(sign))
    yaw_sp = np.radians(L["sp"][idx, 2]) * -1.0          # yaw rate setpoint as logged, to FRD
    thr = L["thr"][idx]
    V = L["vbat"][idx]; I = L["amp"][idx]
    p = P.nominal() if p is None else p
    p = p._replace(v_oc=jnp.float32(np.nanmedian(V + P.R_INT * I)))
    d = plant.Disturbance(jnp.zeros(3), jnp.zeros(3))
    r0 = np.radians(att_ref[0] * np.asarray(sign))
    q0 = euler_to_quat(r0[0], r0[1], 0.0)
    a0 = float(P.thrust_static(thr[0], V[0], 1.0))
    s = plant.initial(jnp.array([0.0, 0, -2.0]), q0, jnp.zeros(3), a0, thr[0])

    def body(s, x):
        s = plant.step(s, x[0], x[1:3], x[3], p, d)
        return s, (plant.attitude_rp(s.q_est), plant.attitude_rp(s.q), s.omega_b)

    xs = jnp.asarray(np.c_[thr, tgt, yaw_sp], jnp.float32)
    _, (est, true, w) = jax.jit(lambda s, xs: jax.lax.scan(body, s, xs))(s, xs)
    return idx, np.degrees(np.asarray(est)) * np.asarray(sign), np.degrees(np.asarray(w))


def r2(y, yh):
    return float(1 - np.var(y - yh) / np.var(y))


def main():
    res = {"estimator_match": {}, "chain": {}}
    # 1) the JAX filter is Betaflight's filter, and which FRD signs equal BF's angles
    L = C.prep(6)
    idx, q = F.mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=L["fs"])
    bf = F.euler_bf(q)
    jx = jax_estimate(L)[: len(bf)]
    sign = [float(np.sign(np.corrcoef(bf[:, i], jx[:, i])[0, 1])) for i in (0, 1)]
    diff = np.abs(bf - jx * np.asarray(sign))
    res["estimator_match"] = dict(sign_bf_from_frd=sign, max_abs_deg=diff.max(0).round(3).tolist(),
                                  p99_deg=np.percentile(diff, 99, 0).round(3).tolist())
    print("JAX vs numpy estimator (LOG6):", res["estimator_match"])

    # 2) the chain: sticks in, attitude out
    for n in LOGS:
        L = C.prep(n)
        fs = L["fs"]; k100 = int(round(fs / F.ATT_HZ))
        idx_att, q = F.mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=fs)
        att = F.euler_bf(q)
        out = []
        for a, b in segments(L):
            # FC attitude at the replay samples (each update holds for its 10 ms)
            j = np.clip(np.arange(a, b, k100) // k100, 0, len(att) - 1)
            ref = att[j]
            ok = (np.abs(ref) < 45).all(1)
            if ok.mean() < 0.95:
                continue
            idx, est, w = replay(L, a, b, ref, sign)
            m = min(len(est), len(ref))
            # simulated estimate at the end of period i ~ reproduced attitude at period i+1
            y, yh = ref[1:m], est[: m - 1]
            # baseline without any dynamics: attitude = stick target
            tgt = (L["rc"][idx, :2] / 500.0 * 55.0)[: m - 1]
            # body rates: logged gyro (board frame) vs simulated, FRD signs
            g = (L["gyro"][idx, :2] * FLU_TO_FRD[:2])[1:m]
            out.append(dict(seg=[int(a), int(b)], seconds=round((b - a) / fs, 1),
                            r2_roll=r2(y[:, 0], yh[:, 0]), r2_pitch=r2(y[:, 1], yh[:, 1]),
                            rms_deg=np.sqrt(np.mean((y - yh) ** 2, 0)).round(2).tolist(),
                            r2_target_only=[r2(y[:, 0], tgt[:, 0]), r2(y[:, 1], tgt[:, 1])],
                            r2_rates=[r2(g[:, 0], w[: m - 1, 0]), r2(g[:, 1], w[: m - 1, 1])]))
        res["chain"][n] = out
        for o in out:
            print(f"LOG{n} {o['seconds']:6.1f}s  attitude R2 roll {o['r2_roll']:.3f} pitch {o['r2_pitch']:.3f}"
                  f"  rms {o['rms_deg']} deg | target-only R2 {np.round(o['r2_target_only'], 3)}"
                  f" | rates R2 {np.round(o['r2_rates'], 3)}")
    (HERE / "results_angle_chain.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
