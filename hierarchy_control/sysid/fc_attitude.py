"""Reproduce Betaflight 4.4's attitude estimate from the logged gyro + accelerometer,
and identify the ANGLE-mode level loop on top of it.

Betaflight 4.4 does not log its attitude (the `heading[]` columns are Blackbox
Explorer's own reconstruction), so the estimate is checked through what the FC
did with it: in ANGLE mode the PID setpoint is exactly

    setpoint = level_gain * (angle_limit * stick - attitude)        (pid.c pidLevel)

so an estimator that reproduces the FC's attitude makes this hold sample by sample.

The estimator is imu.c's Mahony filter, run like the FC's attitude task:
  * 100 Hz update, gyro (filtered, deg/s) and accelerometer averaged over the period;
  * accelerometer used only while 0.9 g < |acc| < 1.1 g (imuIsAccelerometerHealthy);
  * proportional gain dcm_kp (default 2500 -> 0.25), x10 while disarmed; no integral.
Board frame as logged: x forward, y left, z up (acc reads +1 g z at rest).

    python fc_attitude.py          -> results_attitude.json
"""
import json
from pathlib import Path

import numpy as np

import common as C

HERE = Path(__file__).resolve().parent
ATT_HZ = 100.0


def mahony(gyro_dps, acc_g, armed, kp=0.25, fs=2000.0, q0=None, acc_band=(0.9, 1.1),
           gyro_sign=(1, 1, 1), att_hz=ATT_HZ):
    """Run the FC filter; returns (t_index of each update, quaternion (N, 4) w,x,y,z)."""
    k = int(round(fs / att_hz))
    n = len(gyro_dps) // k
    g = np.radians(gyro_dps[: n * k].reshape(n, k, 3).mean(1)) * np.asarray(gyro_sign)
    a = acc_g[: n * k].reshape(n, k, 3).mean(1)
    arm = armed[: n * k].reshape(n, k)[:, -1]
    dt = k / fs
    q = np.array([1.0, 0, 0, 0]) if q0 is None else np.asarray(q0, float)
    out = np.empty((n, 4))
    lo, hi = acc_band[0] ** 2, acc_band[1] ** 2
    for i in range(n):
        w, x, y, z = q
        # third row of the body->earth rotation matrix (earth z in body coordinates)
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)
        gx, gy, gz = g[i]
        ax, ay, az = a[i]
        m2 = ax * ax + ay * ay + az * az
        if lo < m2 < hi:
            inv = 1 / np.sqrt(m2)
            ax, ay, az = ax * inv, ay * inv, az * inv
            ex = ay * r22 - az * r21
            ey = az * r20 - ax * r22
            ez = ax * r21 - ay * r20
            gain = kp * (1.0 if arm[i] else 10.0)
            gx, gy, gz = gx + gain * ex, gy + gain * ey, gz + gain * ez
        hx, hy, hz = 0.5 * dt * gx, 0.5 * dt * gy, 0.5 * dt * gz
        q = np.array([w - x * hx - y * hy - z * hz,
                      x + w * hx + y * hz - z * hy,
                      y + w * hy - x * hz + z * hx,
                      z + w * hz + x * hy - y * hx])
        q /= np.linalg.norm(q)
        out[i] = q
    return np.arange(n) * k + k - 1, out


def euler_bf(q):
    """imu.c imuUpdateEulerAngles: roll, pitch in degrees, Betaflight's signs."""
    w, x, y, z = q.T
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(r21, r22))
    pitch = np.degrees(0.5 * np.pi - np.arccos(np.clip(-r20, -1, 1)))
    return np.stack([roll, pitch], 1)


def level_fit(L, att, idx, ax, sign):
    """Least squares: setpoint = G*limit*stick - G*sign*att + c, on airborne samples."""
    sp = L["sp"][idx, ax]
    st = L["rc"][idx, ax] / 500.0
    air = L["armed"][idx] & (L["thr"][idx] > 0.12)
    ok = air & ~np.isnan(sp) & ~np.isnan(st)
    A = np.c_[st, -sign * att[:, ax], np.ones(len(st))][ok]
    c, *_ = np.linalg.lstsq(A, sp[ok], rcond=None)
    res = sp[ok] - A @ c
    return dict(r2=float(1 - res.var() / sp[ok].var()), gain=float(c[1]),
                limit=float(c[0] / c[1]), offset_deg=float(c[2] / c[1]),
                rms_dps=float(np.sqrt(np.mean(res ** 2))), n=int(ok.sum()))


def run(n, kp, gyro_sign, att_sign, lag=0):
    L = C.prep(n)
    idx, q = mahony(L["gyro"], L["acc"], L["armed"], kp=kp, fs=L["fs"], gyro_sign=gyro_sign)
    att = euler_bf(q)
    # the level loop uses the attitude of the last completed update: shift by `lag` updates
    k = int(round(L["fs"] / ATT_HZ))
    idx_sp = np.clip(idx + 1 + lag * k, 0, len(L["t"]) - 1)
    return L, q, att, [level_fit(L, att, idx_sp, ax, att_sign[ax]) for ax in (0, 1)]


def search(logs=(3, 6, 7)):
    """Frame signs first (cheap kp), then kp."""
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for ax_s in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                gs = (sx, sy, 1)
                r = [run(n, 0.25, gs, ax_s)[3] for n in logs[:1]]
                s = np.mean([f["r2"] for rr in r for f in rr])
                if best is None or s > best[0]:
                    best = (s, gs, ax_s)
    print("signs: gyro", best[1], "attitude", best[2], "R2", round(best[0], 3))
    return best[1], best[2]


def main():
    gs, asn = search()
    res = {"gyro_sign": gs, "att_sign": asn, "kp": {}}
    for kp in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        r2 = []
        for n in (3, 4, 5, 6, 7):
            r2 += [f["r2"] for f in run(n, kp, gs, asn)[3]]
        res["kp"][kp] = float(np.mean(r2))
        print(f"kp {kp:5.2f}: mean level-loop R2 {np.mean(r2):.3f}  {np.round(r2, 3)}")
    kp = max(res["kp"], key=res["kp"].get)
    res["kp_best"] = kp
    res["logs"] = {}
    for n in (3, 4, 5, 6, 7):
        fits = run(n, kp, gs, asn)[3]
        res["logs"][n] = fits
        print(n, [(round(f["r2"], 3), round(f["gain"], 2), round(f["limit"], 1), round(f["offset_deg"], 2))
                  for f in fits])
    (HERE / "results_attitude.json").write_text(json.dumps(res, indent=1))




def refine(gs=(1, 1, 1), asn=(1, 1)):
    """kp near the optimum, and whether the 0.9-1.1 g acceptance band matters."""
    out = {}
    for kp in (0.15, 0.2, 0.25, 0.3, 0.35):
        out[kp] = float(np.mean([f["r2"] for n in (3, 4, 5, 6, 7) for f in run(n, kp, gs, asn)[3]]))
    print("kp refine", {k: round(v, 4) for k, v in out.items()})
    L = C.prep(6)
    nb = {}
    for band in ((0.9, 1.1), (0.0, 10.0)):
        idx, q = mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=L["fs"], acc_band=band)
        k = int(round(L["fs"] / ATT_HZ))
        nb[str(band)] = [level_fit(L, euler_bf(q), np.clip(idx + 1, 0, len(L["t"]) - 1), ax, 1)["r2"]
                         for ax in (0, 1)]
    print("acc band (LOG6 R2 roll, pitch)", nb)
    return out, nb


def pull(window_s=3.0, logs=(3, 4, 5, 6, 7)):
    """How far the accelerometer correction moves the estimate away from pure gyro
    integration within `window_s`: each window restarts gyro-only integration from the
    FC estimate.  Gyro integration over 3 s is good to ~1 deg (residual bias < 0.4 deg/s),
    so this is the size of the correction, i.e. of the estimate's error when the
    accelerometer is wrong about 'down' (sustained acceleration)."""
    devs, frac_acc = [], []
    for n in logs:
        L = C.prep(n)
        fs = L["fs"]
        k = int(round(fs / ATT_HZ))
        idx, q = mahony(L["gyro"], L["acc"], L["armed"], kp=0.25, fs=fs)
        att = euler_bf(q)
        air = (L["armed"] & (L["thr"] > 0.12))[idx]
        m = np.linalg.norm(L["acc"][: len(idx) * k].reshape(len(idx), k, 3).mean(1), axis=1)
        frac_acc.append(float(np.mean((m[air] > 0.9) & (m[air] < 1.1))))
        w = int(window_s * ATT_HZ)
        for s0 in range(0, len(idx) - w, w):
            if not air[s0:s0 + w].all():
                continue
            a = idx[s0] + 1
            _, qg = mahony(L["gyro"][a:a + w * k], L["acc"][a:a + w * k], L["armed"][a:a + w * k],
                           kp=0.0, fs=fs, q0=q[s0])
            d = np.abs(euler_bf(qg) - att[s0 + 1:s0 + 1 + len(qg)])
            devs.append(d.max(0))
    devs = np.array(devs)
    r = dict(windows=len(devs), window_s=window_s,
             p50=np.percentile(devs, 50, 0).round(2).tolist(),
             p90=np.percentile(devs, 90, 0).round(2).tolist(),
             max=devs.max(0).round(2).tolist(), acc_used_fraction=np.round(frac_acc, 2).tolist())
    print("estimate pull vs gyro-only over", window_s, "s windows (roll, pitch deg):", r)
    return r


def static_tilt():
    """LOG9 at rest: the tilt the FC sees from the accelerometer (mounting + surface)."""
    L = C.prep(9)
    a = np.nanmean(L["acc"], 0)
    r = dict(acc_g=a.round(4).tolist(),
             roll_deg=float(np.degrees(np.arctan2(a[1], a[2]))),
             pitch_deg=float(np.degrees(np.arctan2(-a[0], np.hypot(a[1], a[2])))))
    print("LOG9 static tilt", r)
    return r


if __name__ == "__main__":
    main()
    res = json.loads((HERE / "results_attitude.json").read_text())
    res["kp_refine"], res["acc_band"] = refine()
    res["pull_3s"], res["pull_10s"] = pull(3.0), pull(10.0)
    res["static_log9"] = static_tilt()
    (HERE / "results_attitude.json").write_text(json.dumps(res, indent=1))
