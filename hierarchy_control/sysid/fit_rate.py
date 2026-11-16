"""Closed-loop body-rate response of Betaflight's rate loop: PID setpoint -> filtered gyro.

gyro(t) = K * PT1_tau( setpoint(t - d) ), per axis.  The setpoint logged by Betaflight 4.4
is the rate the PID controller actually tracked (axisError == setpoint - gyro exactly),
so angle mode during these flights does not matter for this inner loop.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
import common as C

FS = 1000.0

def segments(logs=(3, 4, 5, 6, 7)):
    S = []
    for n in logs:
        L = C.prep(n); fs = L["fs"]; k = int(round(fs / FS))
        air = L["armed"] & (C.lp(L["thr"], 2, fs) > 0.15)
        if n == 4: air &= L["t"] < 38
        # contiguous airborne runs
        e = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
        for a, b in zip(e[::2], e[1::2]):
            if (b - a) / fs < 2: continue
            S.append(dict(n=n, sp=L["sp"][a:b:k, :3], g=L["gyro"][a:b:k]))
    return S

def sim(p, x):
    K, tau, d = p
    s = int(round(d * FS))
    x = np.r_[np.full(s, x[0]), x][: len(x)]
    al = 1 / (1 + tau * FS)
    return K * lfilter([al], [1, al - 1], x, zi=[x[0] * (1 - al)])[0]

def fit_axis(S, ax):
    def res(p):
        return np.concatenate([sim(p, s["sp"][:, ax]) - s["g"][:, ax] for s in S])
    best = None
    for d0 in (0.002, 0.01, 0.02):
        r = least_squares(res, [1.0, 0.02, d0], bounds=([0.3, 0.001, 0], [2, 0.3, 0.08]))
        if best is None or r.cost < best.cost: best = r
    p = best.x
    rr = res(p); y = np.concatenate([s["g"][:, ax] for s in S])
    return p, np.sqrt(np.mean(rr ** 2)), 1 - np.var(rr) / np.var(y), np.std(np.concatenate([s["sp"][:, ax] for s in S]))

if __name__ == "__main__":
    S = segments()
    out = {}
    for ax, name in enumerate(["roll", "pitch", "yaw"]):
        p, rms, r2, spstd = fit_axis(S, ax)
        loo = [fit_axis([s for s in S if s["n"] != n], ax)[0] for n in (3, 4, 5, 6, 7)]
        loo = np.array(loo)
        print(f"{name}: K {p[0]:.3f} tau {p[1]*1e3:.1f} ms delay {p[2]*1e3:.1f} ms | rms {rms:.1f} dps R2 {r2:.3f} | setpoint std {spstd:.1f} dps"
              f" | leave-one-log-out tau {loo[:,1].min()*1e3:.0f}-{loo[:,1].max()*1e3:.0f} ms, delay {loo[:,2].min()*1e3:.0f}-{loo[:,2].max()*1e3:.0f} ms")
        out[name] = dict(K=p[0], tau_s=p[1], delay_s=p[2], rms_dps=rms, R2=r2, setpoint_std_dps=spstd,
                         loo_tau_s=[loo[:, 1].min(), loo[:, 1].max()], loo_delay_s=[loo[:, 2].min(), loo[:, 2].max()])
    json.dump(out, open("results_rate.json", "w"), indent=1, default=float)
