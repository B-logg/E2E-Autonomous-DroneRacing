"""Static collective-thrust map: specific force along body z (g) vs motor commands and battery voltage."""
import json
import numpy as np
from scipy.optimize import least_squares
import common as C

def collect(logs=(3, 4, 5, 6, 7), fc=3.0):
    X, Y, W = [], [], []
    for n in logs:
        L = C.prep(n); fs = L["fs"]; dec = int(fs / 50)
        u = C.lp(L["u"], fc, fs); V = C.lp(L["vbat"], fc, fs); az = C.lp(L["acc"][:, 2], fc, fs)
        gy = np.abs(C.lp(L["gyro"], fc, fs)).max(1)
        tilt = np.abs(C.lp(L["acc"][:, :2], fc, fs)).max(1)
        ground = (np.abs(L["gyro"]).max(1) < 3) & (np.abs(L["acc"][:, 2] - 1) < 0.03) & (L["thr"] < 0.3)
        near_ground = C.lp(ground.astype(float), fc, fs) > 0.05
        ok = L["armed"] & (u.mean(1) > 0.12) & ~near_ground & (gy < 60) & (tilt < 0.6)
        if n == 4:
            ok &= L["t"] < 38          # crash at the end
        i = np.where(ok)[0][::dec]
        X.append(np.c_[u[i], V[i]]); Y.append(az[i]); W.append(np.full(len(i), n))
    return np.vstack(X), np.concatenate(Y), np.concatenate(W)

def model(p, X, vref=22.2):
    w = X[:, :4] * X[:, 4:5] / vref          # command x voltage ~ rpm proxy
    return np.mean(p[0] * w + p[1] * w ** 2, 1)

if __name__ == "__main__":
    X, Y, W = collect()
    print("samples", len(Y), "az range", Y.min().round(2), Y.max().round(2))
    r = least_squares(lambda p: model(p, X) - Y, [1, 10]); res = model(r.x, X) - Y
    print("quad coeffs", r.x.round(4), "rms", np.sqrt((res ** 2).mean()).round(3),
          {int(k): np.sqrt((res[W == k] ** 2).mean()).round(3) for k in np.unique(W)})
    out = {"form": "a_z[g] = mean_i(k1*w_i + k2*w_i^2), w_i = u_i*V/22.2", "k1": r.x[0], "k2": r.x[1],
           "rms_g": float(np.sqrt((res ** 2).mean()))}
    for V in (25.2, 23.0, 22.2, 21.0):
        uh = np.linspace(0.05, 1, 4000); Xh = np.c_[np.tile(uh[:, None], (1, 4)), np.full(4000, V)]
        a = model(r.x, Xh)
        out[f"hover_u@{V}"] = float(uh[np.argmin(abs(a - 1))]); out[f"a_max@{V}"] = float(a[-1])
        print(f"   V={V}: hover u {out[f'hover_u@{V}']:.3f}, a(u=1) {a[-1]:.2f} g")
    # residual vs command level (look for model bias at high throttle)
    um = X[:, :4].mean(1)
    for lo, hi in [(0.1, .25), (.25, .35), (.35, .5), (.5, .8), (.8, 1.01)]:
        m = (um >= lo) & (um < hi)
        if m.any(): print(f"   u {lo}-{hi}: n {m.sum()}, mean res {res[m].mean():+.3f} g")
    json.dump({k: float(v) if not isinstance(v, str) else v for k, v in out.items()},
              open("results_thrust.json", "w"), indent=1)
