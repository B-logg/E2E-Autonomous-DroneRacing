"""Rate-loop identification in the frequency domain (1-10 Hz, coherence-weighted).

Models: (a) PT1 + delay, (b) 2nd order + delay.  The time-domain fit (fit_rate.py) is
dominated by the large slow content below 2 Hz, where any small tau fits; the phase at
4-10 Hz is what pins the lag.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import csd, welch
import fit_rate as R

def frf(S, ax, nper=2048, fs=1000.0):
    Pxy = Pxx = Pyy = 0
    for s in S:
        x, y = s["sp"][:, ax], s["g"][:, ax]
        if len(x) < 2 * nper: continue
        f, a = csd(x, y, fs=fs, nperseg=nper); _, b = welch(x, fs=fs, nperseg=nper); _, c = welch(y, fs=fs, nperseg=nper)
        Pxy, Pxx, Pyy = Pxy + a, Pxx + b, Pyy + c
    return f, Pxy / Pxx, np.abs(Pxy) ** 2 / (Pxx * Pyy)

def H1(p, f):
    K, tau, d = p; s = 2j * np.pi * f
    return K * np.exp(-s * d) / (1 + s * tau)

def H2(p, f):
    K, wn, z, d = p; s = 2j * np.pi * f
    return K * wn ** 2 * np.exp(-s * d) / (s ** 2 + 2 * z * wn * s + wn ** 2)

def fit(f, H, coh, model, p0, lb, ub, band=(0.8, 10.0)):
    m = (f >= band[0]) & (f <= band[1]) & (coh > 0.3)
    w = np.sqrt(coh[m])
    r = least_squares(lambda p: np.r_[(model(p, f[m]) - H[m]).real * w, (model(p, f[m]) - H[m]).imag * w], p0, bounds=(lb, ub))
    e = model(r.x, f[m]) - H[m]
    return r.x, float(np.sqrt(np.mean(np.abs(e) ** 2)))

if __name__ == "__main__":
    S = R.segments(); out = {}
    for ax, name in enumerate(["roll", "pitch", "yaw"]):
        f, H, coh = frf(S, ax)
        p1, e1 = fit(f, H, coh, H1, [1, .02, .005], [.5, .001, 0], [2, .2, .05])
        p2, e2 = fit(f, H, coh, H2, [1, 60, .5, .005], [.5, 5, .05, 0], [2, 400, 2, .05])
        bw = f[(f > 0.5)][np.argmax(np.abs(H2(p2, f[f > 0.5])) < 0.707)]
        print(f"{name}: PT1+delay K {p1[0]:.2f} tau {p1[1]*1e3:.1f} ms d {p1[2]*1e3:.1f} ms (err {e1:.3f}) | "
              f"2nd: K {p2[0]:.2f} fn {p2[1]/2/np.pi:.1f} Hz zeta {p2[2]:.2f} d {p2[3]*1e3:.1f} ms (err {e2:.3f}) | -3dB ~{bw:.1f} Hz")
        # per-log spread of the PT1 fit
        spread = []
        for n in (3, 4, 5, 6, 7):
            try:
                fn, Hn, cn = frf([s for s in S if s["n"] == n], ax)
                spread.append(fit(fn, Hn, cn, H1, [1, .02, .005], [.5, .001, 0], [2, .2, .05])[0])
            except Exception:
                pass
        spread = np.array(spread)
        print(f"     per-log PT1: tau {np.round(spread[:,1]*1e3,1)} ms, d {np.round(spread[:,2]*1e3,1)} ms")
        out[name] = dict(pt1=dict(K=p1[0], tau_s=p1[1], delay_s=p1[2], err=e1),
                         second=dict(K=p2[0], fn_hz=p2[1] / 2 / np.pi, zeta=p2[2], delay_s=p2[3], err=e2),
                         bw_3db_hz=float(bw), per_log_tau_s=spread[:, 1].tolist(), per_log_delay_s=spread[:, 2].tolist())
    json.dump(out, open("results_rate_freq.json", "w"), indent=1, default=float)
