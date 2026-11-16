"""Dynamic collective-thrust fit on throttle events (punches and chops).

a_meas(t) = PT1_tau( f(u(t-d), V(t)) ) - c_v * v_z(t)
f = mean_i(k1*w + k2*w^2), w = u_i*V/22.2;  v_z from integrating the measured
specific force from rest at window start (vertical, small tilt).  tau lumps motor
spin-up and the FC accelerometer low-pass, since the policy sees that same filtered signal.
"""
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
import common as C

FS = 500.0

def events(n, L):
    fs = L["fs"]; thr = L["thr"]; t = L["t"]
    d = np.abs(np.diff(C.lp(thr, 20, fs))) * fs        # throttle slew, 1/s
    big = np.where(d > 3.0)[0]                           # fast stick moves
    out = []; last = -1e9
    for i in big:
        if t[i] - last < 1.5: continue
        a, b = int(i - 0.4 * fs), int(i + 1.2 * fs)
        if a < 0 or b >= len(t): continue
        seg = slice(a, b)
        if not L["armed"][seg].all(): continue
        if np.abs(L["gyro"][seg]).max() > 150: continue
        if np.abs(C.lp(L["acc"][seg, :2], 10, fs)).max() > 0.5: continue
        if thr[seg].max() < 0.45 and thr[seg].min() > 0.15: continue  # want big moves only
        if n == 4 and t[b] > 38: continue
        out.append((a, b)); last = t[i]
    return out

def windows(logs=(4, 5, 7)):
    W = []
    for n in logs:
        L = C.prep(n); fs = L["fs"]; k = int(round(fs / FS))
        for a, b in events(n, L):
            u = L["u"][a:b:k]; V = C.lp(L["vbat"], 30, fs)[a:b:k]; az = C.lp(L["acc"][:, 2], 60, fs)[a:b:k]
            vz = np.cumsum((az - 1) * 9.81) / FS
            W.append(dict(n=n, t0=L["t"][a], u=u, V=V, az=az, vz=vz))
    return W

def sim(p, w):
    k1, k2, tau, dly, cv, be = p
    s = int(round(dly * FS))
    u = np.r_[np.repeat(w["u"][:1], s, 0), w["u"]][: len(w["u"])]
    x = u * (w["V"][:, None] / 22.2) ** be
    f = np.mean(k1 * x + k2 * x ** 2, 1)
    al = 1 / (1 + tau * FS)
    y = lfilter([al], [1, al - 1], f, zi=[f[0] * (1 - al)])[0]
    return y - cv * w["vz"]

HOVER = None

def hover_samples():
    """Near-hover samples from the hover/track flights (LOG3, LOG6): v_z ~ 0, slow."""
    import fit_thrust as FT
    X, Y, Wn = FT.collect(logs=(3, 6))
    m = np.abs(Y - 1) < 0.15
    return X[m], Y[m]

BETA = 1.0      # thrust ~ (u V)^.. : voltage exponent fixed (physical); free it and the
                # punch windows alone drive it to 0 while hover data need ~1 (see README)
W_HOVER = 4.0   # hover samples weighted 4x all event samples together

def fit(W, w_hover=W_HOVER):
    global HOVER
    if HOVER is None: HOVER = hover_samples()
    Xh, Yh = HOVER
    nev = sum(len(w["az"]) for w in W)
    sc = np.sqrt(w_hover * nev / len(Yh))          # hover set weighted like all events together
    def res(p):
        x = Xh[:, :4] * (Xh[:, 4:5] / 22.2) ** p[5]
        rh = (np.mean(p[0] * x + p[1] * x ** 2, 1) - Yh) * sc
        return np.concatenate([sim(p, w) - w["az"] for w in W] + [rh])
    best = None
    for d0 in (0.0, 0.01, 0.02):
        r = least_squares(lambda q: res(np.r_[q, BETA]), [2.7, 1.5, 0.03, d0, 0.05], bounds=([0, 0, 0.002, 0, 0], [20, 20, 0.2, 0.06, 1]))
        r.x = np.r_[r.x, BETA]
        if best is None or r.cost < best.cost: best = r
    return best

if __name__ == "__main__":
    W = windows()
    print("events", len(W), [(w["n"], round(w["t0"], 1)) for w in W])
    r = fit(W); p = r.x
    rms = np.sqrt(np.mean(np.concatenate([(sim(p, w) - w["az"]) ** 2 for w in W])))
    Xh, Yh = HOVER; x = Xh[:, :4] * (Xh[:, 4:5] / 22.2) ** p[5]
    print("hover-set rms %.3f g, mean res %+.3f g, n %d" % (np.sqrt(np.mean((np.mean(p[0]*x+p[1]*x**2,1)-Yh)**2)), np.mean(np.mean(p[0]*x+p[1]*x**2,1)-Yh), len(Yh)))
    print("k1 %.3f k2 %.3f tau %.1f ms delay %.1f ms cv %.3f g/(m/s) beta %.2f  rms %.3f g" % (p[0], p[1], p[2]*1e3, p[3]*1e3, p[4], p[5], rms))
    # leave-one-log-out
    for n in (4, 5, 7):
        q = fit([w for w in W if w["n"] != n]).x
        print(f"  without LOG{n}: k1 {q[0]:.2f} k2 {q[1]:.2f} tau {q[2]*1e3:.0f} ms d {q[3]*1e3:.0f} ms cv {q[4]:.3f} beta {q[5]:.2f}")
    out = dict(k1=p[0], k2=p[1], tau_s=p[2], delay_s=p[3], cv_g_per_mps=p[4], beta=p[5], rms_g=rms, n_events=len(W))
    for V in (25.2, 23.0, 22.2, 21.0):
        uh = np.linspace(0.05, 1, 4000); x = uh * (V / 22.2) ** p[5]; a = p[0] * x + p[1] * x ** 2
        out[f"hover_u@{V}"] = float(uh[np.argmin(abs(a - 1))]); out[f"a_max@{V}"] = float(a[-1])
        print(f"   V={V}: hover u {out[f'hover_u@{V}']:.3f}, static a(u=1) {a[-1]:.2f} g")
    json.dump({k: float(v) for k, v in out.items()}, open("results_thrust_dyn.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(W), 1, figsize=(10, 2.2 * len(W)))
    for a_, w in zip(np.atleast_1d(ax), W):
        tt = np.arange(len(w["az"])) / FS
        a_.plot(tt, w["az"], label="meas"); a_.plot(tt, sim(p, w), label="model"); a_.plot(tt, w["u"].mean(1) * 5, label="u*5", alpha=.5)
        a_.set_title(f"LOG{w['n']} t0={w['t0']:.1f}", fontsize=8); a_.legend(fontsize=6)
    plt.tight_layout(); plt.savefig("fig_thrust_dyn.png", dpi=60)
