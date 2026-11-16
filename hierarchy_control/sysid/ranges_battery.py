"""Flight envelope seen in the manual flights, and the battery's sag."""
import json
import numpy as np
import common as C

def envelope(logs=(3, 5, 6, 7)):
    R = {}
    for n in logs:
        L = C.prep(n); fs = L["fs"]
        m = L["armed"] & (C.lp(L["thr"], 2, fs) > 0.15)
        g = C.lp(L["gyro"], 30, fs)[m]; a = C.lp(L["acc"], 30, fs)[m]
        roll, pitch = np.degrees(L["raw"]["heading[0]"][m]), np.degrees(L["raw"]["heading[1]"][m])
        R[n] = dict(gyro_p99_dps=np.percentile(np.abs(g), 99, 0).tolist(), gyro_max_dps=np.abs(g).max(0).tolist(),
                    acc_p99_g=np.percentile(np.abs(a), 99, 0).tolist(), acc_max_g=np.abs(a).max(0).tolist(),
                    tilt_p99_deg=float(np.percentile(np.hypot(roll, pitch), 99)), tilt_max_deg=float(np.hypot(roll, pitch).max()),
                    thr_p99=float(np.percentile(L["thr"][m], 99)), airborne_s=float(m.sum() / fs))
        r = R[n]
        print(f"LOG{n}: air {r['airborne_s']:.0f}s | gyro p99 {np.round(r['gyro_p99_dps'])} max {np.round(r['gyro_max_dps'])} dps | "
              f"acc p99 {np.round(r['acc_p99_g'],2)} max {np.round(r['acc_max_g'],2)} g | tilt p99 {r['tilt_p99_deg']:.0f} max {r['tilt_max_deg']:.0f} deg | thr p99 {r['thr_p99']:.2f}")
    return R

def battery(logs=(3, 4, 5, 6, 7)):
    out = {}
    for n in logs:
        L = C.prep(n); fs = L["fs"]
        m = L["armed"] & (L["amp"] > 2)
        if n == 4: m &= L["t"] < 38
        V, I = L["vbat"], L["amp"]
        # sag: fast changes in V against fast changes in I (slow state-of-charge drift removed)
        dV = (V - C.lp(V, 0.3, fs))[m]; dI = (I - C.lp(I, 0.3, fs))[m]
        Rint = -np.sum(dV * dI) / np.sum(dI * dI)
        Voc = V + Rint * I
        mah = np.cumsum(I) / fs / 3.6
        out[n] = dict(R_ohm=Rint, V_start=float(V[m][:int(fs)].mean()), Voc_start=float(Voc[m][:int(fs)].mean()),
                      Voc_end=float(Voc[m][-int(fs):].mean()), V_min=float(V[m].min()), I_max=float(I[m].max()),
                      I_hover_med=float(np.median(I[m])), mAh=float(mah[-1]))
        o = out[n]
        print(f"LOG{n}: R {Rint*1e3:.0f} mOhm | V start {o['V_start']:.2f} (OC {o['Voc_start']:.2f}) -> OC end {o['Voc_end']:.2f} | V min {o['V_min']:.2f} @ I max {o['I_max']:.0f} A | I median {o['I_hover_med']:.1f} A | used {o['mAh']:.0f} mAh")
    return out

if __name__ == "__main__":
    json.dump(dict(envelope=envelope(), battery=battery()), open("results_envelope_battery.json", "w"), indent=1, default=float)
