"""Sensor noise as the policy would see it.

'Noise' = content above FC Hz (what a rigid-body simulator does not produce), measured on
the filtered gyro/acc the FC exposes over MSP, then sampled at the policy rate (aliasing
included, by plain decimation without anti-alias filtering, as the Jetson would read it).
"""
import json
import numpy as np
from scipy.signal import welch
import common as C

FC = 15.0

def run(logs, label, sel):
    out = {}
    G, A = [], []
    for n in logs:
        L = C.prep(n); fs = L["fs"]
        m = sel(L)
        g = L["gyro"] - C.lp(L["gyro"], FC, fs)
        a = L["acc"] - C.lp(L["acc"], FC, fs)
        G.append((g, m, fs)); A.append((a, m, fs))
    for rate in (100, 200, 300, 1000):
        gs, as_ = [], []
        for (g, m, fs), (a, _, _) in zip(G, A):
            k = int(round(fs / rate)); idx = np.arange(0, len(m), k); idx = idx[m[idx]]
            gs.append(g[idx]); as_.append(a[idx])
        gs, as_ = np.vstack(gs), np.vstack(as_)
        out[rate] = dict(gyro_std_dps=gs.std(0).tolist(), acc_std_g=as_.std(0).tolist())
        print(f"{label} @{rate} Hz: gyro std {gs.std(0).round(2)} dps  acc std {as_.std(0).round(3)} g")
    # dominant gyro vibration peaks (filtered gyro, 2 kHz)
    g, m, fs = G[0]
    f, P = welch(g[m], fs=fs, nperseg=4096, axis=0)
    pk = [f[np.argsort(P[:, i])[-3:]][::-1].round(0).tolist() for i in range(3)]
    print(f"   {label} gyro spectral peaks (Hz) roll/pitch/yaw: {pk}")
    out["peaks_hz"] = pk
    return out

if __name__ == "__main__":
    def hover(L):
        fs = L["fs"]; g = C.lp(L["gyro"], 3, fs)
        return L["armed"] & (C.lp(L["thr"], 2, fs) > 0.2) & (np.abs(g).max(1) < 15)
    def flight(L):
        fs = L["fs"]; m = L["armed"] & (C.lp(L["thr"], 2, fs) > 0.15)
        return m & (L["t"] < 38) if L["h"].get("_n") == 4 else m
    res = {"hover": run((3, 5, 6), "hover", hover), "track": run((6, 7), "flight", flight)}
    json.dump(res, open("results_noise.json", "w"), indent=1, default=float)
