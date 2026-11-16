import numpy as np
from scipy.signal import butter, filtfilt
import bblog

def lp(x, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)

def prep(n):
    L = bblog.load(n)
    t = L["t"]; fs = 1 / np.median(np.diff(t))
    mn, mx = [int(x) for x in L["h"]["motorOutput"].split(",")]
    L["fs"] = fs
    L["u"] = (L["motor"] - mn) / (mx - mn)          # per-motor command 0..1
    L["thr"] = (L["rc"][:, 3] - 1000) / 1000.0       # throttle stick 0..1
    L["armed"] = (L["raw"]["flightModeFlags"] % 2 == 1)
    return L
