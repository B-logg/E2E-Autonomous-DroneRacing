"""Load Betaflight Blackbox CSV exports (Blackbox Explorer) into numpy, with a cache.

Units after load: time s, gyro/setpoint deg/s, acc g, vbat V, current A,
motor raw DShot command (motorOutput range in header), rcCommand raw.
"""
from pathlib import Path
import numpy as np

LOGDIR = Path(__file__).resolve().parents[1] / "manual_flight_logs" / "LOGS"
CACHE = Path(__file__).resolve().parent / "cache"


def header(path):
    h = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith('"loopIteration"'):
                h["_skip"] = i
                break
            k, _, v = line.rstrip("\n").partition(",")
            h[k.strip('"')] = v.strip('"')
    return h


def load(n):
    path = LOGDIR / f"LOG{n:05d}.BFL.csv"
    CACHE.mkdir(exist_ok=True)
    c = CACHE / f"log{n}.npz"
    h = header(path)
    if c.exists() and c.stat().st_mtime > path.stat().st_mtime:
        d = dict(np.load(c))
    else:
        import pandas as pd   # only for the first, uncached read
        df = pd.read_csv(path, skiprows=h["_skip"], low_memory=False)
        df.columns = [x.strip() for x in df.columns]
        d = {k: pd.to_numeric(df[k], errors="coerce").to_numpy(float) for k in df.columns}
        np.savez(c, **d)
    acc1g = float(h.get("acc_1G", 2048))
    out = {"h": h, "raw": d, "t": (d["time"] - d["time"][0]) / 1e6}
    out["gyro"] = np.stack([d[f"gyroADC[{i}]"] for i in range(3)], 1)
    out["acc"] = np.stack([d[f"accSmooth[{i}]"] for i in range(3)], 1) / acc1g
    out["sp"] = np.stack([d[f"setpoint[{i}]"] for i in range(4)], 1)
    out["rc"] = np.stack([d[f"rcCommand[{i}]"] for i in range(4)], 1)
    out["motor"] = np.stack([d[f"motor[{i}]"] for i in range(4)], 1)
    out["vbat"] = d["vbatLatest"] / 100.0
    out["amp"] = d["amperageLatest"] / 100.0
    if "debug[0]" in d:
        out["debug"] = np.stack([d[f"debug[{i}]"] for i in range(4)], 1)
    return out
