#!/usr/bin/env python3
"""Read the `.npz` files `diagnose_start.py` writes and answer: which episodes
die early, and what do they have in common?

Plain numpy/pandas.  Nothing here needs JAX.

    python scripts/analyse_start.py mixed.npz [eval.npz ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

CAUSE = {0: "survived/timeout", 1: "hit gate", 2: "hit ground", 3: "diverged"}


def load(path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    n = len(z["gates"])
    dt = float(z["dt"])
    euler0 = z["euler0"]
    df = pd.DataFrame(
        dict(
            start_gate=z["start_gate"],
            alt0=z["alt0"],
            x_g=z["pg0"][:, 0], y_g=z["pg0"][:, 1], z_g=z["pg0"][:, 2],
            roll0=np.degrees(euler0[:, 0]), pitch0=np.degrees(euler0[:, 1]),
            yaw_rel0=np.degrees(z["info_att0"][:, 3]),
            rate0=np.abs(z["rates0"]).sum(1),
            motor0=z["motor0"].mean(1),
            motor0_min=z["motor0"].min(1), motor0_max=z["motor0"].max(1),
            col=np.where(np.isclose(z["d_g"], 0.8, atol=0.05), "train", "eval"),
            d_g=z["d_g"], t_g=z["t_g"], tau=z["tau"], k_w=z["k_w"], w_max=z["w_max_ep"],
            gyro_sigma=z["gyro_sigma"],
            eps_u=z["bounds"][:, 3], eps_M_fast=z["bounds"][:, 2],
            gates=z["gates"], cause=z["cause"], finished=z["finished"],
            death_s=np.where(z["death_t"] >= 0, (z["death_t"] + 1) * dt, np.nan),
            alt_end=-z["p_end"][:, 2], vz_end=z["v_end"][:, 2],
            roll_end=np.degrees(z["euler_end"][:, 0]), pitch_end=np.degrees(z["euler_end"][:, 1]),
            local_y=z["local_end"][:, 1], local_z=z["local_end"][:, 2], local_x=z["local_end"][:, 0],
            plane_end=z["plane_end"],
        )
    )
    # Thrust-to-weight at t=0 from the initial rotor speeds (steady-state formula).
    df["tw0"] = df["k_w"].values * ((z["motor0"] * 3100.0) ** 2).sum(1) / 9.81
    df["twr_max"] = df["k_w"] * 4 * df["w_max"] ** 2 / 9.81
    df["ser_alt"] = list(z["ser_alt"])
    df["ser_vz"] = list(z["ser_vz"])
    df["ser_tilt"] = list(z["ser_tilt"])
    df["ser_motor"] = list(z["ser_motor"])
    df["ser_speed"] = list(z["ser_speed"])
    df["ucmd"] = list(z["ucmd"])
    df["gate_hist"] = list(z["gate_hist"])
    df.attrs["dt"] = dt
    df.attrs["protocol"] = str(z["protocol"])
    return df


def bucket(row) -> str:
    if row.finished:
        return "success"
    if row.cause == 0:
        return "timeout"
    if row.gates <= 1:
        return "early(0-1)"
    if row.gates <= 3:
        return "lap1(2-3)"
    return "late(4+)"


def rate_table(df, col, bins=None, labels=None, target="early"):
    d = df.copy()
    d["k"] = d["bucket"].isin(target if isinstance(target, list) else [target])
    key = pd.cut(d[col], bins) if bins is not None else d[col]
    g = d.groupby(key, observed=True)["k"].agg(["mean", "sum", "count"])
    g["mean"] = (100 * g["mean"]).round(1)
    return g.rename(columns={"mean": "rate%", "sum": "bad", "count": "n"})


def main(paths):
    for p in paths:
        df = load(p)
        df["bucket"] = df.apply(bucket, axis=1)
        n = len(df)
        print("=" * 78)
        print(f"{p}   protocol={df.attrs['protocol']}   N={n}")
        print(df["bucket"].value_counts().reindex(
            ["success", "timeout", "late(4+)", "lap1(2-3)", "early(0-1)"]).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
