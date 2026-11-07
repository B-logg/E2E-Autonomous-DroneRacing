#!/usr/bin/env python3
"""Why flights end on the floor: takeoff crashes and in-flight floor strikes.

    diagnose_failures.py <eval dir> [--laps 3] [--json out.json]

Needs the per-flight trajectories (`evaluate.py --dump`, which `run_compare.sh`
passes): true p, v, q, the motor commands u and the decoded position dp.

Runs 7f and 7g lost 4-7% of their flights on the floor before they ever left
it, and 6-14% on it between G3 and G4.  The episode summaries give where and
in what state each ended, not why.  The trajectories separate the explanations
that call for different fixes.

**Takeoff** (on the floor until the airframe is clear of it, env._floor_contact):
a prop tip touches when the airframe tilts more than asin(alt / reach) -- 22 deg
at 7 cm.  The question is whether the policy leans that far as a habit (every
takeoff close to the edge, crashes the tail) or a few draws push it over (the
airframe's moment coefficients, its thrust-to-weight, the start).

**In flight**, at the point of no return -- from where even full thrust, after
the action delay, cannot stop the sink above the crash height and the flight
never got back out of it -- one of:

  late       already near full throttle: it was pulling as hard as it could;
             what doomed it came earlier (entry height and sink, or a bank
             that spent the thrust sideways -- see the tilt row)
  estimate   not at full throttle, and it decoded itself >= ESTIMATE_BIAS m
             higher than it was: it did not know
  policy     not at full throttle and knew its height: it chose to keep sinking

The first wants more margin earlier (the gate line, the floor penalty), the
second a better altitude estimate, the third a reward the policy is not
respecting.  Finishers' lowest point in the same segment is the comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compare_evals import first_failure, is_launch_crash  # noqa: E402

G = 9.81
try:                  # the simulator's own numbers, when it is importable
    from skydreamer import hardware as _hw, track as _trk
    from skydreamer.dynamics import CONTROL_DT as DT
    REACH, CLEARANCE, LAG = _hw.REACH, _trk.FLOOR["team"], _trk.FLOOR_LAG
except ImportError:   # a laptop without the venv: the same values, copied
    REACH = 0.19      # m, centre to prop tip (hardware.REACH)
    CLEARANCE = 0.10  # m, crash height of the lowest tip (track.FLOOR["team"])
    LAG = 0.04        # s, action delay + spin-up (track.FLOOR_LAG)
    DT = 1.0 / 90.0   # s, control step
FULL_THROTTLE = 0.8   # mean motor command at or above which the policy is pulling up hard
ESTIMATE_BIAS = 0.15  # m, decoded minus true altitude that counts as "did not know"
# about the decode error at the start of a flight (whole-flight p50 ~0.14 m)


def tilt_deg(q: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and the vertical, from [w, x, y, z]."""
    x, y = q[..., 1], q[..., 2]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def lowest_tip(p: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    return -p[..., 2] - REACH * np.sin(np.radians(tilt))


# The stop is the one track.floor_penalty's reach assumes (FLOOR_DECEL: a median
# airframe banked 35 deg), with this flight's own thrust-to-weight: the airframe
# is free to level out first, so its momentary bank does not count against it.
STOP_BANK = 35.0      # deg


def stop_distance(sink: np.ndarray, twr: float) -> np.ndarray:
    """Height lost stopping a sink of `sink` m/s (NED, + down) at full thrust
    banked STOP_BANK, after the action delay."""
    decel = max((twr * np.cos(np.radians(STOP_BANK)) - 1.0) * G, 1e-6)
    s = np.maximum(sink, 0.0)
    return s * LAG + s * s / (2 * decel)


def point_of_no_return(p, v, tilt, twr) -> int:
    """First step of the final stretch in which the floor could no longer be
    avoided: the stop needs more height than the lowest tip has above the crash
    clearance.  The crash step itself always counts."""
    room = lowest_tip(p, tilt) - CLEARANCE
    doomed = stop_distance(v[:, 2], twr) >= room
    t = len(p) - 1
    while t > 0 and doomed[t - 1]:
        t -= 1
    return t


def load(d: pathlib.Path, laps: int):
    stem = next((p.name[: -len("_episodes.json")] for p in sorted(d.glob("evaluation*_episodes.json"))
                 if f"_{laps}lap" in p.name and (d / p.name.replace("_episodes.json", "_traj.npz")).exists()),
                None)
    if stem is None:
        sys.exit(f"no {laps}-lap episodes with a _traj.npz in {d} -- evaluate with --dump")
    recs = json.loads((d / f"{stem}_episodes.json").read_text())
    z = np.load(d / f"{stem}_traj.npz")
    traj = [{k: z[f"{k}_{i}"] for k in ("p", "v", "q", "u", "dp") if f"{k}_{i}" in z}
            for i in range(len(recs))]
    return stem, recs, traj


def takeoff(e, t) -> dict:
    p, q, u = t["p"], t["q"], t.get("u")
    tilt = tilt_deg(q)
    alt = -p[:, 2]
    clear = np.nonzero(alt > CLEARANCE + REACH)[0]
    n = int(clear[0]) if len(clear) else len(p)
    seg = slice(0, max(n, 1))
    out = dict(steps_to_clear=n if len(clear) else None,
               min_tip=float(lowest_tip(p[seg], tilt[seg]).min()),
               max_tilt=float(tilt[seg].max()),
               tilt_at_end=float(tilt[min(n, len(p)) - 1]))
    if u is not None:
        uu = u[seg]
        out["throttle_mean"] = float(uu.mean())
        # roll/pitch command: the motor-pair differences the mixer turns into torque
        out["u_spread"] = float((uu.max(1) - uu.min(1)).mean())
    return out


def floor_strike(e, t) -> dict:
    p, v, q, u, dp = t["p"], t["v"], t["q"], t.get("u"), t.get("dp")
    tilt = tilt_deg(q)
    k = point_of_no_return(p, v, tilt, e["twr"])
    out = dict(pnr_step=k, pnr_before_s=(len(p) - 1 - k) * DT,
               alt=float(-p[k, 2]), sink=float(v[k, 2]), tilt=float(tilt[k]),
               tilt_end=float(tilt[-1]), twr=float(e["twr"]))
    if u is not None:
        out["throttle"] = float(u[k].mean())
    # dp[t] was decoded before step t, i.e. of p[t-1]
    j = k + 1 if k + 1 < len(p) else k
    if dp is not None and j >= 1 and np.isfinite(dp[j]).all():
        out["dec_alt_bias"] = float(-dp[j, 2] - (-p[j - 1, 2]))
    if out.get("throttle", 0.0) >= FULL_THROTTLE:
        out["kind"] = "late"
    elif out.get("dec_alt_bias", 0.0) >= ESTIMATE_BIAS:
        out["kind"] = "estimate"
    else:
        out["kind"] = "policy"
    return out


def segment_low(e, t, seg_gate: int, n_gates: int = 4) -> list[dict]:
    """The lowest point of every pass from gate `seg_gate` to the next."""
    times = e["gate_times"]
    res = []
    for j, gt in enumerate(times):
        if j % n_gates != seg_gate - 1 or j + 1 >= len(times):
            continue
        # gate_times are (t + 1) * DT; trajectory row t is the state after step t
        a, b = int(round(gt / DT)) - 1, int(round(times[j + 1] / DT)) - 1
        p, q = t["p"][a:b], t["q"][a:b]
        if len(p) == 0:
            continue
        tip = lowest_tip(p, tilt_deg(q))
        i = int(tip.argmin())
        r = dict(min_tip=float(tip[i]), sink=float(t["v"][a + i, 2]))
        if t.get("dp") is not None and a + i + 1 < len(t["dp"]):
            r["dec_alt_bias"] = float(-t["dp"][a + i + 1, 2] + p[i, 2])
        if t.get("u") is not None:
            r["throttle"] = float(t["u"][a + i].mean())
        res.append(r)
    return res


def _q(x, f="{:.2f}"):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return "-"
    a = np.percentile(x, [10, 50, 90])
    return " / ".join(f.format(v) for v in a)


def param_effects(recs, bad, good, top=6):
    """Which of the flight's draws separate the bad takeoffs from the good: the
    rank-biserial correlation (+1: every bad flight above every good one)."""
    keys = [k for k, v in recs[0].get("params", {}).items() if isinstance(v, float)]
    feats = {k: [r["params"][k] for r in recs] for k in keys}
    feats["twr"] = [r["twr"] for r in recs]
    if recs[0].get("start"):
        yaw = np.array([r["start"]["yaw_deg"] for r in recs])
        feats["start_yaw_dev"] = list(yaw - np.median(yaw))
        p0 = np.array([r["start"]["p"] for r in recs])
        feats["start_x_dev"] = list(p0[:, 0] - np.median(p0[:, 0]))
        feats["start_y_dev"] = list(p0[:, 1] - np.median(p0[:, 1]))
    out = []
    for k, x in feats.items():
        x = np.asarray(x, float)
        xb, xg = x[bad], x[good]
        if len(xb) == 0 or len(xg) == 0 or np.ptp(x) == 0:
            continue
        gt = (xb[:, None] > xg[None, :]).mean() - (xb[:, None] < xg[None, :]).mean()
        out.append((abs(gt), k, gt, float(np.median(xb)), float(np.median(xg))))
    return sorted(out, reverse=True)[:top]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("evaldir", type=pathlib.Path)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    stem, recs, traj = load(args.evaldir.expanduser(), args.laps)
    n = len(recs)
    L = [f"=== {args.evaldir.name} / {stem}: {n} flights ==="]
    report = {}

    # ---- takeoff ----
    tk = [takeoff(e, t) for e, t in zip(recs, traj)]
    bad = [i for i, e in enumerate(recs) if is_launch_crash(e)]
    good = [i for i, x in enumerate(tk) if x["steps_to_clear"] is not None]
    L.append(f"\n--- takeoff: {len(bad)} crashed on the floor ({len(bad) / n:.1%}), "
             f"{len(good)} cleared it ---")
    L.append("                     p10 / p50 / p90")
    for key, lab in (("max_tilt", "max tilt [deg]"), ("min_tip", "lowest tip [m]"),
                     ("throttle_mean", "mean command"), ("u_spread", "motor spread")):
        L.append(f"  {lab:18} crashed {_q([tk[i].get(key) for i in bad]):>22}"
                 f"   cleared {_q([tk[i].get(key) for i in good]):>22}")
    L.append(f"  steps to clear, cleared: {_q([tk[i]['steps_to_clear'] for i in good], '{:.0f}')}")
    near = sum(tk[i]["min_tip"] < 0.02 for i in good)
    L.append(f"  cleared flights whose tip came within 2 cm of the floor: {near} of {len(good)}")
    eff = param_effects(recs, bad, good) if bad and good else []
    if eff:
        L.append("  draws that separate crashed from cleared (rank-biserial, median crashed / cleared):")
        for _, k, r, mb, mg in eff:
            L.append(f"    {k:14} {r:+.2f}   {mb:.4g} / {mg:.4g}")
    report["takeoff"] = dict(crashed=[recs[i].get("index", i) for i in bad],
                             crashed_detail=[tk[i] for i in bad], near_misses=near,
                             cleared=len(good), effects=[e[1:] for e in eff])

    # ---- in-flight floor strikes ----
    strikes = {}
    for i, e in enumerate(recs):
        kind = first_failure(e)
        if kind.startswith("ground"):
            strikes.setdefault(kind, []).append(i)
    report["floor"] = {}
    for kind, idx in sorted(strikes.items()):
        d = [floor_strike(recs[i], traj[i]) for i in idx]
        seg = int(kind.split("G")[1].split("-")[0]) if "G" in kind.split("->")[0] else None
        L.append(f"\n--- {kind}: {len(idx)} flights ({len(idx) / n:.1%}) ---")
        cnt = {k: sum(x["kind"] == k for x in d) for k in ("late", "estimate", "policy")}
        L.append("  at the point of no return:  " + "  ".join(f"{k} {v}" for k, v in cnt.items()))
        L.append(f"  how long before the crash [s]  {_q([x['pnr_before_s'] for x in d])}")
        L.append(f"  altitude [m]                   {_q([x['alt'] for x in d])}")
        L.append(f"  sink [m/s, +down]              {_q([x['sink'] for x in d])}")
        L.append(f"  tilt [deg]                     {_q([x['tilt'] for x in d], '{:.0f}')}")
        L.append(f"  mean motor command             {_q([x.get('throttle') for x in d])}")
        L.append(f"  decoded - true altitude [m]    {_q([x.get('dec_alt_bias') for x in d])}")
        L.append(f"  thrust-to-weight               {_q([x['twr'] for x in d])}")
        fin = []
        if seg:
            for i, e in enumerate(recs):
                if first_failure(e) == "ok":
                    fin += segment_low(e, traj[i], seg)
            if fin:
                L.append(f"  finishers' lowest point G{seg}->G{seg % 4 + 1} ({len(fin)} passes):")
                L.append(f"    lowest tip [m]  {_q([f['min_tip'] for f in fin])}   "
                         f"sink {_q([f['sink'] for f in fin])}   "
                         f"command {_q([f.get('throttle') for f in fin])}   "
                         f"decoded - true {_q([f.get('dec_alt_bias') for f in fin])}")
        report["floor"][kind] = dict(flights=[recs[i].get("index", i) for i in idx], detail=d,
                                     finishers=fin)
    text = "\n".join(L)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
