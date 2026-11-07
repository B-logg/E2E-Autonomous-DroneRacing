#!/usr/bin/env python3
"""Compare checkpoints flight by flight, with the uncertainty that n allows.

    compare_evals.py 7f3M=<dir>[,<dir>...] 7g1.5M=<dir>[,<dir>...] [--laps 1 3]

Each <dir> is an evaluation directory (what `run_test.sh` or `run_compare.sh`
writes): every `evaluation*_<L>lap*_episodes.json` in it is read, so results
from several seeds pool into one estimate.

Why flight by flight.  evaluate.py deals flights from `jax.random.key(seed + k)`:
the same (seed, episode count) gives every checkpoint the same airframes, starts
and noise.  Run 7g's snapshots, scored at seed 10 000, failed the same takeoffs
(flights 43 and 89 under five of six checkpoints) -- the draw, not the policy.
Pairing removes that shared variation; McNemar's exact test then asks only
about the flights where the two checkpoints disagree.  At n = 100 the 95%
interval on a 40% success rate is +-10 points, which is the size of every
difference runs 7f and 7g produced, so a single n = 100 score cannot rank them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import re
import sys
from collections import Counter

import numpy as np

DEFAULT_SEED = 10_000          # evaluate.py's default; files written before it was recorded
SWEPT_LIMIT = 0.6              # m, half the real 1.2 m opening (docs/hardware.md)
# A floor crash this early, before the first gate and below the airborne height
# (clearance 0.10 + reach 0.19), is a takeoff that never got off the floor.
LAUNCH_STEPS, LAUNCH_ALT = 90, 0.29


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: the discordant pairs as Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def is_launch_crash(e: dict) -> bool:
    end = e.get("end")
    return bool(end) and e["gates"] == 0 and end["step"] <= LAUNCH_STEPS and end["alt"] < LAUNCH_ALT


def first_failure(e: dict, n_gates: int = 4) -> str:
    """What ended the flight first: the swept-footprint check at a real gate, or
    the simulator's termination -- whichever happened earlier."""
    if e["success"]:
        return "ok"
    pf, end = e.get("phys_fail"), e.get("end")
    end_step = end["step"] if end else math.inf
    if pf and pf["step"] <= end_step:
        return "swept"
    cause = e["cause"]
    if cause == 2:
        if is_launch_crash(e):
            return "launch"
        k = e["gates"] % n_gates
        return f"ground G{k if k else n_gates}->G{k + 1}" if e["gates"] else "ground ->G1"
    return {0: "timeout", 1: "gate", 3: "diverged", 4: "wall"}.get(cause, f"cause{cause}")


def load(dirs: list[pathlib.Path], laps: int) -> list[dict]:
    """Every flight at `laps` laps in `dirs`, each tagged with its (seed, n, index)."""
    out, seen = [], set()
    pat = re.compile(rf"^evaluation.*_{laps}lap(_.*)?_episodes\.json$")
    for d in dirs:
        for p in sorted(d.glob("evaluation*_episodes.json")):
            if not pat.match(p.name):
                continue
            summ = p.with_name(p.name.replace("_episodes.json", ".json"))
            meta = json.loads(summ.read_text()) if summ.exists() else {}
            recs = json.loads(p.read_text())
            n = len(recs)
            seed = meta.get("seed", DEFAULT_SEED)
            for i, e in enumerate(recs):
                key = (e.get("seed", seed), n, e.get("index", i))
                if key in seen:          # the same file reached twice
                    continue
                seen.add(key)
                e["_key"] = key
                out.append(e)
    return out


def stats(recs: list[dict], n_gates: int = 4) -> dict:
    n = len(recs)
    k = sum(e["success"] for e in recs)
    lo, hi = wilson(k, n)
    first = Counter(first_failure(e, n_gates) for e in recs)
    ev = [x for e in recs for x in e.get("phys_events", [])]
    w = np.array([x[2] for x in ev]) if ev else np.zeros(0)
    g = np.array([x[0] for x in ev], int) if ev else np.zeros(0, int)
    per_gate = {f"G{j + 1}": float((w[g == j] > SWEPT_LIMIT).mean()) if (g == j).any() else None
                for j in range(n_gates)}
    lap = [e["gate_times"][n_gates - 1] for e in recs if len(e["gate_times"]) >= n_gates]
    return dict(
        n=n, success=k / n if n else float("nan"), ci=(lo, hi),
        first={c: v / n for c, v in sorted(first.items())},
        per_crossing_swept=float((w > SWEPT_LIMIT).mean()) if len(w) else None,
        per_gate_swept=per_gate,
        lap_time_median=float(np.median(lap)) if lap else None,
        floor_pen_mean=float(np.mean([e["floor_pen"] for e in recs])) if n else None,
    )


def paired(a: list[dict], b: list[dict]) -> dict | None:
    A = {e["_key"]: e["success"] for e in a}
    B = {e["_key"]: e["success"] for e in b}
    common = sorted(set(A) & set(B))
    if not common:
        return None
    both = sum(A[k] and B[k] for k in common)
    a_only = sum(A[k] and not B[k] for k in common)
    b_only = sum(B[k] and not A[k] for k in common)
    neither = len(common) - both - a_only - b_only
    diff = (a_only - b_only) / len(common)
    # Paired standard error of the difference in proportions.
    n = len(common)
    se = math.sqrt(max(a_only + b_only - (a_only - b_only) ** 2 / n, 0.0)) / n
    return dict(n=n, both=both, a_only=a_only, b_only=b_only, neither=neither,
                diff=diff, ci=(diff - 1.96 * se, diff + 1.96 * se),
                p=mcnemar_p(a_only, b_only))


def fails_everywhere(groups: dict[str, list[dict]], kind: str) -> list[tuple]:
    """Flights that fail the same way under every checkpoint: the draw's doing."""
    sets = []
    for recs in groups.values():
        sets.append({e["_key"] for e in recs if first_failure(e) == kind})
    return sorted(set.intersection(*sets)) if sets else []


def report(groups: dict[str, list[dict]], laps: int) -> str:
    L = [f"=== {laps} lap{'s' if laps > 1 else ''} ==="]
    S = {name: stats(r) for name, r in groups.items()}
    kinds = sorted({c for s in S.values() for c in s["first"]} - {"ok"})
    w = max(len(n) for n in S) + 2
    L.append(f"{'':{w}} {'n':>5} {'success':>8} {'95% CI':>15} {'swept/x':>8} "
             f"{'G1':>5} {'G2':>5} {'G3':>5} {'G4':>5} {'lap s':>6} {'floor':>6}")
    for name, s in S.items():
        g = s["per_gate_swept"]
        pg = " ".join(f"{(v * 100 if v is not None else float('nan')):5.1f}" for v in g.values())
        ci = f"{s['ci'][0]:.0%}-{s['ci'][1]:.0%}"
        L.append(f"{name:{w}} {s['n']:>5} {s['success']:>8.1%} {ci:>15} "
                 f"{(s['per_crossing_swept'] or 0) * 100:>7.1f}% {pg} "
                 f"{s['lap_time_median'] or float('nan'):>6.2f} {s['floor_pen_mean']:>6.2f}")
    L.append("\nfirst failure (share of all flights)")
    L.append(f"{'':{w}} " + " ".join(f"{k:>14}" for k in kinds))
    for name, s in S.items():
        L.append(f"{name:{w}} " + " ".join(f"{s['first'].get(k, 0):>14.1%}" for k in kinds))
    names = list(groups)
    for x, y in itertools.combinations(names, 2):
        pr = paired(groups[x], groups[y])
        if pr is None:
            L.append(f"\n{x} vs {y}: no flights in common (different seeds or counts)")
            continue
        L.append(f"\n{x} vs {y}, paired on {pr['n']} identical flights:"
                 f"  both ok {pr['both']}, only {x} {pr['a_only']}, only {y} {pr['b_only']},"
                 f" neither {pr['neither']}")
        L.append(f"  {x} - {y} = {pr['diff']:+.1%}  (95% CI {pr['ci'][0]:+.1%} to {pr['ci'][1]:+.1%})"
                 f"  McNemar exact p = {pr['p']:.3f}")
    if len(groups) > 1:
        for kind in ("launch",):
            same = fails_everywhere(groups, kind)
            n_common = len(set.intersection(*[{e["_key"] for e in r} for r in groups.values()]))
            if n_common:
                L.append(f"\n{kind} crashes under every checkpoint: {len(same)} of {n_common} flights"
                         f" -- the draw, not the policy" if same else
                         f"\nno flight crashes at {kind} under every checkpoint")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="NAME=DIR[,DIR...]")
    ap.add_argument("--laps", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--out", type=pathlib.Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)
    spec = {}
    for r in args.runs:
        if "=" not in r:
            sys.exit(f"expected NAME=DIR, got {r!r}")
        name, dirs = r.split("=", 1)
        spec[name] = [pathlib.Path(d).expanduser() for d in dirs.split(",")]
    parts = []
    for laps in args.laps:
        groups = {name: load(dirs, laps) for name, dirs in spec.items()}
        empty = [n for n, g in groups.items() if not g]
        if empty:
            parts.append(f"=== {laps} laps === no results for {', '.join(empty)}")
            continue
        parts.append(report(groups, laps))
    text = "\n\n".join(parts)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
