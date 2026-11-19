#!/usr/bin/env python3
"""Run the CTBA curriculum: one DreamerV3 invocation per stage, each resuming
the previous stage's checkpoint and replay in the same logdir.

  stage  steps (cumulative)  task           DreamerV3 settings
  1      0    -> 6M          c1 (complete)  phase 1: batch_length 64
  2      6M   -> 8M          c2 (speed)     phase 2: batch_length 256
  3      8M   -> 14M         c4 (gate)      phase 2
  4      14M  -> 16M         c4             phase 3: actent 1e-5, lr 2e-6

c3 (floor penalty alone) was dropped on 2026-11-19 after the 6.24M evaluation
(ce, 200 flights): 1 lap 92%, 3 laps 66%; floor crashes 1-2% and airframe
outside the opening 0-1%, while gate collisions were 62 of the 68 3-lap failures
-- c1/c2 train on the 0.60 m window, the test is 0.45 m.  c4 already carries
the floor penalty and altitude margin, so its 3M became 6M.

Phase settings are SkyDreamer's (section III-A).  Demonstrations, if any, are
copied into <logdir>/replay before stage 1 (run.sh --demos).

  python scripts/train.py --logdir <run> [--track team44up] [--from-stage N]
                          [--steps-scale 0.001] [--dry-run] [dreamerv3 flags...]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
SKYDREAMER = pathlib.Path(os.environ.get("SKYDREAMER", HERE.parent)).expanduser()
MAIN = SKYDREAMER / "third_party" / "dreamerv3" / "dreamerv3" / "main.py"

PHASE = {
    1: [],
    2: ["--batch_length", "256"],
    3: ["--batch_length", "256", "--agent.imag_loss.actent", "1e-5", "--agent.opt.lr", "2e-6"],
}
SCHEDULE = [
    (6e6, "c1", 1),
    (8e6, "c2", 2),
    (14e6, "c4", 2),
    (16e6, "c4", 3),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--to-stage", type=int, default=len(SCHEDULE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="size12m")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()
    if not MAIN.exists():
        sys.exit(f"{MAIN} missing: run ./run.sh --setup first (SKYDREAMER={SKYDREAMER})")
    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    for i, (steps, stage, phase) in enumerate(SCHEDULE, 1):
        if i < args.from_stage or i > args.to_stage:
            continue
        task = f"ctba_{stage}_{args.track}"
        cmd = [sys.executable, str(MAIN), "--configs", "ctba", args.size,
               "--logdir", str(logdir), "--seed", str(args.seed),
               "--run.steps", f"{steps * args.steps_scale:.0f}", "--task", task,
               *PHASE[phase], *extra]
        print(f"\n=== stage {i}: {task}, phase {phase}, -> {steps * args.steps_scale:,.0f} steps ===\n"
              f"{' '.join(cmd)}\n", flush=True)
        (logdir / "stage").write_text(f"{i}\n")
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
