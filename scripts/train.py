#!/usr/bin/env python3
"""Drive the paper's three-phase training schedule (section III-A).

DreamerV3 has no mid-run config schedule, so each phase is a separate
invocation resuming from the previous phase's checkpoint in the same logdir.

  small tracks (17M steps, ~50 h on one A100 80GB)
    phase 1   0 ->  8M   defaults
    phase 2   8M -> 13M  batch_length 64 -> 256   (long-horizon parameter ID)
    phase 3  13M -> 17M  actent 3e-4 -> 1e-5, lr 4e-5 -> 2e-6   (fine-tune)

  big track (35M steps; III-D)
    phase boundaries move to 23M and 31M, train_ratio 64, symlog off.

Usage:
  python scripts/train.py --logdir ~/logdir/sd/run1
  python scripts/train.py --logdir ~/logdir/sd/big1 --preset big
  python scripts/train.py --logdir ~/logdir/sd/run1 --from-phase 2   # resume
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "third_party" / "dreamerv3" / "dreamerv3" / "main.py"

PRESETS = {
    "small": dict(configs="skydreamer", boundaries=(8e6, 13e6, 17e6)),
    "big": dict(configs="skydreamer_big", boundaries=(23e6, 31e6, 35e6)),
}

# Phase 2 and 3 overrides, exactly as the paper words them.
PHASE_ARGS = {
    1: [],
    2: ["--batch_length", "256"],
    3: [
        "--batch_length", "256",
        "--agent.imag_loss.actent", "1e-5",
        "--agent.opt.lr", "2e-6",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--from-phase", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hw", action="store_true",
                    help="train on our drone and gate (docs/hardware.md): the hw_* task")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    if not MAIN.exists():
        sys.exit(f"{MAIN} missing -- run scripts/setup_dreamerv3.sh first")

    preset = PRESETS[args.preset]
    if args.hw:
        # `main.py` splits the task on its first underscore: skydreamer_hw_<track>.
        track = "big" if args.preset == "big" else "inverted_loop"
        extra = ["--task", f"skydreamer_hw_{track}", *extra]
    logdir = pathlib.Path(args.logdir).expanduser()

    for phase in (1, 2, 3):
        if phase < args.from_phase:
            continue
        steps = preset["boundaries"][phase - 1]
        cmd = [
            sys.executable, str(MAIN),
            "--configs", preset["configs"], "size12m",
            "--logdir", str(logdir),
            "--seed", str(args.seed),
            # `run.steps` is cumulative, so each phase just raises the ceiling
            # and DreamerV3 resumes from the checkpoint already in the logdir.
            "--run.steps", f"{steps:.0f}",
            *PHASE_ARGS[phase],
            *extra,
        ]
        print(f"\n=== phase {phase} -> {steps:,.0f} steps ===\n{' '.join(cmd)}\n", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
