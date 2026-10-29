#!/usr/bin/env python3
"""Keep a copy of the checkpoint every N steps while a run trains.

The trainer keeps exactly one checkpoint and replaces it every 15 minutes, so a
point in the run nobody evaluated in time is gone for good -- run 6's 7M, the
step just before its biggest jump, was lost that way.  This copies the first
checkpoint past each multiple of `--every` to `<logdir>/snapshots/step_<N>/`,
in the same evaluable form `./run_test.sh` makes (`scripts/snapshot_ckpt.py`):
point `scripts/evaluate.py --logdir` at one to score it, during or after the run.

It only reads `ckpt/` and copies files; the trainer never waits on it.  run.sh
starts it next to the replay pruner and stops it when training ends.  Nothing
here is ever pruned: ~126 MB per snapshot, ~1.3 GB for a 10M run at 1M.

  python scripts/milestone_snapshots.py --logdir <run> [--every 1e6] [--interval 300]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from snapshot_ckpt import latest_complete, read_step, snapshot  # noqa: E402


def done_milestones(out_root: pathlib.Path, every: int) -> int:
    """Highest milestone already copied, so a resumed run does not redo them."""
    steps = [int(p.name[5:]) for p in out_root.glob("step_*")
             if p.name[5:].isdigit() and (p / "ckpt" / "latest").exists()]
    return max((s // every for s in steps), default=0)


def poll(logdir: pathlib.Path, out_root: pathlib.Path, every: int, last: int) -> int:
    """One check: copy the checkpoint if it has crossed a new milestone.
    Returns the milestone now reached."""
    try:
        step = read_step(latest_complete(logdir / "ckpt"))
    except (SystemExit, FileNotFoundError, OSError, EOFError, ValueError):
        return last   # no checkpoint yet, or one being written: try again later
    if step // every <= last:
        return last
    try:
        _, step = snapshot(logdir, out_root)
    except SystemExit as e:   # it kept rotating under us; the next poll retries
        print(f"snapshot at step {step:,} failed: {e}", flush=True)
        return last
    return step // every


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--every", type=float, default=1e6)
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between checks")
    ap.add_argument("--once", action="store_true", help="check once and exit (testing)")
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    every = int(args.every)
    out_root = logdir / "snapshots"
    last = done_milestones(out_root, every)
    print(f"snapshotting {logdir} every {every:,} steps -> {out_root} "
          f"(milestones done: {last})", flush=True)
    while True:
        last = poll(logdir, out_root, every, last)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
