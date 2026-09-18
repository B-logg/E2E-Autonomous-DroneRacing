#!/usr/bin/env python3
"""Keep the on-disk replay directory bounded.

DreamerV3 writes every replay chunk to `logdir/replay/*.npz` and **never
deletes any of them**, so the directory grows with the whole run rather than
with the buffer window.  Measured on a real run: 3782 bytes/step on disk, so
17M steps is ~64 GB of chunks even though the live buffer is only
`replay.size` steps.

Deleting the oldest files is safe.  `Replay.load()` sorts filenames newest
first and reads only `capacity` steps' worth; anything older is already dead
weight.  Sequences walk forward through `chunk.succ`, so the retained
newest-N set is closed under "successor" and stays self-consistent.

Run it once to reclaim space, or with `--interval` to keep pruning during a
long run (which is what run.sh does).
"""

from __future__ import annotations

import argparse
import pathlib
import time

# DreamerV3's `replay.chunksize` default; chunks hold this many steps each.
CHUNK_STEPS = 1024
# Measured on a full-length run, not estimated from a synthetic sample.
BYTES_PER_STEP = 3782


def prune(replay_dir: pathlib.Path, keep_steps: int, margin: float, dry: bool) -> int:
    chunks = sorted(replay_dir.glob("*.npz"))  # names start with a timestamp
    keep = int((keep_steps / CHUNK_STEPS) * margin) + 1
    doomed = chunks[:-keep] if len(chunks) > keep else []
    freed = 0
    for path in doomed:
        try:
            size = path.stat().st_size
            if not dry:
                path.unlink()
            freed += size
        except FileNotFoundError:
            pass  # the trainer may still be rotating files underneath us
    return freed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument(
        "--keep-steps",
        type=float,
        default=10e6,
        help="must be >= the run's replay.size, or the buffer shrinks on resume",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=1.25,
        help="retain this much more than the buffer window, for safety",
    )
    ap.add_argument("--interval", type=float, default=0, help="seconds; 0 = run once")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    replay_dir = args.logdir.expanduser() / "replay"
    if not replay_dir.is_dir():
        raise SystemExit(f"no replay directory at {replay_dir}")

    need_gb = args.keep_steps * args.margin * BYTES_PER_STEP / 1e9
    print(
        f"keeping the newest {args.keep_steps:,.0f} steps x{args.margin} "
        f"(~{need_gb:.0f} GB) in {replay_dir}",
        flush=True,
    )
    while True:
        freed = prune(replay_dir, int(args.keep_steps), args.margin, args.dry_run)
        if freed:
            verb = "would free" if args.dry_run else "freed"
            print(f"  {verb} {freed / 1e9:.1f} GB", flush=True)
        if not args.interval:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
