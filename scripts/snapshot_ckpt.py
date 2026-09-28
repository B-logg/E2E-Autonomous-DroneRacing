#!/usr/bin/env python3
"""Copy a training run's newest checkpoint somewhere safe to evaluate from.

Evaluating a run while it is still training needs two things the training
directory cannot give you directly.

**A checkpoint that will not vanish.** `elements.Checkpoint` keeps exactly one
(`keep=1`): the moment a new one is written, the old folder is deleted. Loading
straight out of `ckpt/` therefore races the trainer every 15 minutes. Copying
first turns a corrupted read into a failed copy, which is retryable.

**A config that will not fight the trainer for the GPU.** The run's own
`config.yaml` is reused verbatim except for the memory knobs, so the evaluation
shares the card instead of preallocating against the process that is doing the
real work.

The snapshot is named by the checkpoint's own step -- `step.pkl` carries it --
so repeated runs accumulate a history rather than overwriting each other.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import shutil
import sys
import time


def read_step(ckpt_dir: pathlib.Path) -> int:
    return int(pickle.loads((ckpt_dir / "step.pkl").read_bytes()))


def latest_complete(ckpt_root: pathlib.Path) -> pathlib.Path:
    """The folder `latest` names, which is only written after `save()` returns,
    and which must carry the `done` marker `elements.checkpoint.exists` looks
    for."""
    marker = ckpt_root / "latest"
    if not marker.exists():
        raise SystemExit(f"no checkpoint yet under {ckpt_root}")
    folder = ckpt_root / marker.read_text().strip()
    if not (folder / "done").exists():
        raise SystemExit(f"{folder} has no 'done' marker -- still being written?")
    return folder


def snapshot(logdir: pathlib.Path, out_root: pathlib.Path, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        src = latest_complete(logdir / "ckpt")
        step = read_step(src)
        dest = out_root / f"step_{step:012d}"
        ckpt_dest = dest / "ckpt" / src.name
        if (dest / "ckpt" / "latest").exists() and (ckpt_dest / "done").exists():
            print(f"already snapshotted step {step:,}")
            return dest, step
        try:
            ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, ckpt_dest, dirs_exist_ok=True)
            # Re-read after the copy: if the trainer rotated the checkpoint out
            # from under us the copy is a mixture of two of them, and the
            # cheapest way to notice is that the source is gone.
            if not (src / "done").exists():
                raise FileNotFoundError(src)
            read_step(ckpt_dest)
        except (FileNotFoundError, OSError, EOFError, pickle.UnpicklingError) as e:
            shutil.rmtree(dest, ignore_errors=True)
            if attempt == attempts:
                raise SystemExit(f"checkpoint kept moving under us: {e}")
            print(f"  checkpoint rotated mid-copy, retrying ({attempt}/{attempts})")
            time.sleep(5)
            continue
        (dest / "ckpt" / "latest").write_text(src.name)
        _write_config(logdir, dest)
        print(f"snapshotted step {step:,} -> {dest}")
        return dest, step
    raise SystemExit("unreachable")


def _write_config(logdir: pathlib.Path, dest: pathlib.Path) -> None:
    """The run's config, with the GPU memory knobs relaxed.

    `prealloc: false` is what run.sh already trains with, but assert it rather
    than assume: an evaluation that preallocates would take the card out from
    under a 60-hour run."""
    text = (logdir / "config.yaml").read_text()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("prealloc:"):
            line = line.split("prealloc:")[0] + "prealloc: false"
        out.append(line)
    (dest / "config.yaml").write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--print-dir", action="store_true", help="print only the path")
    args = ap.parse_args()

    dest, step = snapshot(args.logdir.expanduser(), args.out.expanduser())
    if args.print_dir:
        print(dest)
    else:
        print(f"step={step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
