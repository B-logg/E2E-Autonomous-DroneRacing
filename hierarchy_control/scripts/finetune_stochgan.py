#!/usr/bin/env python3
"""Fine-tune a finished CTBA agent on StochGAN masks (paper II-F).

The curriculum (scripts/train.py) trains on clean rendered masks.  This starts a new
run from the best agent of that curriculum and keeps training it on stage `c4g`:
c4's task (real window, whole airframe, floor and altitude terms) with every rendered
mask sent through the StochGAN generator before rolling shutter and erosion
(ctba.camera.render_augmented), so the world model learns GateNet's real-mask
statistics -- holes, ragged edges, blobs -- before flight.

  stage  steps        task  DreamerV3 settings
  g      0 -> 2M      c4g   phase 3 (actent 1e-5, lr 2e-6, batch_length 256)

Why these settings.  Phase 3 is what the curriculum's last stage used and what every
earlier fine-tune of this project used (scripts/train.py --finetune runs 6-7k, 2-3M
steps each from --run.from_checkpoint).  It moves the agent slowly enough that the
policy is not lost while the encoder adapts.  2M steps is one stage-4 length: stage 4
improved over its first 2M (3 laps 65 -> 79.5% at 16M) and then fell back (20M: 61.5%),
so longer phase-3 training is not better here.  Snapshots every 0.5M (run.sh) let the
best one be picked on `ceg`.

The agent is loaded with --run.from_checkpoint (weights, optimizer state and update
counters) into a fresh logdir whose step counter starts at 0.  The replay starts empty
by default -- every sample is then a StochGAN mask -- as in the earlier fine-tunes;
--replay-steps N copies the newest N steps of the source run's replay (clean masks) in
as a warm buffer instead.  Re-running with the same --logdir resumes (no reload).

    python scripts/finetune_stochgan.py --source <snapshot dir | run dir> --logdir <new run>
        [--weights stochgan_G.npz | $STOCHGAN_WEIGHTS] [--steps 2e6] [--replay-steps 0]
        [--track team44up] [--dry-run] [dreamerv3 flags...]

(./run.sh --finetune-stochgan <source> does this with snapshots, pruning and the final
evaluations on ce and ceg.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import MAIN, PHASE  # noqa: E402

CHUNK_STEPS = 1024
DEMO_MANIFEST = ".demo-chunks"


def checkpoint_folder(src: pathlib.Path) -> pathlib.Path:
    """A complete elements checkpoint folder from a snapshot dir, a run dir, its ckpt/
    dir, or the folder itself."""
    for root in (src, src / "ckpt"):
        if (root / "done").exists():
            return root
        if (root / "latest").exists():
            folder = root / (root / "latest").read_text().strip()
            if (folder / "done").exists():
                return folder
            sys.exit(f"{folder} has no 'done' marker (still being written?)")
    sys.exit(f"no checkpoint under {src} (expected <dir>/ckpt/latest or a folder with 'done')")


def source_run(src: pathlib.Path) -> pathlib.Path:
    """The training run a snapshot came from (<run>/snapshots/step_N) or the run itself."""
    src = src.resolve()
    return src.parent.parent if src.parent.name == "snapshots" else src


def check_weights(path: pathlib.Path) -> None:
    f = np.load(path)
    missing = [k for i in range(18) for k in (f"conv{i}.w", f"conv{i}.b") if k not in f]
    if missing:
        sys.exit(f"{path} is not a converted StochGAN generator (missing {missing[:3]} ...): "
                 "scripts/convert_stochgan.py")


def seed_replay(run: pathlib.Path, dest: pathlib.Path, steps: float) -> int:
    src = run / "replay"
    demos = set((src / DEMO_MANIFEST).read_text().split()) if (src / DEMO_MANIFEST).exists() else set()
    chunks = [c for c in sorted(src.glob("*.npz")) if c.name not in demos]   # names start with a timestamp
    take = chunks[-int(np.ceil(steps / CHUNK_STEPS)):] if steps > 0 else []
    dest.mkdir(parents=True, exist_ok=True)
    for c in take:
        shutil.copy2(c, dest / c.name)
    return len(take)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=pathlib.Path, help="snapshot dir or run dir to start from")
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--weights", type=pathlib.Path, default=None, help="StochGAN .npz (else $STOCHGAN_WEIGHTS)")
    ap.add_argument("--steps", type=float, default=2e6)
    ap.add_argument("--replay-steps", type=float, default=0, help="clean-mask replay copied in as a warm buffer")
    ap.add_argument("--phase", type=int, default=3, choices=sorted(PHASE))
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="size12m")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()
    if not MAIN.exists() and not args.dry_run:
        sys.exit(f"{MAIN} missing: run ./run.sh --setup first")

    logdir = args.logdir.expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    weights = logdir / "stochgan_G.npz"
    given = args.weights or (pathlib.Path(os.environ["STOCHGAN_WEIGHTS"]) if os.environ.get("STOCHGAN_WEIGHTS") else None)
    if given is not None:
        given = given.expanduser()
        check_weights(given)
        if weights.exists() and weights.resolve() != given.resolve() and weights.read_bytes() != given.read_bytes():
            sys.exit(f"{logdir} was started with different StochGAN weights ({weights}); use a new --logdir")
        if not weights.exists():
            shutil.copy2(given, weights)
    if not weights.exists():
        sys.exit("no StochGAN weights: --weights <npz> or STOCHGAN_WEIGHTS=<npz>")

    resuming = (logdir / "ckpt" / "latest").exists()
    ckpt = checkpoint_folder(args.source.expanduser())
    meta_path = logdir / "finetune.json"
    if not resuming:
        n = seed_replay(source_run(args.source.expanduser()), logdir / "replay", args.replay_steps)
        meta = dict(source=str(args.source.resolve()), checkpoint=str(ckpt.resolve()),
                    weights_sha=hashlib.sha256(weights.read_bytes()).hexdigest()[:16],
                    steps=args.steps, phase=args.phase, task=f"ctba_c4g_{args.track}",
                    replay_seed_chunks=n, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"fine-tune from {ckpt}  ({n} clean-mask replay chunks seeded)")
    else:
        print(f"resuming {logdir} (its own checkpoint; the source is not reloaded)")

    cmd = [sys.executable, str(MAIN), "--configs", "ctba", args.size,
           "--logdir", str(logdir), "--seed", str(args.seed),
           "--run.steps", f"{args.steps:.0f}", "--task", f"ctba_c4g_{args.track}",
           *PHASE[args.phase]]
    if not resuming:
        cmd += ["--run.from_checkpoint", str(ckpt)]
    cmd += extra
    env = dict(os.environ, STOCHGAN_WEIGHTS=str(weights.resolve()))
    print("STOCHGAN_WEIGHTS=" + env["STOCHGAN_WEIGHTS"] + " " + " ".join(cmd), flush=True)
    (logdir / "stage").write_text("g\n")
    if args.dry_run:
        return 0
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
