#!/usr/bin/env python3
"""Give a trained agent an input for the true state, with zero weights on it.

Run 7j (`hwo_`, a diagnostic) fine-tunes run 7f's 3M agent with observation
`true_state` (env.ORACLE_KEYS, 19 numbers) added to the encoder.  The encoder's
vector inputs are concatenated in sorted key order (embodied/jax/nets.DictConcat),
so `true_state` lands after `rpm`, as the last rows of `enc/mlp0/kernel`.  This
appends those rows as zeros, in the parameters and in Adam's two moments, so the
converted agent computes exactly what the original did until training moves them.
Every other array is copied as it is.

    python scripts/oracle_ckpt.py <run 7f 3M snapshot>/ckpt  oracle_7f3M/ckpt
    ./run.sh --finetune oracle_7f3M/ckpt --oracle
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import shutil
import sys

import numpy as np

KERNEL = "enc/mlp0/kernel"
# flight_plan 24 + rates 3 + rpm 4: what the encoder's vector branch reads today
BASE_ROWS = 24 + 3 + 4


def resolve(path: pathlib.Path) -> pathlib.Path:
    """A ckpt directory (with `latest`) or one of its timestamped folders."""
    if (path / "latest").exists():
        path = path / (path / "latest").read_text().strip()
    if not (path / "agent.pkl").exists() or not (path / "done").exists():
        raise SystemExit(f"no complete checkpoint (agent.pkl + done) at {path}")
    return path


def widen(params: dict, extra: int) -> list[str]:
    """Append `extra` zero rows to the encoder's first kernel and its moments."""
    hits = [k for k in params if k == KERNEL or k.endswith("/" + KERNEL)]
    if KERNEL not in hits or len(hits) < 2:
        raise SystemExit(f"expected {KERNEL} and its optimizer moments, found {hits}")
    for k in hits:
        w = np.asarray(params[k])
        if w.shape[0] != BASE_ROWS:
            raise SystemExit(f"{k} has {w.shape[0]} input rows, expected {BASE_ROWS} "
                             "(already converted, or a different observation space?)")
        params[k] = np.concatenate([w, np.zeros((extra, w.shape[1]), w.dtype)], 0)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=pathlib.Path, help="ckpt directory to convert")
    ap.add_argument("dst", type=pathlib.Path, help="new ckpt directory (created)")
    args = ap.parse_args()

    from skydreamer.env import ORACLE_DIM

    src = resolve(args.src)
    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists; pick a new directory")
    data = pickle.loads((src / "agent.pkl").read_bytes())
    hits = widen(data["params"], ORACLE_DIM)

    out = args.dst / src.name
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.name not in ("agent.pkl", "done"):
            shutil.copy2(f, out / f.name)
    (out / "agent.pkl").write_bytes(pickle.dumps(data))
    (out / "done").write_bytes(b"")
    (args.dst / "latest").write_text(src.name)
    print(f"{src} -> {out}")
    for k in hits:
        print(f"  {k}: +{ORACLE_DIM} zero rows -> {data['params'][k].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
