#!/usr/bin/env python3
"""Train GateNet (paper Appendix A; hyperparameters from MonoRace arXiv:2601.15222).

  AdamW, 100 epochs, base lr 1e-3, cosine decay.

Data layout -- see docs/datasets.md:

    data/gatenet/<gate_type>/
      real/images/*.png     RGB, already remapped to the nominal K
      real/masks/*.png      {0,255}, matching basenames
      backgrounds/*.jpg     optional, for synthetic compositing
      cutouts/*.png         optional, RGBA gate cutouts

Examples
  python scripts/train_gatenet.py --data data/gatenet/orange --gate orange
  python scripts/train_gatenet.py --data data/gatenet/mavlab --gate mavlab --synthetic 8500
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from skydreamer.gatenet.data import (  # noqa: E402
    MixedGateDataset,
    RealGateDataset,
    SyntheticGateCompositor,
    SyntheticGateDataset,
)
from skydreamer.gatenet.losses import iou, multiscale_loss  # noqa: E402
from skydreamer.gatenet.model import build  # noqa: E402

EPOCHS = 100
BASE_LR = 1e-3


def make_datasets(root: pathlib.Path, gate: str, size: int, n_synth: int, val_frac: float, seed: int):
    real = RealGateDataset(root / "real", size=size, train=True, seed=seed)
    n_val = max(1, int(len(real) * val_frac))
    train_real, val = random_split(
        real, [len(real) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
    )
    # Validation must not be augmented: augmented val scores drift with the
    # augmentation strength and stop being comparable across runs.
    val.dataset = RealGateDataset(root / "real", size=size, train=False, seed=seed)

    train = train_real
    if n_synth:
        comp = SyntheticGateCompositor(root / "cutouts", root / "backgrounds", size, seed)
        train = MixedGateDataset(train_real, SyntheticGateDataset(comp, n_synth))
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--gate", default="mavlab", choices=("mavlab", "orange"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("checkpoints/gatenet"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=BASE_LR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--synthetic", type=int, default=0, help="synthetic samples per epoch")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, size = build(args.gate)
    model.to(args.device)

    train_ds, val_ds = make_datasets(
        args.data, args.gate, size, args.synthetic, args.val_frac, args.seed
    )
    print(f"gate={args.gate} res={size} train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * e / args.epochs))
    )

    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            loss, _ = multiscale_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().item()
        sched.step()

        model.eval()
        v_loss, v_iou, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                outs = model(x)
                v_loss += float(multiscale_loss(outs, y)[0])
                v_iou += float(iou(outs[0], y))
                n += 1
        row = dict(
            epoch=epoch,
            train_loss=total / max(1, len(train_dl)),
            val_loss=v_loss / max(1, n),
            val_iou=v_iou / max(1, n),
            lr=sched.get_last_lr()[0],
        )
        history.append(row)
        print(
            f"epoch {epoch:3d}  train {row['train_loss']:.4f}  "
            f"val {row['val_loss']:.4f}  IoU {row['val_iou']:.4f}",
            flush=True,
        )

        # Select on IoU, not loss: Dice/BCE keep improving well after the mask
        # has stopped getting better, and IoU is what the policy actually feels.
        if row["val_iou"] > best:
            best = row["val_iou"]
            torch.save(
                {"model": model.state_dict(), "gate": args.gate, "size": size, "val_iou": best},
                args.out / f"gatenet_{args.gate}.pt",
            )
        (args.out / f"history_{args.gate}.json").write_text(json.dumps(history, indent=2))

    print(f"best val IoU {best:.4f} -> {args.out / f'gatenet_{args.gate}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
