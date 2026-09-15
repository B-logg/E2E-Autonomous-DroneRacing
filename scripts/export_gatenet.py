#!/usr/bin/env python3
"""Export a trained GateNet to ONNX for TensorRT on the Jetson Orin NX.

The paper budgets 3 ms for segmentation inside an 11.1 ms control period, so
only the full-resolution head is exported -- the four auxiliary heads exist to
shape training and are dead weight at deployment.

  python scripts/export_gatenet.py --ckpt checkpoints/gatenet/gatenet_orange.pt

Then on the Jetson:
  trtexec --onnx=gatenet_orange.onnx --saveEngine=gatenet_orange.engine --fp16
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from skydreamer.gatenet.model import build  # noqa: E402


class DeployGateNet(nn.Module):
    """Full-resolution logits only, with the sigmoid folded in."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, size = build(ckpt["gate"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt['gate']} @ {size}px, val IoU {ckpt.get('val_iou', float('nan')):.4f}")

    deploy = DeployGateNet(model).eval()
    dummy = torch.zeros(1, 3, size, size)
    out = args.out or args.ckpt.with_suffix(".onnx")
    torch.onnx.export(
        deploy,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["mask"],
        opset_version=args.opset,
        dynamo=False,
    )

    # A silent numerical drift here surfaces later as a policy that flies fine
    # in simulation and not at all on the drone, so check it now.
    try:
        import numpy as np
        import onnxruntime as ort

        probe = torch.rand(1, 3, size, size)
        ref = deploy(probe).detach().numpy()
        got = ort.InferenceSession(str(out)).run(None, {"image": probe.numpy()})[0]
        print(f"onnxruntime max abs diff: {np.abs(ref - got).max():.2e}")
    except ImportError:
        print("onnxruntime not installed; skipping numerical check")

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
