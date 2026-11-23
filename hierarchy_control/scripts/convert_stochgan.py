#!/usr/bin/env python3
"""Trained StochGAN generator (PyTorch) -> the .npz the simulator loads (skydreamer/gan.py).

The generator must be the rendered->real direction and follow Appendix B: Zhu et al.'s
ResNet generator with ngf=32, 6 residual blocks, InstanceNorm (affine=False), conv
biases, input = mask + one noise channel (2 x 64 x 64), output 1 channel tanh.  That
is exactly `ResnetGenerator` below (equivalently pytorch-CycleGAN-and-pix2pix's
`ResnetGenerator(input_nc=2, output_nc=1, ngf=32, norm_layer=InstanceNorm2d,
use_dropout=False, n_blocks=6, padding_type='reflect')`).

    python scripts/convert_stochgan.py <G_rendered2real.pth> --out stochgan_G.npz [--key netG_A] [--check]

The checkpoint may be a bare state_dict, or a dict holding one under --key (e.g.
'netG_A', 'G_AB', 'state_dict'); 'module.' prefixes are stripped.  The 18 conv
layers are taken in order of appearance and their shapes checked against the
architecture, so key names do not matter.  --check also runs this file's
PyTorch reference and the JAX forward on random input and compares them
(needs torch and jax in the same environment).

Then:  STOCHGAN_WEIGHTS=stochgan_G.npz ./run.sh --finetune-stochgan <best snapshot>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

EXPECTED = ([(32, None, 7, 7), (64, 32, 3, 3), (128, 64, 3, 3)] + [(128, 128, 3, 3)] * 12
            + [(128, 64, 3, 3), (64, 32, 3, 3), (1, 32, 7, 7)])   # transposed convs: (in, out, k, k)


def reference_generator(in_ch=2):
    """PyTorch module with the exact layer order the converter expects."""
    import torch.nn as nn

    def block(c):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                             nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv_block = block(c)

        def forward(self, x):
            return x + self.conv_block(x)

    layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, 32, 7), nn.InstanceNorm2d(32), nn.ReLU(True),
              nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU(True),
              nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True)]
    layers += [Res(128) for _ in range(6)]
    layers += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(64),
               nn.ReLU(True),
               nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(32),
               nn.ReLU(True),
               nn.ReflectionPad2d(3), nn.Conv2d(32, 1, 7), nn.Tanh()]
    return nn.Sequential(*layers)


def extract(state: dict) -> dict[str, np.ndarray]:
    items = [(k[len("module."):] if k.startswith("module.") else k, v) for k, v in state.items()]
    convs = []
    for i, (k, v) in enumerate(items):
        if k.endswith("weight") and getattr(v, "ndim", 0) == 4:
            bias = dict(items).get(k[:-len("weight")] + "bias")
            if bias is None:
                sys.exit(f"{k}: conv without bias -- not Appendix B's generator (norm must be InstanceNorm)")
            convs.append((k, np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, np.float32),
                          np.asarray(bias.detach().cpu().numpy() if hasattr(bias, "detach") else bias, np.float32)))
    if len(convs) != len(EXPECTED):
        sys.exit(f"found {len(convs)} conv layers, expected {len(EXPECTED)} (ngf 32, 6 blocks): "
                 + ", ".join(f"{k}{tuple(w.shape)}" for k, w, _ in convs))
    out = {}
    for i, ((k, w, b), exp) in enumerate(zip(convs, EXPECTED)):
        ok = all(e is None or e == s for e, s in zip(exp, w.shape))
        if not ok:
            sys.exit(f"conv {i} ({k}) has shape {tuple(w.shape)}, expected {exp}")
        out[f"conv{i}.w"], out[f"conv{i}.b"] = w, b
    in_ch = out["conv0.w"].shape[1]
    if in_ch not in (1, 2):
        sys.exit(f"first conv takes {in_ch} channels; expected 2 (mask + noise) or 1")
    if in_ch == 1:
        print("note: 1 input channel -- a plain CycleGAN generator, no noise channel")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("stochgan_G.npz"))
    ap.add_argument("--key", default=None, help="sub-dict holding the generator's state_dict")
    ap.add_argument("--check", action="store_true", help="compare PyTorch and JAX forward passes")
    args = ap.parse_args()

    import torch
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if args.key:
        obj = obj[args.key]
    elif not any(getattr(v, "ndim", 0) == 4 for v in obj.values()):
        subs = [k for k, v in obj.items() if isinstance(v, dict)]
        sys.exit(f"no conv weights at the top level; pass --key (one of {subs})")
    arrays = extract(obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    n = sum(v.size for v in arrays.values())
    print(f"wrote {args.out}: 18 convs, {n / 1e6:.2f}M parameters, input channels {arrays['conv0.w'].shape[1]}")

    if args.check:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        import jax.numpy as jnp
        from skydreamer import gan
        in_ch = arrays["conv0.w"].shape[1]
        ref = reference_generator(in_ch).eval()
        convs = [m for m in ref.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
        with torch.no_grad():
            for i, m in enumerate(convs):
                m.weight.copy_(torch.from_numpy(arrays[f"conv{i}.w"]))
                m.bias.copy_(torch.from_numpy(arrays[f"conv{i}.b"]))
            x = torch.rand(1, in_ch, 64, 64) * 2 - 1
            y_t = ref(x)[0].numpy()
        y_j = np.asarray(gan.generator(gan.load(args.out), jnp.asarray(x[0].numpy())))
        err = float(np.abs(y_t - y_j).max())
        print(f"PyTorch vs JAX max |diff| = {err:.2e}")
        if err > 1e-3:
            sys.exit("forward passes differ: the generator is not Appendix B's layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
