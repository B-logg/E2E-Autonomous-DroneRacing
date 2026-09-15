"""GateNet: the gate segmentation U-Net (paper Appendix A).

PyTorch rather than JAX on purpose.  GateNet is not part of the simulation
training loop -- it runs on *real* camera images at deployment -- and its
deployment path is ONNX -> TensorRT on the Jetson, which starts from PyTorch.

Architecture, transcribed from Appendix A:

    Encoder            Decoder          Outputs
    inc-64/f      ->   up4-64/f    ->   outc4-1
      |                   ^
    down1-128/f   ->   up3-64/f    ->   outc3-1
      |                   ^
    down2-256/f   ->   up2-128/f   ->   outc2-1
      |                   ^
    down3-512/f   ->   up1-256/f   ->   outc1-1
      |                  /
    down4-512/f   ------------------>   outc0-1

Two details that differ from a textbook U-Net and are easy to get wrong:

1. Skips are **added**, not concatenated ("followed by adding the skip
   connections from the corresponding encoder layer").  The transposed
   convolution therefore has to emit the skip's channel count, and the double
   conv afterwards is what moves it to `k`.
2. Five outputs at five resolutions, all supervised (see losses.py).

`f` divides every channel count.  Appendix A: f=2 at 196x196 for the MAVLab
gates, f=4 at 384x384 for the orange gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 Conv-BatchNorm-ReLU layers, the standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Max-pool then a double conv.  The *input* is what gets stored as the
    skip connection, which is why `forward` only returns the downsampled path."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed conv + BN, add the encoder skip, then a double conv to `out_ch`.

    `skip_ch` is both the transposed conv's output width and the skip's width;
    they must match because the merge is an addition."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(skip_ch)
        self.conv = DoubleConv(skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.norm(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:
            # Odd input sizes (196 -> 98 -> 49 -> 24) make the transposed conv
            # come back a pixel short.  Deployment resolution is fixed, so ONNX
            # constant-folding this branch is correct.
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(x + skip)


class OutConv(nn.Module):
    """1x1 conv to one channel.  Returns *logits*; the sigmoid lives in the loss
    (BCEWithLogits) and in `predict`, so training stays numerically stable."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        return self.conv(x)


class GateNet(nn.Module):
    """Returns five logit maps, finest first.

    The finest-first ordering matters: the loss weights are 4, 2, 1, 1, 1 and
    the paper says they "emphasize higher-resolution predictions", so index 0
    has to be the full-resolution head.  See docs/paper_gaps.md A8.
    """

    def __init__(self, f: int = 2, in_ch: int = 3, base: int = 64):
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base // f,  # inc-64/f
            base * 2 // f,  # down1-128/f
            base * 4 // f,  # down2-256/f
            base * 8 // f,  # down3-512/f
            base * 8 // f,  # down4-512/f
        )
        self.f = f
        self.inc = DoubleConv(in_ch, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, base * 4 // f)  # up1-256/f
        self.up2 = Up(base * 4 // f, c3, base * 2 // f)  # up2-128/f
        self.up3 = Up(base * 2 // f, c2, base // f)  # up3-64/f
        self.up4 = Up(base // f, c1, base // f)  # up4-64/f

        self.outc4 = OutConv(base // f)  # full resolution
        self.outc3 = OutConv(base // f)
        self.outc2 = OutConv(base * 2 // f)
        self.outc1 = OutConv(base * 4 // f)
        self.outc0 = OutConv(c5)  # bottleneck

        self.apply(self._init)

    @staticmethod
    def _init(m):
        # Appendix A: "All convolutional layers are initialized using Xavier
        # uniform initialization."
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x) -> list[torch.Tensor]:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.down4(s4)

        d1 = self.up1(b, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)

        # finest -> coarsest
        return [self.outc4(d4), self.outc3(d3), self.outc2(d2), self.outc1(d1), self.outc0(b)]

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5) -> torch.Tensor:
        """Binary mask at full resolution -- what SkyDreamer consumes after a
        resize to 64x64."""
        return (torch.sigmoid(self.forward(x)[0]) > threshold).float()


def build(gate_type: str = "mavlab") -> tuple[GateNet, int]:
    """Appendix A, "Gate-specific implementation": returns (model, resolution)."""
    cfg = {
        "mavlab": (2, 196),  # dark blue/black, thin, logos -> needs more capacity
        "orange": (4, 384),  # high contrast, near-perfect masks from a small net
    }
    if gate_type not in cfg:
        raise ValueError(f"unknown gate type {gate_type!r}; have {sorted(cfg)}")
    f, res = cfg[gate_type]
    return GateNet(f=f), res
