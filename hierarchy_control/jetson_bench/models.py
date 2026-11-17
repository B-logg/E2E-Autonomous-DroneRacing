"""The two networks that run on the Jetson every frame, in PyTorch, with random
weights: latency depends on shapes and operators, not on weight values.

PolicyStep  one DreamerV3 policy step at the size we train (size12m), exactly the
            work agent.policy() does per control step: encoder (64x64 mask CNN +
            vector MLP) -> RSSM core (block GRU) -> posterior -> actor.  Layer
            sizes from dreamerv3/configs.yaml (size12m) and rssm.py.
GateNet     skydreamer/gatenet/model.py, copied so the Jetson needs nothing else.

Self-contained on purpose: torch + numpy only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DreamerV3 size12m (configs.yaml) and our observation space (ctba.env) -------------
DETER, HIDDEN, STOCH, CLASSES, BLOCKS = 2048, 256, 32, 16, 8
DEPTH, MULTS, KERNEL, UNITS, ENC_LAYERS, POL_LAYERS = 16, (2, 3, 4, 4), 5, 256, 3, 3
IMG = 64
VEC = 3 + 3 + 2 + 24          # rates, acc, FC attitude, flight plan (3 gates x 4 x 2)
ACT = 4


class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))
        self.shift = nn.Parameter(torch.zeros(n))
        self.eps = eps

    def forward(self, x, dim=-1):
        x = x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + self.eps)
        if dim == 1:   # channels-first conv features
            return x * self.scale[None, :, None, None] + self.shift[None, :, None, None]
        return x * self.scale + self.shift


class Dense(nn.Module):
    """Linear -> RMS norm -> SiLU (DreamerV3's standard layer)."""

    def __init__(self, i, o):
        super().__init__()
        self.lin, self.norm = nn.Linear(i, o), RMSNorm(o)

    def forward(self, x):
        return F.silu(self.norm(self.lin(x)))


class BlockLinear(nn.Module):
    """nn.BlockLinear: `blocks` independent dense maps on equal slices."""

    def __init__(self, i, o, blocks):
        super().__init__()
        self.blocks = blocks
        self.w = nn.Parameter(torch.randn(blocks, i // blocks, o // blocks) * 0.02)
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.blocks, -1)
        x = torch.einsum("bki,kio->bko", x, self.w)
        return x.reshape(b, -1) + self.b


class PolicyStep(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: vector MLP + mask CNN (stride 1 conv, 2x2 max pool, norm, SiLU)
        self.vec = nn.Sequential(Dense(VEC, UNITS), *[Dense(UNITS, UNITS) for _ in range(ENC_LAYERS - 1)])
        chans = [1] + [DEPTH * m for m in MULTS]
        self.convs = nn.ModuleList(nn.Conv2d(chans[i], chans[i + 1], KERNEL, padding=KERNEL // 2)
                                   for i in range(len(MULTS)))
        self.cnorms = nn.ModuleList(RMSNorm(c) for c in chans[1:])
        tokens = UNITS + chans[-1] * (IMG // 2 ** len(MULTS)) ** 2
        # RSSM core (rssm.py _core), one dynamics layer
        self.din0, self.din1, self.din2 = Dense(DETER, HIDDEN), Dense(STOCH * CLASSES, HIDDEN), Dense(ACT, HIDDEN)
        g_in = DETER // BLOCKS + 3 * HIDDEN
        self.hid = BlockLinear(BLOCKS * g_in, DETER, BLOCKS)
        self.hidnorm = RMSNorm(DETER)
        self.gru = BlockLinear(DETER, 3 * DETER, BLOCKS)
        # posterior (rssm.py _observe), one obs layer
        self.obs = Dense(DETER + tokens, HIDDEN)
        self.obslogit = nn.Linear(HIDDEN, STOCH * CLASSES)
        # actor (policy head): mean and std of 4 actions
        self.pol = nn.Sequential(Dense(DETER + STOCH * CLASSES, UNITS),
                                 *[Dense(UNITS, UNITS) for _ in range(POL_LAYERS - 1)])
        self.polout = nn.Linear(UNITS, 2 * ACT)

    def forward(self, mask, vec, deter, stoch, prev_action):
        b = mask.shape[0]
        v = self.vec(torch.sign(vec) * torch.log1p(vec.abs()))           # symlog
        x = mask / 255.0 - 0.5
        for conv, norm in zip(self.convs, self.cnorms):
            x = F.max_pool2d(conv(x), 2)
            x = F.silu(norm(x, dim=1))
        tokens = torch.cat([v, x.reshape(b, -1)], -1)

        a = prev_action / torch.clamp(prev_action.abs(), min=1.0)
        h = torch.cat([self.din0(deter), self.din1(stoch), self.din2(a)], -1)
        g = BLOCKS
        h = torch.cat([deter.reshape(b, g, -1), h[:, None, :].expand(b, g, h.shape[-1])], -1).reshape(b, -1)
        h = F.silu(self.hidnorm(self.hid(h)))
        r, c, u = torch.split(self.gru(h).reshape(b, g, -1), DETER // g, -1)
        r, c, u = r.reshape(b, -1), c.reshape(b, -1), u.reshape(b, -1)
        cand = torch.tanh(torch.sigmoid(r) * c)
        upd = torch.sigmoid(u - 1)
        deter = upd * cand + (1 - upd) * deter

        logit = self.obslogit(self.obs(torch.cat([deter, tokens], -1))).reshape(b, STOCH, CLASSES)
        stoch = F.one_hot(logit.argmax(-1), CLASSES).to(deter.dtype).reshape(b, -1)   # mode: deterministic eval
        out = self.polout(self.pol(torch.cat([deter, stoch], -1)))
        action = torch.tanh(out[:, :ACT])
        return action, deter, stoch

    @staticmethod
    def example_inputs(device="cpu", dtype=torch.float32, batch=1):
        return (torch.randint(0, 2, (batch, 1, IMG, IMG), device=device).to(dtype) * 255,
                torch.randn(batch, VEC, device=device, dtype=dtype),
                torch.randn(batch, DETER, device=device, dtype=dtype),
                torch.zeros(batch, STOCH * CLASSES, device=device, dtype=dtype),
                torch.zeros(batch, ACT, device=device, dtype=dtype))


# --- GateNet (skydreamer/gatenet/model.py) --------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = DoubleConv(i, o)

    def forward(self, x):
        return self.conv(F.max_pool2d(x, 2))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, skip, 2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(skip)
        self.conv = DoubleConv(skip, o)

    def forward(self, x, skip):
        x = self.norm(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:      # odd sizes (196 -> 98 -> 49 -> 24) come back short
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(x + skip)


class GateNet(nn.Module):
    """Deployment form: only the full-resolution head (the other four heads exist for
    the training loss), sigmoid included."""

    def __init__(self, f=2, base=64):
        super().__init__()
        c1, c2, c3, c4, c5 = base // f, base * 2 // f, base * 4 // f, base * 8 // f, base * 8 // f
        self.inc, self.d1, self.d2, self.d3, self.d4 = (DoubleConv(3, c1), Down(c1, c2), Down(c2, c3),
                                                         Down(c3, c4), Down(c4, c5))
        self.u1, self.u2 = Up(c5, c4, base * 4 // f), Up(base * 4 // f, c3, base * 2 // f)
        self.u3, self.u4 = Up(base * 2 // f, c2, base // f), Up(base // f, c1, base // f)
        self.out = nn.Conv2d(base // f, 1, 1)

    def forward(self, x):
        s1 = self.inc(x); s2 = self.d1(s1); s3 = self.d2(s2); s4 = self.d3(s3); b = self.d4(s4)
        d = self.u4(self.u3(self.u2(self.u1(b, s4), s3), s2), s1)
        return torch.sigmoid(self.out(d))


from configs import GATENET_CONFIGS, gatenet_name  # noqa: E402,F401
