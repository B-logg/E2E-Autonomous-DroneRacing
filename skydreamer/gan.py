"""StochGAN generator (paper II-F, Appendix B) as a pure-JAX forward pass.

The generator turns a clean rendered 64x64 gate mask into one that looks like
GateNet's output on real footage (holes, ragged edges, spurious blobs).  It is
trained elsewhere (PyTorch, unpaired CycleGAN-style); this module only runs the
trained rendered->real generator inside the simulator, so it can sit in the
jitted environment step between rendering and the rolling-shutter / erosion
augmentations (augment.apply).

Architecture (Appendix B, Zhu et al.'s ResNet generator, ngf = 32, 6 blocks):

    input   mask in [-1, 1]  +  noise channel ~ U[-1, 1]        (2 x 64 x 64)
    c7s1-32   reflect-pad 3, conv 7x7, InstanceNorm, ReLU
    d64       conv 3x3 stride 2 pad 1 (zeros), InstanceNorm, ReLU
    d128      conv 3x3 stride 2 pad 1 (zeros), InstanceNorm, ReLU
    6 x R128  [reflect-pad 1, conv 3x3, IN, ReLU, reflect-pad 1, conv 3x3, IN] + skip
    u64       transposed conv 3x3 stride 2 pad 1 output_pad 1, InstanceNorm, ReLU
    u32       transposed conv 3x3 stride 2 pad 1 output_pad 1, InstanceNorm, ReLU
    c7s1-1    reflect-pad 3, conv 7x7, tanh

InstanceNorm is non-affine (eps 1e-5) and every conv has a bias, as in the
reference CycleGAN code (norm='instance' -> affine=False, use_bias=True).

Weights: an .npz written by hierarchy_control/scripts/convert_stochgan.py with
`conv{i}.w` / `conv{i}.b` for the 18 convolutions in the order above (PyTorch
layouts: Conv2d (out, in, kh, kw), ConvTranspose2d (in, out, kh, kw)).  The
simulator finds it through $STOCHGAN_WEIGHTS.
"""

from __future__ import annotations

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

ENV_VAR = "STOCHGAN_WEIGHTS"
N_BLOCKS = 6
N_CONV = 3 + 2 * N_BLOCKS + 3          # 18
UPSAMPLE = (3 + 2 * N_BLOCKS, 3 + 2 * N_BLOCKS + 1)   # conv indices 15, 16 are transposed

_CACHE: dict[str, dict] = {}


def load(path: str | os.PathLike | None = None) -> dict:
    """{'w': [18 arrays], 'b': [18 arrays]} as jnp constants; cached per path."""
    path = str(path or os.environ.get(ENV_VAR, ""))
    if not path:
        raise RuntimeError(f"StochGAN weights not set: export {ENV_VAR}=<generator .npz> "
                           "(hierarchy_control/scripts/convert_stochgan.py)")
    if path not in _CACHE:
        f = np.load(pathlib.Path(path).expanduser())
        w = [jnp.asarray(f[f"conv{i}.w"], jnp.float32) for i in range(N_CONV)]
        b = [jnp.asarray(f[f"conv{i}.b"], jnp.float32) for i in range(N_CONV)]
        _CACHE[path] = {"w": w, "b": b, "in_ch": int(w[0].shape[1])}
    return _CACHE[path]


def _reflect(x, p):
    return jnp.pad(x, ((0, 0), (p, p), (p, p)), mode="reflect")


def _conv(x, w, b, stride=1, pad=0):
    """x (C, H, W); w (O, I, kh, kw) PyTorch layout; zero padding `pad`."""
    y = jax.lax.conv_general_dilated(
        x[None], w, window_strides=(stride, stride), padding=((pad, pad), (pad, pad)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))[0]
    return y + b[:, None, None]


def _conv_t(x, w, b):
    """PyTorch ConvTranspose2d(k=3, stride=2, padding=1, output_padding=1); w (I, O, kh, kw)."""
    k = w.shape[-1]
    w_f = jnp.flip(w, (-2, -1)).transpose(1, 0, 2, 3)          # -> (O, I, kh, kw), flipped
    lo, hi = k - 1 - 1, k - 1 - 1 + 1                           # k-1-padding, + output_padding
    y = jax.lax.conv_general_dilated(
        x[None], w_f, window_strides=(1, 1), padding=((lo, hi), (lo, hi)), lhs_dilation=(2, 2),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))[0]
    return y + b[:, None, None]


def _inorm(x, eps=1e-5):
    m = x.mean(axis=(1, 2), keepdims=True)
    v = ((x - m) ** 2).mean(axis=(1, 2), keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def generator(params: dict, x: jax.Array) -> jax.Array:
    """x (C_in, 64, 64) in [-1, 1] -> (1, 64, 64) in [-1, 1]."""
    w, b = params["w"], params["b"]
    relu = jax.nn.relu
    h = relu(_inorm(_conv(_reflect(x, 3), w[0], b[0])))
    h = relu(_inorm(_conv(h, w[1], b[1], stride=2, pad=1)))
    h = relu(_inorm(_conv(h, w[2], b[2], stride=2, pad=1)))
    for i in range(N_BLOCKS):
        j = 3 + 2 * i
        r = relu(_inorm(_conv(_reflect(h, 1), w[j], b[j])))
        r = _inorm(_conv(_reflect(r, 1), w[j + 1], b[j + 1]))
        h = h + r
    h = relu(_inorm(_conv_t(h, w[UPSAMPLE[0]], b[UPSAMPLE[0]])))
    h = relu(_inorm(_conv_t(h, w[UPSAMPLE[1]], b[UPSAMPLE[1]])))
    return jnp.tanh(_conv(_reflect(h, 3), w[-1], b[-1]))


def translate(params: dict, mask: jax.Array, key: jax.Array, binary: bool = True) -> jax.Array:
    """Rendered mask (H, W) in {0, 1} -> real-looking mask (H, W) in {0, 1} (or [0, 1])."""
    x = (2.0 * mask.astype(jnp.float32) - 1.0)[None]
    if params["in_ch"] == 2:
        x = jnp.concatenate([x, jax.random.uniform(key, x.shape, minval=-1.0, maxval=1.0)], 0)
    y = generator(params, x)[0]
    return (y > 0.0).astype(jnp.float32) if binary else 0.5 * (y + 1.0)
