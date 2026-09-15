"""Visual augmentations applied to rendered masks (paper II-F).

Three effects, in the order the paper describes them:
  1. rolling-shutter shear/scale from camera-frame yaw and pitch rate,
  2. random 1-pixel erosion (real masks come out thinner than rendered ones),
  3. StochGAN mask-to-mask translation -- see gan.py, and note that it needs
     real flight footage before it can be trained at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .dynamics import euler_to_quat, quat_to_matrix

# Paper II-F: s in [0, 0.02], sampled per episode.  It stands in for the whole
# rolling-shutter family of effects, not just the geometric part.
ROLLING_SHUTTER_RANGE = (0.0, 0.02)

# Paper II-F: erosion is on 50% of the time and held for ~100 env steps.
EROSION_PROB = 0.5
EROSION_HOLD_STEPS = 100


def camera_rates(omega_b: jax.Array, c_e: jax.Array) -> jax.Array:
    """Body rates expressed in the camera-forward frame -> [p_c, q_c, r_c]."""
    R_bc = quat_to_matrix(euler_to_quat(c_e[..., 0], c_e[..., 1], c_e[..., 2]))
    return jnp.einsum("...ji,...j->...i", R_bc, omega_b)


def rolling_shutter(mask: jax.Array, q_c: jax.Array, r_c: jax.Array, s: jax.Array):
    """Horizontal shear from yaw rate + vertical scale from pitch rate.

    A = [[1, -s r_c,  W/2 s r_c],
         [0,  1+s q_c, -H/2 s q_c]]

    Applied as an inverse map (destination pixel -> source pixel) with nearest
    neighbour sampling, which is exact for a binary mask.  Out-of-bounds reads
    return background."""
    h, w = mask.shape
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    xx, yy = xx.astype(jnp.float32), yy.astype(jnp.float32)

    src_x = xx - s * r_c * yy + (w / 2) * s * r_c
    src_y = (1.0 + s * q_c) * yy - (h / 2) * s * q_c

    xi = jnp.round(src_x).astype(jnp.int32)
    yi = jnp.round(src_y).astype(jnp.int32)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    sampled = mask[jnp.clip(yi, 0, h - 1), jnp.clip(xi, 0, w - 1)]
    return jnp.where(inside, sampled, 0.0)


def erode1(mask: jax.Array) -> jax.Array:
    """1-pixel erosion via 3x3 average pooling followed by thresholding."""
    padded = jnp.pad(mask, 1, constant_values=0.0)
    acc = sum(
        padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
    )
    return (acc / 9.0 > 0.999).astype(mask.dtype)


def apply(
    mask: jax.Array,
    omega_b: jax.Array,
    c_e: jax.Array,
    s: jax.Array,
    erode: jax.Array,
) -> jax.Array:
    """Full geometric augmentation chain for one frame."""
    rates_c = camera_rates(omega_b, c_e)
    out = rolling_shutter(mask, rates_c[1], rates_c[2], s)
    return jnp.where(erode, erode1(out), out)


def resample_erosion(key: jax.Array, current: jax.Array) -> jax.Array:
    """Hold the erosion level for ~EROSION_HOLD_STEPS, then re-draw it."""
    k_hold, k_draw = jax.random.split(key)
    change = jax.random.uniform(k_hold) < (1.0 / EROSION_HOLD_STEPS)
    fresh = jax.random.uniform(k_draw) < EROSION_PROB
    return jnp.where(change, fresh, current)
