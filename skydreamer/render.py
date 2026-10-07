"""Binary gate segmentation mask rendering (paper II-E).

The paper renders masks with PyTorch3D.  We don't: gates are flat rectangular
annuli, so the mask can be computed exactly by casting one ray per pixel and
intersecting it with each gate plane.  That is a handful of array ops, it is
exact rather than sampled, and -- importantly -- it keeps the whole environment
in JAX instead of bouncing every frame across a JAX/PyTorch boundary, which
would otherwise dominate rollout time.

Camera convention: the policy always sees images mapped to the paper's nominal
pinhole intrinsics
    K = [[25/64 W, 0, W/2], [0, 25/64 H, H/2], [0, 0, 1]]
so a 64x64 image has focal length 25 (~104 deg diagonal FOV).  Real images are
undistorted and cropped onto this same K using the (calibrated, constant)
camera intrinsics, which is why the policy is intrinsics-invariant.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .dynamics import euler_to_quat, quat_to_matrix
from .track import Track, gate_rotation

FOCAL_RATIO = 25.0 / 64.0


def intrinsics(h: int, w: int) -> tuple[float, float, float, float]:
    return FOCAL_RATIO * w, FOCAL_RATIO * h, 0.5 * w, 0.5 * h


def fov_valid(h: int, w: int, fov_deg: tuple[float, float]) -> jax.Array:
    """(H, W) float32, 1 where a camera of this (horizontal, vertical) field of view
    has data once its image is remapped onto the nominal intrinsics, 0 outside.

    The nominal K is a ~104.4 degree square.  A camera whose field of view is wider
    than that in both directions fills it; a narrower one leaves blank borders,
    which is what the policy would be handed on the real drone.  Ours is 114.6 x
    92.15 degrees: all columns are valid, the top and bottom ~6 rows are not."""
    import numpy as np

    fx, fy, cx, cy = intrinsics(h, w)
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    ok = (np.abs((u - cx) / fx) <= np.tan(np.radians(fov_deg[0]) / 2)) & (
        np.abs((v - cy) / fy) <= np.tan(np.radians(fov_deg[1]) / 2))
    return jnp.asarray(ok, jnp.float32)


@functools.partial(jax.jit, static_argnames=("h", "w"))
def pixel_rays(h: int, w: int) -> jax.Array:
    """(H, W, 3) unnormalized ray directions in the camera-forward frame
    (x forward, y right, z down) for the nominal intrinsics."""
    fx, fy, cx, cy = intrinsics(h, w)
    v, u = jnp.meshgrid(jnp.arange(h) + 0.5, jnp.arange(w) + 0.5, indexing="ij")
    x_cv = (u - cx) / fx  # right
    y_cv = (v - cy) / fy  # down
    z_cv = jnp.ones_like(x_cv)  # forward
    return jnp.stack([z_cv, x_cv, y_cv], axis=-1)


def camera_rotation(q: jax.Array, c_e: jax.Array) -> jax.Array:
    """Camera-forward -> world rotation.

    `c_e = [phi_c, theta_c, psi_c]` is the body->camera Euler rotation in
    psi -> theta -> phi order (paper II-B).  Aerospace NED sign convention, so
    the paper's theta_c in [45, 55] deg tilts the camera *up* by ~50 deg, as
    drawn in Figure 2."""
    R_wb = quat_to_matrix(q)
    R_bc = quat_to_matrix(euler_to_quat(c_e[..., 0], c_e[..., 1], c_e[..., 2]))
    return R_wb @ R_bc


def render_mask(
    p_cam: jax.Array,
    q: jax.Array,
    c_e: jax.Array,
    track: Track,
    h: int = 64,
    w: int = 64,
) -> jax.Array:
    """(H, W) float32 mask in {0, 1}: 1 where a gate frame is visible.

    Depth-correct: when gates overlap in image space the nearest one wins, and
    a nearer gate's *hole* correctly shows whatever is behind it."""
    d_world = jnp.einsum("ij,hwj->hwi", camera_rotation(q, c_e), pixel_rays(h, w))

    yaw = track.yaw
    normals = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], -1)  # (N,3)

    denom = jnp.einsum("nj,hwj->hwn", normals, d_world)
    safe = jnp.where(jnp.abs(denom) < 1e-8, 1e-8, denom)
    t = jnp.einsum("nj,nj->n", normals, track.pos - p_cam)[None, None, :] / safe

    hit = p_cam + t[..., None] * d_world[:, :, None, :]  # (H, W, N, 3)
    local = jnp.einsum("nij,hwnj->hwni", gate_rotation(yaw), hit - track.pos)

    off = jnp.maximum(jnp.abs(local[..., 1]), jnp.abs(local[..., 2]))
    on_frame = (off <= track.outer / 2) & (off >= track.inner / 2)
    valid = on_frame & (t > 1e-3) & (jnp.abs(denom) >= 1e-8) & track.visible

    # Nearest hit wins; if nothing is hit the pixel is background.
    depth = jnp.where(valid, t, jnp.inf)
    return (jnp.min(depth, axis=-1) < jnp.inf).astype(jnp.float32)


render_batch = jax.jit(
    jax.vmap(render_mask, in_axes=(0, 0, 0, None, None, None)),
    static_argnames=("h", "w"),
)
