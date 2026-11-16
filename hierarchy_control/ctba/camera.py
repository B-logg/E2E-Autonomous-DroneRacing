"""Our camera: Raspberry Pi Camera Module v2 (Sony IMX219, standard 3.04 mm lens) on the
Jetson's CSI port, tilted 40 deg up, fixed.

SkyDreamer renders every mask on the paper's nominal pinhole (focal 25/64 of the
width, a ~104 deg square) and remaps real images onto it.  Our lens sees far less
than that, so here the policy's 64x64 image spans *this camera's* field instead:

    fx = (W/2) / tan(HFOV/2),   fy = (H/2) / tan(VFOV/2),   cx = W/2,  cy = H/2

(anisotropic when the field is not square).  Every pixel then carries data, and on
the drone the undistorted camera image is resized onto the same K (`policy_K`).

IMX219 modes on the Jetson (fields from the 3.04 mm focal length, 1.12 um pixels):
  1640x1232 @ 30 fps  full sensor, 2x2 binned        62.2 x 48.8 deg
  1280x720  @ 60 fps  2560x1440 centre crop, binned   50.5 x 29.7 deg
At 40 deg tilt the cropped mode sees the gate being flown 8% of the time on the demo
flights, the full-field mode 31% (scripts/camera_visibility.py), so the full-field
mode is the default.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from skydreamer import augment
from skydreamer.dynamics import euler_to_quat, quat_to_matrix
from skydreamer.render import camera_rotation
from skydreamer.track import Track, gate_rotation

DEG = np.pi / 180.0
SD_FOCAL_RATIO = 25.0 / 64.0     # skydreamer's nominal focal / width (rolling-shutter scale reference)


class CameraSpec(NamedTuple):
    name: str
    hfov_deg: float
    vfov_deg: float
    fps: tuple                 # (lo, hi) frames/s the policy receives (camera or GateNet, whichever is slower)
    tilt_deg: float            # mounted tilt, up from the body x axis
    tilt_range: tuple          # training randomization (deg)
    roll_yaw_range: float = 5.0   # +- deg, mounting roll / yaw (SkyDreamer Table III)


PICAM_V2_FULL = CameraSpec("picam_v2_1640x1232", 62.2, 48.8, (25.0, 30.0), 40.0, (35.0, 45.0))
PICAM_V2_720P = CameraSpec("picam_v2_1280x720", 50.5, 29.7, (50.0, 60.0), 40.0, (35.0, 45.0))
CAMERAS = {c.name: c for c in (PICAM_V2_FULL, PICAM_V2_720P)}
DEFAULT = PICAM_V2_FULL


def intrinsics(spec: CameraSpec, h: int, w: int):
    fx = 0.5 * w / np.tan(0.5 * spec.hfov_deg * DEG)
    fy = 0.5 * h / np.tan(0.5 * spec.vfov_deg * DEG)
    return float(fx), float(fy), 0.5 * w, 0.5 * h


def policy_K(spec: CameraSpec, h: int = 64, w: int = 64) -> np.ndarray:
    """The pinhole the deployed pipeline must resize undistorted frames onto."""
    fx, fy, cx, cy = intrinsics(spec, h, w)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def pixel_rays(spec: CameraSpec, h: int, w: int) -> jax.Array:
    """(H, W, 3) ray directions, camera-forward frame (x forward, y right, z down)."""
    fx, fy, cx, cy = intrinsics(spec, h, w)
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    return jnp.asarray(np.stack([np.ones_like(u), (u - cx) / fx, (v - cy) / fy], -1), jnp.float32)


def sample_extrinsics(key: jax.Array, spec: CameraSpec) -> jax.Array:
    """c_e = [roll, pitch (tilt up, NED sign as skydreamer), yaw], rad."""
    kr, kp, ky = jax.random.split(key, 3)
    ry = spec.roll_yaw_range * DEG
    return jnp.stack([
        jax.random.uniform(kr, minval=-ry, maxval=ry),
        jax.random.uniform(kp, minval=spec.tilt_range[0] * DEG, maxval=spec.tilt_range[1] * DEG),
        jax.random.uniform(ky, minval=-ry, maxval=ry),
    ])


def render_mask(p_cam, q, c_e, track: Track, spec: CameraSpec, h: int = 64, w: int = 64,
                shrink_px=0.0) -> jax.Array:
    """skydreamer.render.render_mask with this camera's intrinsics (same ray casting:
    exact, depth-correct, nearest gate wins, gate holes show what is behind)."""
    d_world = jnp.einsum("ij,hwj->hwi", camera_rotation(q, c_e), pixel_rays(spec, h, w))
    yaw = track.yaw
    normals = jnp.stack([jnp.cos(yaw), jnp.sin(yaw), jnp.zeros_like(yaw)], -1)
    denom = jnp.einsum("nj,hwj->hwn", normals, d_world)
    safe = jnp.where(jnp.abs(denom) < 1e-8, 1e-8, denom)
    t = jnp.einsum("nj,nj->n", normals, track.pos - p_cam)[None, None, :] / safe
    hit = p_cam + t[..., None] * d_world[:, :, None, :]
    local = jnp.einsum("nij,hwnj->hwni", gate_rotation(yaw), hit - track.pos)
    off = jnp.maximum(jnp.abs(local[..., 1]), jnp.abs(local[..., 2]))
    fx = intrinsics(spec, h, w)[0]
    thin = shrink_px * jnp.maximum(t, 0.0) / fx
    on_frame = (off <= track.outer / 2 - thin) & (off >= track.inner / 2 + thin)
    valid = on_frame & (t > 1e-3) & (jnp.abs(denom) >= 1e-8) & track.visible
    depth = jnp.where(valid, t, jnp.inf)
    return (jnp.min(depth, axis=-1) < jnp.inf).astype(jnp.float32)


def render_augmented(state, cfg, spec: CameraSpec) -> jax.Array:
    """skydreamer.env._render_augmented for this camera: camera ahead of the CoM, mask
    erosion, rolling shutter.  The rolling-shutter strength is per radian of camera
    rotation, so it scales with the focal length in pixels."""
    p_cam = state.s.p
    if cfg.cam_offset:
        p_cam = p_cam + quat_to_matrix(state.s.q)[..., :, 0] * cfg.cam_offset
    shrink = 0.0 if cfg.erode_px is None else jnp.where(state.erode, cfg.erode_px, 0.0)
    n = cfg.image_size
    mask = render_mask(p_cam, state.s.q, state.c_e, cfg.track, spec, n, n, shrink)
    erode = state.erode if cfg.erode_px is None else jnp.array(False)
    fx = intrinsics(spec, n, n)[0]
    return augment.apply(mask, state.s.omega_b, state.c_e, state.rs_s * fx / (SD_FOCAL_RATIO * n), erode)
