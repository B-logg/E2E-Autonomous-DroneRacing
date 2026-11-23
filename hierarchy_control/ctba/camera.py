"""Our camera: Raspberry Pi Camera Module v2 (Sony IMX219, standard 3.04 mm lens) on the
Jetson's CSI port, mounted 30 deg up (CAD drone_design_tilt30.step; was 40 deg).

SkyDreamer renders every mask on the paper's nominal pinhole (focal 25/64 of the
width, a ~104 deg square) and remaps real images onto it.  Our lens sees far less
than that, so here the policy's 64x64 image spans *this camera's* field instead:

    fx = (W/2) / tan(HFOV/2),   fy = (H/2) / tan(VFOV/2),   cx = W/2,  cy = H/2

(anisotropic when the field is not square).  Every pixel then carries data, and on
the drone the undistorted camera image is resized onto the same K (`policy_K`).

IMX219 modes on the Jetson (datasheet fields from the 3.04 mm focal length, 1.12 um pixels):
  1640x1232 @ 30 fps  full sensor, 2x2 binned        62.2 x 48.8 deg (datasheet)
      calibrated 2026-11-18, 18 images, RMS 0.78 px (drone_geometry/camera_calibration_imx219_1640x1232.yaml):
      fx 1273.3, fy 1279.0, radtan distortion <= 2.9 %; raw field 65.6 x 51.4 deg.  The
      largest centred 4:3 pinhole whose every pixel lands inside the undistorted frame
      is 63.25 x 49.65 deg (`usable_field`): the spec below uses 63.2 x 49.6.
  1280x720  @ 60 fps  2560x1440 centre crop, binned   50.5 x 29.7 deg
At 40 deg tilt the cropped mode sees the gate being flown 8% of the time on the demo
flights, the full-field mode 31% (scripts/camera_visibility.py), so the full-field
mode is the default.  Tilt, full-field mode, gate centre in view on demo flights:
    tilt        25    30    35    40 deg
    2.5 m/s     50%   48%   42%   31%
    4.0 m/s     40%   41%   41%   36%
30 deg is best or tied at both speeds; trained 25-35 deg.

GateNet (fixed 2026-11-17): f=4 at 240x320 (H x W), the 4:3 shape of the full-field mode.
Orin Nano, TensorRT FP16: 2.1 ms median / 4.6 ms p99 alone (25 W, clocks not locked).
Deployed chain: ISP resize to 320x240 -> undistort onto `intrinsics(spec, 240, 320)` ->
GateNet -> resize the mask to 64x64.  Both pinholes span the same field, so that last
resize lands exactly on `policy_K` (tests/test_env.py).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from skydreamer import augment, gan
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


PICAM_V2_FULL = CameraSpec("picam_v2_1640x1232", 63.2, 49.6, (25.0, 30.0), 30.0, (25.0, 35.0))
PICAM_V2_720P = CameraSpec("picam_v2_1280x720", 50.5, 29.7,   # datasheet, not calibrated
                            (50.0, 60.0), 30.0, (25.0, 35.0))
CAMERAS = {c.name: c for c in (PICAM_V2_FULL, PICAM_V2_720P)}
DEFAULT = PICAM_V2_FULL

GATENET_F = 4
GATENET_HW = (240, 320)


def intrinsics(spec: CameraSpec, h: int, w: int):
    fx = 0.5 * w / np.tan(0.5 * spec.hfov_deg * DEG)
    fy = 0.5 * h / np.tan(0.5 * spec.vfov_deg * DEG)
    return float(fx), float(fy), 0.5 * w, 0.5 * h


def policy_K(spec: CameraSpec, h: int = 64, w: int = 64) -> np.ndarray:
    """The pinhole the deployed pipeline must resize undistorted frames onto."""
    fx, fy, cx, cy = intrinsics(spec, h, w)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def gatenet_K(spec: CameraSpec = DEFAULT) -> np.ndarray:
    """The pinhole GateNet's input is undistorted onto (GATENET_HW), in this module's
    convention (pixel centres at +0.5, so cx = W/2).  For OpenCV use `gatenet_K_opencv`."""
    return policy_K(spec, *GATENET_HW)


def gatenet_K_opencv(spec: CameraSpec = DEFAULT) -> np.ndarray:
    """The same pinhole in OpenCV's convention (pixel centres at integers: cx = (W-1)/2),
    i.e. the newCameraMatrix for cv2.initUndistortRectifyMap on the deployed Jetson."""
    K = gatenet_K(spec).copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


CALIBRATION = __import__("pathlib").Path(__file__).resolve().parents[2] / "drone_geometry" / \
    "camera_calibration_imx219_1640x1232.yaml"


def load_calibration(path=CALIBRATION):
    """-> dict(K (3x3), dist [k1 k2 p1 p2 k3] (OpenCV radtan), width, height)."""
    try:
        import yaml
        doc = yaml.safe_load(open(path))
    except ImportError:                     # the training venv has DreamerV3's ruamel.yaml
        from ruamel.yaml import YAML
        doc = YAML(typ="safe").load(open(path))
    c = doc["camera"]["calibration"]
    return dict(K=np.array(c["camera_matrix"]["data"], float).reshape(3, 3),
                dist=np.array(c["distortion_coefficients"]["data"], float),
                width=int(c["image_width"]), height=int(c["image_height"]))


def calibration_at(calib, w, h):
    """The same calibration for a frame the ISP resized to w x h (same field): K scales
    per axis about pixel centres, distortion is unchanged."""
    sx, sy = w / calib["width"], h / calib["height"]
    K = calib["K"].copy()
    K[0, 0] *= sx; K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy; K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return dict(calib, K=K, width=w, height=h)


def distort(calib, x, y):
    """Normalized pinhole coordinates -> raw image pixels (OpenCV radtan model)."""
    k1, k2, p1, p2, k3 = calib["dist"][:5]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = calib["K"]
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


def coverage(calib, spec: CameraSpec, h: int, w: int) -> float:
    """Share of the h x w pinhole (`intrinsics(spec)`) whose pixels land inside the raw
    frame: 1.0 means no black border after undistortion."""
    fx, fy, cx, cy = intrinsics(spec, h, w)
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    U, V = distort(calib, (u - cx) / fx, (v - cy) / fy)
    U, V = U + 0.5, V + 0.5          # pixel-centre convention -> edges at 0 and width
    return float(np.mean((U >= 0) & (U <= calib["width"]) & (V >= 0) & (V <= calib["height"])))


def usable_field(calib, aspect=None):
    """Largest centred pinhole field (h, v deg) with full coverage, at `aspect` = tan_v / tan_h
    (default: the frame's H/W)."""
    aspect = aspect or calib["height"] / calib["width"]
    u = np.linspace(-1, 1, 401)
    lo, hi = 0.2, 1.5
    for _ in range(50):
        th = 0.5 * (lo + hi)
        X, Y = np.meshgrid(u * th, u * th * aspect)
        U, V = distort(calib, X, Y)
        ok = np.all((U >= -0.5) & (U <= calib["width"] - 0.5) & (V >= -0.5) & (V <= calib["height"] - 0.5))
        lo, hi = (th, hi) if ok else (lo, th)
    return 2 * np.degrees(np.arctan(lo)), 2 * np.degrees(np.arctan(lo * aspect))


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


def render_augmented(state, cfg, spec: CameraSpec, stochgan: bool = False) -> jax.Array:
    """skydreamer.env._render_augmented for this camera: camera ahead of the CoM, mask
    erosion, rolling shutter.  The rolling-shutter strength is per radian of camera
    rotation, so it scales with the focal length in pixels.

    `stochgan`: the clean render goes through the StochGAN generator first (paper II-F:
    GAN, then rolling shutter and erosion), with weights from $STOCHGAN_WEIGHTS
    (skydreamer/gan.py).  Erosion is then the 1-pixel one on the GAN's output, never the
    geometric shrink of the render."""
    p_cam = state.s.p
    if cfg.cam_offset:
        p_cam = p_cam + quat_to_matrix(state.s.q)[..., :, 0] * cfg.cam_offset
    n = cfg.image_size
    if stochgan:
        mask = render_mask(p_cam, state.s.q, state.c_e, cfg.track, spec, n, n, 0.0)
        mask = gan.translate(gan.load(), (mask > 0.5).astype(jnp.float32),
                             jax.random.fold_in(state.key, 0x5106A))
        fx = intrinsics(spec, n, n)[0]
        return augment.apply(mask, state.s.omega_b, state.c_e, state.rs_s * fx / (SD_FOCAL_RATIO * n),
                             state.erode)
    shrink = 0.0 if cfg.erode_px is None else jnp.where(state.erode, cfg.erode_px, 0.0)
    mask = render_mask(p_cam, state.s.q, state.c_e, cfg.track, spec, n, n, shrink)
    erode = state.erode if cfg.erode_px is None else jnp.array(False)
    fx = intrinsics(spec, n, n)[0]
    return augment.apply(mask, state.s.omega_b, state.c_e, state.rs_s * fx / (SD_FOCAL_RATIO * n), erode)
