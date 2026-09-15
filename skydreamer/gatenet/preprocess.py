"""Map real camera images onto SkyDreamer's nominal pinhole model (paper II-E).

    K = [[25/64 W, 0, W/2], [0, 25/64 H, H/2], [0, 0, 1]]

This is the single most important piece of glue between the real drone and the
simulator.  The policy is trained on images that obey this K exactly, and it is
*only* invariant to the real lens because every real frame is remapped onto it
first.  Extrinsics are deliberately NOT corrected -- the world model estimates
those on the fly (that is the paper's "no extrinsic calibration" claim).

The onboard lens is a 175-degree Arducam IMX219, so the equidistant (fisheye)
model is the default; `radtan` is there for narrower lenses.  Calibrate with
Kalibr, as the paper's authors did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FOCAL_RATIO = 25.0 / 64.0


def nominal_K(h: int, w: int) -> np.ndarray:
    return np.array(
        [[FOCAL_RATIO * w, 0.0, 0.5 * w], [0.0, FOCAL_RATIO * h, 0.5 * h], [0.0, 0.0, 1.0]],
        np.float64,
    )


@dataclass
class CameraCalibration:
    """Intrinsics of the physical camera, from Kalibr or cv2.calibrateCamera."""

    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (4,) equidistant, or (4..8,) radtan
    width: int
    height: int
    model: str = "equidistant"  # or "radtan"

    @classmethod
    def from_yaml(cls, path):
        """Read a Kalibr `camchain` entry (`intrinsics: [fx, fy, cx, cy]`)."""
        import yaml

        with open(path) as fh:
            doc = yaml.safe_load(fh)
        cam = doc[next(iter(doc))] if "cam0" not in doc else doc["cam0"]
        fx, fy, cx, cy = cam["intrinsics"]
        model = "equidistant" if "equi" in cam.get("distortion_model", "") else "radtan"
        return cls(
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64),
            dist=np.asarray(cam["distortion_coeffs"], np.float64),
            width=int(cam["resolution"][0]),
            height=int(cam["resolution"][1]),
            model=model,
        )


class NominalRemapper:
    """Precomputes the remap once, then applies it per frame.

    Build it once at startup and reuse -- `initUndistortRectifyMap` is far too
    slow to run inside a 90 Hz loop."""

    def __init__(self, calib: CameraCalibration, out_h: int, out_w: int):
        import cv2

        self.out_h, self.out_w = out_h, out_w
        self.K_new = nominal_K(out_h, out_w)
        size = (out_w, out_h)
        if calib.model == "equidistant":
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                calib.K, calib.dist[:4].reshape(4, 1), np.eye(3), self.K_new, size, cv2.CV_16SC2
            )
        else:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                calib.K, calib.dist, np.eye(3), self.K_new, size, cv2.CV_16SC2
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def check_coverage(self, sample: np.ndarray) -> float:
        """Fraction of output pixels that map inside the source image.

        The 25/64 focal ratio was chosen so the undistorted image has no black
        borders.  If this comes back below ~0.99 for your lens, the policy will
        see borders it never saw in training -- widen the crop or lower the
        output resolution before you waste a training run."""
        import cv2

        probe = np.full(sample.shape[:2] + (1,), 255, np.uint8)
        out = cv2.remap(probe, self.map1, self.map2, interpolation=cv2.INTER_NEAREST)
        return float((out > 0).mean())
