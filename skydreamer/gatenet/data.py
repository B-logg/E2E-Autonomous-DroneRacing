"""Datasets and augmentation for GateNet.

Expected layout (see docs/datasets.md):

    data/gatenet/
      real/
        images/0001.png     RGB, already remapped to the nominal K
        masks/0001.png      {0,255} grayscale, same basename
      backgrounds/*.jpg     scenes with no gates, for synthetic compositing
      cutouts/*.png         RGBA gate photos, alpha = the gate

Real and synthetic are mixed at a fixed ratio (MonoRace uses 3500 synthetic to
500 real); `MixedGateDataset` does that without materialising the synthetic set
on disk.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Appendix A: "Since our images are captured with a low exposure time (1 ms), we
# do not simulate motion blur with KernelBlur.  Instead, we apply stronger shot
# noise (standard deviation of 40 for pixel values in the range 0-255 instead
# of 25)."
SHOT_NOISE_STD = 40.0

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.float32)


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------


def augment_photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """HSV jitter, brightness/contrast, then shot noise.

    Geometry is handled separately because the mask has to follow it; these
    touch pixels only."""
    import cv2

    out = img.astype(np.float32)

    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.5, 1.5)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    out = (out - 128.0) * rng.uniform(0.7, 1.3) + 128.0 + rng.uniform(-25, 25)
    out += np.random.normal(0.0, SHOT_NOISE_STD, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_geometric(img: np.ndarray, mask: np.ndarray, rng: random.Random):
    """Scale, rotation and perspective warp, applied identically to both.

    Nearest-neighbour on the mask so it stays binary; the gate rails are only a
    few pixels wide and bilinear resampling erodes them away."""
    import cv2

    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, rng.uniform(-15, 15), rng.uniform(0.85, 1.25))
    M[0, 2] += rng.uniform(-0.05, 0.05) * w
    M[1, 2] += rng.uniform(-0.05, 0.05) * h
    M = np.vstack([M, [0, 0, 1]])

    jitter = 0.06
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32(
        [[rng.uniform(-jitter, jitter) * w, rng.uniform(-jitter, jitter) * h] for _ in range(4)]
    )
    M = cv2.getPerspectiveTransform(src, dst) @ M

    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


# --------------------------------------------------------------------------
# synthetic compositing
# --------------------------------------------------------------------------


class SyntheticGateCompositor:
    """MonoRace's recipe: gate cutouts under randomized scale / rotation /
    perspective warp, composited onto a gate-free background, plus HSV colour
    augmentation and Gaussian noise.

    The mask comes from the cutout's alpha channel, so it is exact and free --
    which is the whole point of synthetic data here."""

    def __init__(self, cutouts: Path, backgrounds: Path, size: int, seed: int = 0):
        self.cutouts = _list_images(Path(cutouts))
        self.backgrounds = _list_images(Path(backgrounds))
        if not self.cutouts:
            raise FileNotFoundError(f"no gate cutouts in {cutouts}")
        if not self.backgrounds:
            raise FileNotFoundError(f"no backgrounds in {backgrounds}")
        self.size = size
        self.rng = random.Random(seed)

    def __call__(self):
        import cv2

        rng = self.rng
        bg = _read_rgb(rng.choice(self.backgrounds))
        bg = cv2.resize(bg, (self.size, self.size), interpolation=cv2.INTER_AREA)

        rgba = cv2.imread(str(rng.choice(self.cutouts)), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[-1] != 4:
            raise ValueError("gate cutouts must be RGBA with the gate in the alpha channel")
        gate = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = (rgba[..., 3] > 127).astype(np.float32)

        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas_a = np.zeros((self.size, self.size), np.float32)
        scale = rng.uniform(0.25, 1.5) * self.size / max(gate.shape[:2])
        gh, gw = max(2, int(gate.shape[0] * scale)), max(2, int(gate.shape[1] * scale))
        gate = cv2.resize(gate, (gw, gh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_NEAREST)

        # Allow the gate to run off the edge -- close passes are mostly
        # partially-visible gates, and a net that has only seen whole gates
        # falls apart exactly when precision matters most.
        y0 = rng.randint(-gh // 2, self.size - gh // 2)
        x0 = rng.randint(-gw // 2, self.size - gw // 2)
        ys, ye = max(0, y0), min(self.size, y0 + gh)
        xs, xe = max(0, x0), min(self.size, x0 + gw)
        if ye > ys and xe > xs:
            canvas[ys:ye, xs:xe] = gate[ys - y0 : ye - y0, xs - x0 : xe - x0]
            canvas_a[ys:ye, xs:xe] = alpha[ys - y0 : ye - y0, xs - x0 : xe - x0]

        img, mask = augment_geometric(
            canvas.astype(np.uint8), canvas_a, random.Random(rng.random())
        )
        a = mask[..., None]
        img = (img.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)
        return augment_photometric(img, rng), mask


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def _to_tensors(img: np.ndarray, mask: np.ndarray):
    x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    y = torch.from_numpy(mask[None].astype(np.float32))
    return x, y


class RealGateDataset(Dataset):
    """Hand-labelled frames: `images/NAME.png` with `masks/NAME.png`."""

    def __init__(self, root, size: int, train: bool = True, seed: int = 0):
        root = Path(root)
        self.images = _list_images(root / "images")
        self.masks = root / "masks"
        missing = [p.name for p in self.images if not self._mask_for(p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images have no mask, e.g. {missing[:3]}")
        self.size, self.train = size, train
        self.rng = random.Random(seed)

    def _mask_for(self, image_path: Path) -> Path:
        for ext in IMAGE_EXTS:
            cand = self.masks / (image_path.stem + ext)
            if cand.exists():
                return cand
        return self.masks / (image_path.stem + ".png")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        import cv2

        img = _read_rgb(self.images[i])
        mask = _read_mask(self._mask_for(self.images[i]))
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = augment_geometric(img, mask, self.rng)
            img = augment_photometric(img, self.rng)
        return _to_tensors(img, mask)


class SyntheticGateDataset(Dataset):
    """Endless compositor output, presented as a fixed-length dataset."""

    def __init__(self, compositor: SyntheticGateCompositor, length: int):
        self.compositor, self.length = compositor, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return _to_tensors(*self.compositor())


class MixedGateDataset(Dataset):
    """Real and synthetic in a fixed ratio, e.g. MonoRace's 500:3500."""

    def __init__(self, real: Dataset, synthetic: Dataset):
        self.real, self.synthetic = real, synthetic

    def __len__(self):
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, i):
        if i < len(self.real):
            return self.real[i]
        return self.synthetic[i - len(self.real)]
