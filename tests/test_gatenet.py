"""GateNet checks, including an end-to-end overfit on generated data.

The overfit test is the important one: it proves the architecture, the loss
weighting and the data path can actually drive IoU up, so when real labels
arrive a failure is a data problem rather than a code problem.
"""

import numpy as np
import pytest
import torch

from skydreamer.gatenet.data import augment_geometric, augment_photometric
from skydreamer.gatenet.losses import SCALE_WEIGHTS, dice_loss, downsample_to, iou, multiscale_loss
from skydreamer.gatenet.model import GateNet, build
from skydreamer.gatenet.preprocess import nominal_K

cv2 = pytest.importorskip("cv2")


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("gate,f,res", [("mavlab", 2, 196), ("orange", 4, 384)])
def test_build_matches_appendix_a(gate, f, res):
    model, size = build(gate)
    assert (model.f, size) == (f, res)


def test_five_heads_finest_first():
    """The loss weights 4, 2, 1, 1, 1 are meant to emphasize high resolution,
    so head 0 must be the full-resolution one."""
    net = GateNet(f=4)
    outs = net(torch.randn(1, 3, 64, 64))
    assert len(outs) == len(SCALE_WEIGHTS) == 5
    sizes = [o.shape[-1] for o in outs]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[0] == 64
    assert SCALE_WEIGHTS[0] == max(SCALE_WEIGHTS)


def test_skips_are_additive_not_concatenated():
    """Appendix A says the decoder *adds* the encoder skip.  With concatenation
    the first decoder conv would take twice the channels; assert the shape that
    only the additive form produces."""
    net = GateNet(f=4)
    assert net.up1.conv.net[0].in_channels == net.up1.up.out_channels


def test_handles_sizes_not_divisible_by_sixteen():
    """196 / 2^4 is not an integer, so the transposed convs come back a pixel
    short and have to be resized onto the skip."""
    net, size = build("mavlab")
    assert size % 16 != 0
    outs = net(torch.randn(1, 3, size, size))
    assert outs[0].shape[-2:] == (size, size)


def test_xavier_initialization_applied():
    net = GateNet(f=4)
    conv = net.inc.net[0]
    fan_in, fan_out = conv.in_channels * 9, conv.out_channels * 9
    expected = float(np.sqrt(2.0 / (fan_in + fan_out)))
    assert conv.weight.std().item() == pytest.approx(expected, rel=0.35)


def test_predict_returns_binary_full_resolution():
    net = GateNet(f=4)
    out = net.predict(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert set(np.unique(out.numpy())).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def test_dice_is_zero_for_a_perfect_prediction():
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    logits = torch.where(target > 0, 20.0, -20.0)
    assert float(dice_loss(logits, target)) == pytest.approx(0.0, abs=1e-4)


def test_iou_endpoints():
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    assert float(iou(torch.where(target > 0, 20.0, -20.0), target)) == pytest.approx(1.0)
    assert float(iou(torch.where(target > 0, -20.0, 20.0), target)) == pytest.approx(0.0)


def test_label_downsampling_keeps_thin_structures():
    """Gate rails are a few pixels wide.  Nearest-neighbour downsampling drops
    them entirely, which silently removes the supervision on the coarse heads."""
    m = torch.zeros(1, 1, 64, 64)
    m[..., 30:32, :] = 1.0  # a 2px horizontal rail
    small = downsample_to(m, (8, 8))
    assert float(small.sum()) > 0


def test_multiscale_weighting_is_applied():
    net = GateNet(f=4)
    x, y = torch.randn(1, 3, 32, 32), torch.zeros(1, 1, 32, 32)
    outs = net(x)
    total, per = multiscale_loss(outs, y)
    manual = sum(w * float(v) for w, v in zip(SCALE_WEIGHTS, per))
    assert total.detach().item() == pytest.approx(manual, rel=1e-5)


# --------------------------------------------------------------------------
# preprocessing and augmentation
# --------------------------------------------------------------------------


def test_nominal_intrinsics_match_the_paper():
    K = nominal_K(64, 64)
    assert K[0, 0] == pytest.approx(25.0)  # 25/64 * 64
    assert K[0, 2] == pytest.approx(32.0)
    assert nominal_K(384, 384)[0, 0] == pytest.approx(150.0)


def test_geometric_augmentation_keeps_the_mask_binary():
    import random

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    mask = np.zeros((64, 64), np.float32)
    mask[20:40, 20:40] = 1.0
    _, out = augment_geometric(img, mask, random.Random(0))
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_photometric_augmentation_stays_in_range():
    import random

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    out = augment_photometric(img, random.Random(0))
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def _synthetic_batch(n, size, seed=0):
    """Bright rectangular annuli on noisy backgrounds -- a toy stand-in for
    gates that the real dataset will replace."""
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    for _ in range(n):
        img = rng.integers(0, 90, (size, size, 3), dtype=np.uint8)
        m = np.zeros((size, size), np.float32)
        h = rng.integers(size // 3, size // 2)
        y0 = rng.integers(0, size - h)
        x0 = rng.integers(0, size - h)
        t = max(2, h // 8)
        m[y0 : y0 + h, x0 : x0 + h] = 1.0
        m[y0 + t : y0 + h - t, x0 + t : x0 + h - t] = 0.0
        img[m > 0] = np.array([240, 120, 30], np.uint8)
        imgs.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        masks.append(m[None])
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(masks))


def test_gatenet_learns_to_segment_gates():
    """Train briefly on generated gate-like annuli and require real IoU."""
    torch.manual_seed(0)
    size = 64
    x, y = _synthetic_batch(16, size, seed=1)
    net = GateNet(f=4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    start = float(iou(net(x)[0], y))
    for _ in range(120):
        loss, _ = multiscale_loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        end = float(iou(net(x)[0], y))
    assert end > 0.8, f"IoU only reached {end:.3f} (started {start:.3f})"


def test_mask_feeds_the_policy_at_64x64():
    """GateNet's output is resized to 64x64 before SkyDreamer sees it (II-E);
    check the handoff produces exactly the env's observation shape."""
    from skydreamer.env import IMAGE_SIZE

    net, size = build("orange")
    mask = net.predict(torch.zeros(1, 3, size, size))[0, 0].numpy()
    resized = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    obs = (resized > 0.5).astype(np.uint8)[..., None] * 255
    assert obs.shape == (IMAGE_SIZE, IMAGE_SIZE, 1)
