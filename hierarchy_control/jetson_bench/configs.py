"""Benchmark configurations, importable without torch."""

# (f, H, W): the paper's two settings, then 16:9 / 4:3 inputs a Pi camera image would be
# resized to.  Divisible by 16 (four 2x poolings) except the paper's 196 (handled by interpolation).
GATENET_CONFIGS = [
    (2, 196, 196), (4, 384, 384),
    (4, 192, 320), (2, 192, 320),
    (4, 256, 448), (2, 256, 448),
    (4, 352, 640), (2, 352, 640),
    (4, 240, 320), (4, 480, 640),
]


def gatenet_name(f, h, w):
    return f"gatenet_f{f}_{h}x{w}"
