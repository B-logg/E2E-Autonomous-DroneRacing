#!/usr/bin/env python3
"""What the camera sees, mode by mode: renders the gate masks from the same drone
poses (one demonstration flight, floor launch) for each IMX219 mode at the 40 deg
mount, at each mode's own aspect ratio.  The gate being flown is orange, the
others white.

    python scripts/camera_views.py --out ../drone_geometry/camera_views.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

MODES = [
    ("assumed before\n175° fisheye 114.6×92.2°, tilt 50°", 114.6, 92.15, 50.0),
    ("1640×1232 @30 (full sensor)\n62.2×48.8°, tilt 40°", 62.2, 48.8, 40.0),
    ("3264×1848 @28 (full width)\n62.2×37.6°, tilt 40°", 62.2, 37.6, 40.0),
    ("1280×720 @60 (crop)\n50.5×29.7°, tilt 40°", 50.5, 29.7, 40.0),
    ("1920×1080 @30 (crop)\n38.9×22.5°, tilt 40°", 38.9, 22.5, 40.0),
]

# Raspberry Pi Camera Module 3 (Sony IMX708), for comparison only (not our camera).
# Fields from the module datasheet: standard lens 66 x 41 deg, wide lens 102 x 67 deg
# (full-sensor 16:9 modes, e.g. 2304x1296 binned).  The 1536x864 high-rate mode reads a
# 3072x1728 centre crop (2/3 of the width), so its field is narrower (pinhole estimate).
MODES_IMX708 = [
    MODES[0],
    ("IMX708 std 2304×1296 (full)\n66×41°, tilt 40°", 66.0, 41.0, 40.0),
    ("IMX708 std 1536×864 (crop)\n46.8×28.0°, tilt 40°", 46.8, 28.0, 40.0),
    ("IMX708 wide 2304×1296 (full)\n102×67°, tilt 40°", 102.0, 67.0, 40.0),
    ("IMX708 wide 1536×864 (crop)\n78.9×47.6°, tilt 40°", 78.9, 47.6, 40.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--speed", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tilt", type=float, default=40.0, help="tilt of the compared cameras, deg up")
    ap.add_argument("--camera", choices=("imx219", "imx708"), default="imx219")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    base = MODES if args.camera == "imx219" else MODES_IMX708
    stem = "camera_views" if args.camera == "imx219" else "camera_views_imx708"
    name = f"{stem}.png" if (args.tilt == 40.0 and args.camera == "imx219") else f"{stem}_tilt{args.tilt:g}.png"
    args.out = args.out or HERE.parent / "drone_geometry" / name
    modes = [base[0]] + [(n.replace("tilt 40°", f"tilt {args.tilt:g}°"), h, v, args.tilt) for n, h, v, _ in base[1:]]

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skydreamer import track as trk
    from skydreamer.dynamics import quat_to_euler, quat_to_matrix
    from ctba import airframe, camera as CAM, env as E, expert as X, tasks

    ccfg = tasks.make_config(f"c1_{args.track}")
    ccfg = ccfg._replace(base=ccfg.base._replace(train_fraction=0.0))
    tr = ccfg.base.track
    ref = X.reference(tr, args.speed)

    # one clean demonstration flight
    with jax.default_device(jax.devices("cpu")[0]):
        for seed in range(args.seed, args.seed + 40):
            st, _ = E.reset(jax.random.key(seed), ccfg)
            c = X.init(ref, st.s.p)
            step = jax.jit(lambda st, a: E.step(st, a, ccfg))
            ctrl = jax.jit(lambda c, st: X.control(c, st, ref, args.speed, 0.8, 0.6, X.Gains()))
            traj = []
            for _ in range(900):
                c, a = ctrl(c, st)
                st, _, _, term, _ = step(st, a)
                g = int((st.plane // trk.PLANES_PER_GATE) % tr.n_gates)
                traj.append((np.asarray(st.s.p), np.asarray(st.s.q), g, bool(st.airborne)))
                if bool(term):
                    break
            if not bool(term) and int(E.gates_passed(st)) >= 4:
                break
        print("flight seed", seed, "gates", int(E.gates_passed(st)))

        # moments: on the floor before G1, then ~2 m before each gate
        pos = np.asarray(tr.pos)
        moments = [("on the floor, G1 ahead", 0)]
        for g in range(tr.n_gates):
            for k, (p, q, gg, air) in enumerate(traj):
                if gg == g and air and np.linalg.norm(pos[g] - p) < 2.0:
                    moments.append((f"2 m before G{g + 1}", k))
                    break

        fig, axes = plt.subplots(len(moments), len(modes), figsize=(3.3 * len(modes), 2.9 * len(moments)),
                                 squeeze=False)
        W = 240
        for j, (name, hf, vf, tilt) in enumerate(modes):
            h = int(round(W * np.tan(np.radians(vf) / 2) / np.tan(np.radians(hf) / 2)))
            spec = CAM.CameraSpec(name, hf, vf, (30.0, 30.0), tilt, (tilt, tilt))
            c_e = jnp.array([0.0, np.radians(tilt), 0.0])
            for i, (label, k) in enumerate(moments):
                p, q, g, _ = traj[k]
                R = np.asarray(quat_to_matrix(jnp.asarray(q)))
                p_cam = jnp.asarray(p + R[:, 0] * airframe.cam_offset())
                onehot = jnp.arange(tr.n_gates) == g
                other = CAM.render_mask(p_cam, jnp.asarray(q), c_e, tr._replace(visible=tr.visible & ~onehot), spec, h, W)
                target = CAM.render_mask(p_cam, jnp.asarray(q), c_e, tr._replace(visible=tr.visible & onehot), spec, h, W)
                img = np.zeros((h, W, 3)) + 0.12
                img[np.asarray(other) > 0] = (0.85, 0.85, 0.85)
                img[np.asarray(target) > 0] = (1.0, 0.55, 0.1)
                ax = axes[i, j]
                ax.imshow(img, extent=(-hf / 2, hf / 2, -vf / 2, vf / 2))
                ax.set_xlim(-60, 60); ax.set_ylim(-48, 48)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor("white")
                seen = (np.asarray(target) > 0).mean() * 100
                ax.text(-58, -46, f"G{g + 1} {seen:.0f}% of frame", color="black", fontsize=8)
                if i == 0:
                    ax.set_title(name, fontsize=9)
                if j == 0:
                    pitch = np.degrees(float(quat_to_euler(jnp.asarray(q))[1]))
                    ax.set_ylabel(f"{label}\n(drone pitch {pitch:+.0f}°)", fontsize=9)
        title = "Pi Camera v2 (IMX219) modes on the Jetson" if args.camera == "imx219" else \
            "Pi Camera Module 3 (IMX708) modes, for comparison"
        fig.suptitle(f"{title}: gate masks from the same flight "
                     "(orange = gate being flown). Box size = field of view, same angular scale.", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
