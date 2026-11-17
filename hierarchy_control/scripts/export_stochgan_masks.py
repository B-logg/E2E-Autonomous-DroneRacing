#!/usr/bin/env python3
"""The "rendered" half of StochGAN's training data (paper II-F, Appendix B; docs/datasets.md).

Clean 64x64 binary gate masks from the CTBA simulator, exactly what the sim feeds the
GAN: our camera (ctba.camera: IMX219 full field on the anisotropic policy_K, tilt
randomized over the training range), camera ahead of the CoM, NO rolling shutter and NO
erosion (the paper applies those after the GAN).

Viewpoints come from flights, not random poses, so their distribution resembles the
real recordings: the CTBA expert flies the track at several speeds from the training
resets (start gate, position and attitude randomized), and every `--every` control
steps one frame is kept until the episode ends (crashes included: real footage has
those too).

    python scripts/export_stochgan_masks.py --out stochgan_rendered --n 5000
        [--track team44up] [--speeds 1.5 2.5 3.5] [--every 10] [--seed 0]

Writes  <out>/rendered/000000.png ...  (uint8 0/255, 64x64, 1 channel)
        <out>/meta.csv                 (file, episode, step, speed cmd, tilt, |v|, gate in view)
        <out>/README.txt               (camera, track, settings)

The real half is GateNet's output on recorded flights, undistorted onto
ctba.camera.gatenet_K() (240x320) and resized to 64x64: then both halves share one
pinhole.  Regenerate after the camera calibration (C3) and mount measurement (C6);
it takes minutes.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import zlib
import struct

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def write_png(path, img):
    """Grayscale 8-bit PNG without extra dependencies."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("stochgan_rendered"))
    ap.add_argument("--n", type=int, default=5000, help="masks to write")
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.5, 2.5, 3.5], help="expert v_nom, m/s")
    ap.add_argument("--every", type=int, default=10, help="keep one frame every N control steps (10 = 0.1 s)")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from ctba import camera as CAM
    from ctba import env as E
    from ctba import expert as X
    from ctba import tasks
    from skydreamer.dynamics import quat_to_matrix

    ccfg = tasks.make_config(f"c1_{args.track}", max_steps=args.max_steps)
    cfg, spec = ccfg.base, ccfg.camera
    n_keep = args.max_steps // args.every

    def clean_mask(st):
        p_cam = st.s.p
        if cfg.cam_offset:
            p_cam = p_cam + quat_to_matrix(st.s.q)[..., :, 0] * cfg.cam_offset
        return CAM.render_mask(p_cam, st.s.q, st.c_e, cfg.track, spec, 64, 64, 0.0)

    def rollout(key, v_nom, ref):
        st, _ = E.reset(key, ccfg)
        c = X.init(ref, st.s.p)

        def body(carry, i):
            st, c, done = carry
            c2, a = X.control(c, st, ref, v_nom, 0.8, 0.6, X.Gains())
            st2, _, _, term, trunc = E.step(st, a, ccfg)
            st2 = jax.tree.map(lambda x, y: jnp.where(done, y, x), st2, st)
            keep = (i % args.every == 0) & ~done
            out = jax.lax.cond(keep, lambda: clean_mask(st), lambda: jnp.zeros((64, 64)))
            return (st2, c2, done | term | trunc), (out, keep, jnp.linalg.norm(st.s.v), st.c_e[1])

        _, (m, k, v, tilt) = jax.lax.scan(body, (st, c, jnp.bool_(False)), jnp.arange(args.max_steps))
        sl = slice(0, None, args.every)
        return m[sl], k[sl], v[sl], tilt[sl]

    out_dir = args.out / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_speed = -(-args.n // len(args.speeds))          # equal share per speed
    pool = {v: [] for v in args.speeds}                 # (mask, episode, step, tilt, speed)
    ep = 0
    key = jax.random.key(args.seed)
    with jax.default_device(jax.devices("cpu")[0]):
        for v_nom in args.speeds:
            ref = X.reference(cfg.track, v_nom)
            fn = jax.jit(jax.vmap(lambda k, ref=ref, v=v_nom: rollout(k, v, ref)))
            while len(pool[v_nom]) < per_speed:
                key, sub = jax.random.split(key)
                m, k, vel, tilt = map(np.asarray, fn(jax.random.split(sub, args.batch)))
                for e in range(args.batch):
                    for j in np.nonzero(k[e])[0]:
                        pool[v_nom].append((m[e, j] > 0.5, ep, int(j) * args.every,
                                            float(np.degrees(tilt[e, j])), float(vel[e, j])))
                    ep += 1
                print(f"  {v_nom} m/s: {len(pool[v_nom])} candidate frames ({ep} episodes so far)", flush=True)

    rng = np.random.default_rng(args.seed)
    chosen = []
    for v_nom, items in pool.items():
        idx = rng.choice(len(items), size=min(per_speed, len(items)), replace=False)
        chosen += [(v_nom,) + items[i] for i in idx]
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    rows = []
    for n, (v_nom, m, e, step, tilt, vel) in enumerate(chosen):
        img = m.astype(np.uint8) * 255
        name = f"{n:06d}.png"
        write_png(out_dir / name, img)
        rows.append([name, e, step, v_nom, round(tilt, 2), round(vel, 2), int(img.any())])
    n = len(rows)

    with open(args.out / "meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "episode", "step", "v_nom", "tilt_deg", "speed", "gate_in_view"])
        w.writerows(rows)
    shown = np.mean([r[-1] for r in rows])
    K = CAM.policy_K(spec)
    (args.out / "README.txt").write_text(
        f"StochGAN rendered domain: {n} clean 64x64 binary masks (0/255), no rolling shutter, no erosion.\n"
        f"camera: {spec.name}, field {spec.hfov_deg} x {spec.vfov_deg} deg, tilt {spec.tilt_range[0]}-{spec.tilt_range[1]} deg"
        f" (+-{spec.roll_yaw_range} deg roll/yaw)\n"
        f"64x64 pinhole: fx {K[0, 0]:.3f}, fy {K[1, 1]:.3f}, cx {K[0, 2]}, cy {K[1, 2]}\n"
        f"track: {args.track}; expert speeds {args.speeds} m/s; one frame every {args.every} steps "
        f"({args.every * 10} ms); seed {args.seed}\n"
        f"frames with a gate in view: {100 * shown:.0f}%\n"
        "real domain must be: GateNet mask on the undistorted 240x320 image (ctba.camera.gatenet_K), "
        "resized to 64x64, thresholded to 0/255.\n")
    print(f"wrote {n} masks to {out_dir}  (gate in view in {100 * shown:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
