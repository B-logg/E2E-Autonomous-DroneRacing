#!/usr/bin/env python3
"""How often the camera sees the gate being flown, for a given lens / mode / tilt.

Flies the CTBA demonstration controller (camera-independent: it reads the true
state) on the track and, at every step, projects the next gate's frame into the
camera.  Reports the fraction of airborne steps where (a) the gate centre and
(b) any part of the frame (24 points around it) is inside the field of view
and in front of the camera, within 10 m.

    python scripts/camera_visibility.py [--track team44up] [--episodes 64] [--speed 2.5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# (name, horizontal deg, vertical deg, tilt up deg)
CAMERAS = [
    ("assumed before: 175 deg fisheye 114.6x92.15, tilt 50", 114.6, 92.15, 50.0),
    # Pi Camera v2 (IMX219, 3.04 mm) Jetson CSI modes, tilt 40 (fixed mount)
    ("3264x2464 @21  full sensor        62.2x48.8", 62.2, 48.8, 40.0),
    ("1640x1232 @30  full sensor, bin2  62.2x48.8", 62.2, 48.8, 40.0),
    ("3264x1848 @28  full width, 16:9   62.2x37.6", 62.2, 37.6, 40.0),
    ("1920x1080 @30  centre crop 1:1    38.9x22.5", 38.9, 22.5, 40.0),
    ("1280x720  @60  crop 2560x1440 bin2 50.5x29.7", 50.5, 29.7, 40.0),
]


def frame_points(track, g, n_side=6):
    """Points along the centre line of gate g's frame (world), plus its centre."""
    import numpy as np
    pos, yaw = np.asarray(track.pos[g]), float(track.yaw[g])
    r = (float(track.inner[g]) + float(track.outer[g])) / 4      # mid-frame half size
    s = np.linspace(-r, r, n_side, endpoint=False)
    sq = np.concatenate([np.c_[s, np.full_like(s, -r)], np.c_[np.full_like(s, r), s],
                         np.c_[-s, np.full_like(s, r)], np.c_[np.full_like(s, -r), -s]])
    # gate frame: x along the normal, y lateral, z vertical (NED)
    c, sn = np.cos(yaw), np.sin(yaw)
    lat = np.array([-sn, c, 0.0])
    pts = pos + sq[:, :1] * lat + sq[:, 1:2] * np.array([0.0, 0.0, 1.0])
    return np.vstack([pos, pts])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--speed", type=float, default=2.5)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from skydreamer.dynamics import quat_to_matrix
    from skydreamer import track as trk
    from ctba import airframe, env as E, expert as X, tasks

    ccfg = tasks.make_config(f"c1_{args.track}")
    ccfg = ccfg._replace(base=ccfg.base._replace(train_fraction=0.0))   # floor launches, like the real start
    tr = ccfg.base.track
    ref = X.reference(tr, args.speed)

    def one(key):
        st, _ = E.reset(key, ccfg)
        c = X.init(ref, st.s.p)

        def body(carry, _):
            st, c, done = carry
            c2, a = X.control(c, st, ref, args.speed, 0.8, 0.6, X.Gains())
            st2, _, _, term, trunc = E.step(st, a, ccfg)
            gate = (st2.plane // trk.PLANES_PER_GATE) % tr.n_gates
            out = dict(p=st2.s.p, q=st2.s.q, gate=gate, live=~(done | term | trunc), air=st2.airborne)
            return (st2, c2, done | term | trunc), out

        _, o = jax.lax.scan(body, (st, c, jnp.bool_(False)), None, length=ccfg.base.max_steps)
        return o

    with jax.default_device(jax.devices("cpu")[0]):
        o = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(7), args.episodes))
        o = {k: np.asarray(v) for k, v in o.items()}
        R = np.asarray(jax.vmap(jax.vmap(quat_to_matrix))(jnp.asarray(o["q"])))
    pts = {g: frame_points(tr, g) for g in range(tr.n_gates)}
    m = o["live"] & o["air"]
    cam_off = airframe.cam_offset()
    res = {}
    for name, hf, vf, tilt in CAMERAS:
        t = np.radians(tilt)
        # camera axes in body FRD: forward tilted up, right = body y, down
        fwd = np.array([np.cos(t), 0.0, -np.sin(t)])
        down = np.array([np.sin(t), 0.0, np.cos(t)])
        right = np.array([0.0, 1.0, 0.0])
        th, tv = np.tan(np.radians(hf) / 2), np.tan(np.radians(vf) / 2)
        cen, anyv = [], []
        for e, k in zip(*np.nonzero(m)):
            Rb = R[e, k]
            pc = o["p"][e, k] + Rb[:, 0] * cam_off
            d = (pts[int(o["gate"][e, k])] - pc) @ Rb               # body frame
            z = d @ fwd; x = d @ right; y = d @ down
            ok = (z > 0.05) & (np.linalg.norm(d, axis=1) < 10.0)
            ok &= (np.abs(x) <= th * z) & (np.abs(y) <= tv * z)
            cen.append(ok[0]); anyv.append(ok[1:].any())
        res[name] = dict(centre=float(np.mean(cen)), any_part=float(np.mean(anyv)))
        print(f"{name:58s} centre visible {100 * np.mean(cen):5.1f}%   any part {100 * np.mean(anyv):5.1f}%")
    print(f"({int(m.sum())} airborne steps, {args.episodes} floor launches, demo controller at {args.speed} m/s)")
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
