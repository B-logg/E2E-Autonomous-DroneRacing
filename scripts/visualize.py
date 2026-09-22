#!/usr/bin/env python3
"""Fly a trained policy and render what it did.

Produces, per episode:

  flight_<i>.gif    what the policy sees (the segmentation mask it is actually
                    given, delays and augmentation included) next to the
                    trajectory building up in top-down and side view
  figure_<i>.png    a still in the style of the paper's Figures 4 and 6:
                    ground truth against the world model's decoded position,
                    coloured by speed, with the camera's principal axis drawn
                    as arrows

The camera arrows are the interesting part.  The paper has no perception
reward, so nothing tells the drone to look at the gates -- if the arrows point
at them anyway, that behaviour is emergent, which is one of the paper's claims.

Usage
  python scripts/visualize.py --logdir logdir/small-20260920-070239
  python scripts/visualize.py --logdir ... --episodes 5 --laps 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
DV3 = ROOT / "third_party" / "dreamerv3"
if DV3.exists():
    sys.path.insert(0, str(DV3))

GROUND, SKY = "#f2f2ef", "#ffffff"
INK, MUTED = "#1a1a1a", "#8a8a8a"
TRUTH, DECODED = "#1a1a1a", "#2d7dd2"


def fly(agent, cfg, seed, laps):
    """One episode, recording everything needed to draw it."""
    import jax
    import jax.numpy as jnp

    from skydreamer.dynamics import CONTROL_DT
    from skydreamer.env import gates_passed, reset, step, unpack_mask
    from skydreamer.render import camera_rotation

    with jax.transfer_guard("allow"):
        st, obs = reset(jax.random.key(seed), cfg)
        carry = agent.init_policy(1)
        target = laps * cfg.track.n_gates
        rec = {k: [] for k in ("p", "v", "mask", "axis", "decoded", "gates", "speed")}
        is_first, crashed = True, False

        for t in range(cfg.max_steps):
            o = {k: np.asarray(v)[None] for k, v in obs.items()}
            o["mask"] = o["mask"].astype(np.uint8)
            o = {k: (v if v.dtype == np.uint8 else v.astype(np.float32)) for k, v in o.items()}
            o.update(
                reward=np.zeros(1, np.float32), is_first=np.array([is_first]),
                is_last=np.zeros(1, bool), is_terminal=np.zeros(1, bool))
            carry, act, outs = agent.policy(carry, o, mode="eval")
            is_first = False
            u = np.clip((np.asarray(act["action"], np.float32)[0] + 1.0) / 2.0, 0.0, 1.0)

            rec["mask"].append(np.asarray(unpack_mask(obs["mask"])))
            rec["p"].append(np.asarray(st.s.p))
            rec["v"].append(np.asarray(st.s.v))
            rec["speed"].append(float(np.linalg.norm(np.asarray(st.s.v))))
            rec["axis"].append(np.asarray(camera_rotation(st.s.q, st.c_e) @ jnp.array([1.0, 0, 0])))
            rec["decoded"].append(
                np.asarray(outs["recon_info_p_w"])[0] if "recon_info_p_w" in outs else None)
            rec["gates"].append(int(gates_passed(st)))

            st, obs, rew, term, trunc = step(st, jnp.asarray(u), cfg)
            if bool(term):
                crashed = True
                break
            if int(gates_passed(st)) >= target:
                break

    out = {k: np.array([x for x in v if x is not None]) for k, v in rec.items()}
    out["crashed"] = crashed
    out["duration"] = len(rec["p"]) * CONTROL_DT
    return out


def _gate_segments(track):
    """Endpoints for drawing each gate edge-on in both views."""
    pos, yaw = np.asarray(track.pos), np.asarray(track.yaw)
    outer, vis = np.asarray(track.outer), np.asarray(track.visible)
    segs = []
    for p, y, o, v in zip(pos, yaw, outer, vis):
        d = np.array([-np.sin(y), np.cos(y)])          # in-plane, horizontal
        a, b = p[:2] - d * o / 2, p[:2] + d * o / 2
        segs.append(dict(top=(a, b), side=(p[1], -p[2], o), visible=bool(v)))
    return segs


def _gate_rings(track):
    """Outer and inner square outlines of each gate, in 3D (altitude up)."""
    pos, yaw = np.asarray(track.pos), np.asarray(track.yaw)
    inner, outer = np.asarray(track.inner), np.asarray(track.outer)
    vis = np.asarray(track.visible)
    rings = []
    for p, y, i_, o_, v in zip(pos, yaw, inner, outer, vis):
        d1 = np.array([-np.sin(y), np.cos(y), 0.0])     # in-plane, horizontal
        d2 = np.array([0.0, 0.0, 1.0])                  # in-plane, vertical
        c = np.array([p[0], p[1], -p[2]])               # altitude up
        loops = []
        for half in (o_ / 2, i_ / 2):
            corners = [c + a * half * d1 + b * half * d2
                       for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
            loops.append(np.array(corners))
        rings.append(dict(loops=loops, visible=bool(v)))
    return rings


def _setup3d(ax, bounds):
    ax.set_facecolor(SKY)
    ax.set_title("third person", color=INK, fontsize=9, pad=2)
    (xl, yl, zl) = bounds
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], max(1.0, zl[1] - zl[0])))
    ax.view_init(elev=22, azim=-62)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane._axinfo["grid"]["color"] = "#eeeeee"
        pane.set_tick_params(colors=MUTED, labelsize=6)
    ax.set_xlabel("X [m]", color=MUTED, fontsize=7, labelpad=2)
    ax.set_ylabel("Y [m]", color=MUTED, fontsize=7, labelpad=2)
    ax.set_zlabel("alt [m]", color=MUTED, fontsize=7, labelpad=2)


def _draw_gates3d(ax, rings):
    for g in rings:
        style = dict(color=INK, lw=1.8) if g["visible"] else dict(
            color=MUTED, lw=1.2, linestyle=(0, (3, 2)))
        for loop in g["loops"]:
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], **style)


def _setup(ax, xlabel, ylabel, title):
    ax.set_facecolor(SKY)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.set_title(title, color=INK, fontsize=9, pad=6)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    for s in ax.spines.values():
        s.set_color("#dddddd")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)


def _draw_gates(ax_top, ax_side, segs):
    for g in segs:
        style = dict(color=INK, lw=3, solid_capstyle="butt") if g["visible"] else dict(
            color=MUTED, lw=2, linestyle=(0, (3, 2)))
        (a, b) = g["top"]
        ax_top.plot([a[1], b[1]], [a[0], b[0]], **style)
        y, alt, o = g["side"]
        ax_side.plot([y, y], [alt - o / 2, alt + o / 2], **style)


def render(ep, track, out_stem, stride, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image

    p, speed, axis = ep["p"], ep["speed"], ep["axis"]
    segs = _gate_segments(track)
    vmax = max(1e-3, float(speed.max()))
    # Speed is a magnitude, so it gets a sequential ramp -- viridis is
    # perceptually uniform and is what the paper uses.
    cmap = plt.get_cmap("viridis")

    xs, ys, alt = p[:, 0], p[:, 1], -p[:, 2]
    pad = 1.0
    # Include the gates in the limits, or half a gate hangs outside the frame.
    gpos, gout = np.asarray(track.pos), np.asarray(track.outer)
    gx, gy, galt = gpos[:, 0], gpos[:, 1], -gpos[:, 2]
    half = gout.max() / 2
    xlim = (min(ys.min(), gy.min() - half) - pad, max(ys.max(), gy.max() + half) + pad)
    ylim = (min(xs.min(), gx.min() - half) - pad, max(xs.max(), gx.max() + half) + pad)
    zlim = (0.0, max(alt.max(), (galt + half).max()) + pad)

    rings = _gate_rings(track)
    bounds3d = (
        (min(xs.min(), gx.min()) - pad, max(xs.max(), gx.max()) + pad),
        (min(ys.min(), gy.min()) - pad, max(ys.max(), gy.max()) + pad),
        (0.0, max(alt.max(), 4.5) + 0.5),
    )

    frames = []
    for i in range(0, len(p), stride):
        fig = plt.figure(figsize=(10.5, 7.4))
        fig.patch.set_facecolor(GROUND)
        axm = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 2, projection="3d")
        axt = fig.add_subplot(2, 2, 3)
        axs_ = fig.add_subplot(2, 2, 4)

        axm.imshow(ep["mask"][i], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axm.set_title("what the policy sees", color=INK, fontsize=9, pad=6)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_color("#dddddd")

        # third person: gates, the trail so far, and the drone with its camera axis
        _setup3d(ax3, bounds3d)
        _draw_gates3d(ax3, rings)
        if i > 1:
            seg = np.stack([xs[: i + 1], ys[: i + 1], alt[: i + 1]], -1)
            pts = seg.reshape(-1, 1, 3)
            lc3 = Line3DCollection(
                np.concatenate([pts[:-1], pts[1:]], axis=1),
                cmap=cmap, norm=plt.Normalize(0, vmax), linewidth=1.8)
            lc3.set_array(speed[: i + 1])
            ax3.add_collection3d(lc3)
        a = axis[i]
        ax3.plot([xs[i]], [ys[i]], [alt[i]], "o", ms=7,
                 color=cmap(speed[i] / vmax), markeredgecolor="white", markeredgewidth=1.4)
        ax3.plot([xs[i], xs[i] + a[0]], [ys[i], ys[i] + a[1]],
                 [alt[i], alt[i] - a[2]], color=INK, lw=1.4)

        for ax, (h, v), lims, labels, title in (
            (axt, (ys[: i + 1], xs[: i + 1]), (xlim, ylim), ("Y [m]", "X [m]"), "top-down"),
            (axs_, (ys[: i + 1], alt[: i + 1]), (xlim, zlim), ("Y [m]", "altitude [m]"), "side"),
        ):
            _setup(ax, *labels, title)
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal")
            if len(h) > 1:
                pts = np.array([h, v]).T.reshape(-1, 1, 2)
                lc = LineCollection(
                    np.concatenate([pts[:-1], pts[1:]], axis=1),
                    cmap=cmap, norm=plt.Normalize(0, vmax), linewidth=2)
                lc.set_array(speed[: i + 1])
                ax.add_collection(lc)
        _draw_gates(axt, axs_, segs)

        axt.arrow(ys[i], xs[i], a[1] * 0.7, a[0] * 0.7, head_width=0.16, color=INK, lw=1.2)
        axs_.arrow(ys[i], alt[i], a[1] * 0.7, -a[2] * 0.7, head_width=0.16, color=INK, lw=1.2)
        for ax, h, v in ((axt, ys[i], xs[i]), (axs_, ys[i], alt[i])):
            ax.plot(h, v, "o", ms=6, color=cmap(speed[i] / vmax),
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        fig.suptitle(
            f"t = {i * 0.011:5.2f} s     {speed[i]:5.1f} m/s     "
            f"gates passed: {ep['gates'][i]}",
            color=INK, fontsize=11, y=0.985)
        fig.tight_layout(rect=(0.0, 0.10, 1, 0.95))
        cax = fig.add_axes((0.30, 0.055, 0.40, 0.016))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("speed [m/s]", color=MUTED, fontsize=7, labelpad=2)
        cb.ax.tick_params(colors=MUTED, labelsize=6, length=2)
        cb.outline.set_edgecolor("#dddddd")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    gif = out_stem.with_suffix(".gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return gif


def still(ep, track, out_png):
    """Paper Figure 4/6 style: ground truth vs the world model's belief."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    p, speed, axis = ep["p"], ep["speed"], ep["axis"]
    dec = ep["decoded"] if len(ep["decoded"]) == len(p) else None
    segs = _gate_segments(track)
    vmax = max(1e-3, float(speed.max()))
    cmap = plt.get_cmap("viridis")
    xs, ys, alt = p[:, 0], p[:, 1], -p[:, 2]

    fig, (axt, axs_) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(GROUND)
    for ax, (h, v), labels, title in (
        (axt, (ys, xs), ("Y [m]", "X [m]"), "Top-down view"),
        (axs_, (ys, alt), ("Y [m]", "altitude [m]"), "Side view"),
    ):
        _setup(ax, *labels, title)
        ax.set_aspect("equal")
        pts = np.array([h, v]).T.reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                            cmap=cmap, norm=plt.Normalize(0, vmax), linewidth=2.2)
        lc.set_array(speed)
        ax.add_collection(lc)
        ax.autoscale_view()
    _draw_gates(axt, axs_, segs)

    if dec is not None:
        axt.plot(dec[:, 1], dec[:, 0], color=DECODED, lw=1.1, alpha=0.85,
                 label="world model's belief")
        axs_.plot(dec[:, 1], -dec[:, 2], color=DECODED, lw=1.1, alpha=0.85)
        err = float(np.linalg.norm(dec - p, axis=-1).mean())
        axt.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    else:
        err = float("nan")

    every = max(1, len(p) // 26)
    for i in range(0, len(p), every):
        a = axis[i]
        axt.arrow(ys[i], xs[i], a[1] * 0.55, a[0] * 0.55, head_width=0.13, color=INK, lw=0.9)
        axs_.arrow(ys[i], alt[i], a[1] * 0.55, -a[2] * 0.55, head_width=0.13, color=INK, lw=0.9)

    cb = fig.colorbar(lc, ax=[axt, axs_], fraction=0.025, pad=0.02)
    cb.set_label("speed [m/s]", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor("#dddddd")

    gates = ep["gates"][-1]
    fig.suptitle(
        f"{gates} gates in {ep['duration']:.2f} s    peak {speed.max():.1f} m/s"
        + ("" if np.isnan(err) else f"    mean decode error {err:.2f} m")
        + ("    (crashed)" if ep["crashed"] else ""),
        color=INK, fontsize=11, y=0.97)
    fig.savefig(out_png, dpi=150, facecolor=GROUND, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--stride", type=int, default=3, help="render every Nth sim step")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    if not (logdir / "config.yaml").exists():
        sys.exit(f"no config.yaml in {logdir}")
    out = args.out or (logdir / "video")
    out.mkdir(parents=True, exist_ok=True)

    import ruamel.yaml as yaml

    task = yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())["task"]
    track_name = task.split("_", 1)[1]

    from evaluate import build_agent

    from skydreamer.embodied_env import SkyDreamer

    env = SkyDreamer(track_name, max_steps=args.laps * 1200, seed=args.seed)
    agent, _ = build_agent(logdir, env)
    print(f"flying {args.episodes} episodes on {track_name}, {args.laps} laps each", flush=True)

    for i in range(args.episodes):
        ep = fly(agent, env.cfg, args.seed + i, args.laps)
        stem = out / f"flight_{i}"
        png = still(ep, env.cfg.track, out / f"figure_{i}.png")
        gif = render(ep, env.cfg.track, stem, args.stride, args.fps)
        print(f"  episode {i}: {ep['gates'][-1]} gates, {ep['duration']:.2f} s, "
              f"peak {ep['speed'].max():.1f} m/s"
              f"{' (crashed)' if ep['crashed'] else ''}", flush=True)
        print(f"    {gif.name}  {png.name}", flush=True)

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
