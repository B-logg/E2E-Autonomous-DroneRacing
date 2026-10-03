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
FLOOR, FLOOR_GRID = "#e4e4df", "#c9c9c1"
INK, MUTED = "#1a1a1a", "#8a8a8a"
TRUTH, DECODED = "#1a1a1a", "#2d7dd2"


def fly(agent, cfg, seed, laps, expert=None):
    """One episode, recording everything needed to draw it.

    `expert` is the demonstration controller (docs/deviations.md section 2).
    It cannot be wrapped behind the agent interface: it reads the *true* state
    and the episode's true coefficients, which is the whole point of it, while
    `agent.policy` only ever sees `o_t`.  So it gets a branch instead."""
    import jax
    import jax.numpy as jnp

    from skydreamer.dynamics import CONTROL_DT
    from skydreamer.env import gates_passed, reset, step, unpack_mask
    from skydreamer.render import camera_rotation

    with jax.transfer_guard("allow"):
        st, obs = reset(jax.random.key(seed), cfg)
        if expert is not None:
            from skydreamer import expert as _ex
            from skydreamer import expert_path as _ep
            from skydreamer.params import NOMINAL as _NOM
            _r = _ep.reference(cfg.track, _NOM, margin=0.35,
                               v_cap=expert["v_nom"] * 1.4, iters=1, omega_max=1e9)
            eref = {k: jnp.asarray(_r[k], jnp.float32)
                    for k in ("p", "tangent", "v")}
            eref["ds"] = float(_r["ds"])
            ecarry = _ex.track_init(eref, st.s.p)
        carry = None if expert is not None else agent.init_policy(1)
        target = laps * cfg.track.n_gates
        rec = {k: [] for k in ("p", "q", "v", "mask", "axis", "decoded", "gates", "speed")}
        is_first, crashed = True, False

        for t in range(cfg.max_steps):
            # log/* is diagnostics; the agent asserts it never arrives, and
            # driving the env directly skips the driver that strips it.
            o = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith("log/")}
            o["mask"] = o["mask"].astype(np.uint8)
            o = {k: (v if v.dtype == np.uint8 else v.astype(np.float32)) for k, v in o.items()}
            o.update(
                reward=np.zeros(1, np.float32), is_first=np.array([is_first]),
                is_last=np.zeros(1, bool), is_terminal=np.zeros(1, bool))
            if expert is not None:
                from skydreamer import expert as _ex
                ecarry, uj = _ex.control(
                    ecarry, st.s, st.params, eref, expert["v_nom"],
                    expert["lookahead"], expert["alt_floor"], expert["gains"])
                u, outs = np.asarray(uj, np.float32), {}
            else:
                carry, act, outs = agent.policy(carry, o, mode="eval")
                u = np.clip((np.asarray(act["action"], np.float32)[0] + 1.0) / 2.0, 0.0, 1.0)
            is_first = False

            rec["mask"].append(np.asarray(unpack_mask(obs["mask"])))
            rec["p"].append(np.asarray(st.s.p))
            rec["q"].append(np.asarray(st.s.q))
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


CHASE_RES = 256       # px; the policy's own view is 64, this is just for looking at
CHASE_BACK = 4.0      # m behind the drone
CHASE_UP = 1.3        # m above it
CHASE_PITCH = 14.0    # deg, tilted down towards the drone
GROUND_RANGE = 45.0   # m; beyond this the floor grid is just aliasing


def _rgb(hexstr):
    h = hexstr.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def chase_view(p, q, track, res=CHASE_RES):
    """Render the scene from a chase camera **with the simulator's own ray
    caster** -- the same `render_mask` that produces the image the policy flies
    on, just from a different pose and at a higher resolution.

    Worth being precise about what this can and cannot be.  The simulated world
    contains gates and nothing else: no ground texture, no walls, no drone
    body.  That is not an omission, it is the modelling choice the whole paper
    rests on -- the policy only ever sees a binary gate mask, so a gate mask is
    all the simulator needs to produce.  There is no photoreal scene to film
    because none exists.

    So the gates here are genuine simulator output, and the drone is projected
    in afterwards through the same camera model, which puts its marker exactly
    where this camera would see it.
    """
    import jax
    import jax.numpy as jnp

    from skydreamer.dynamics import euler_to_quat, quat_to_euler
    from skydreamer.render import camera_rotation, intrinsics, pixel_rays, render_mask

    p = np.asarray(p, np.float64)
    with jax.transfer_guard("allow"):
        yaw = float(quat_to_euler(jnp.asarray(q))[2])
        eye = p + np.array(
            [-CHASE_BACK * np.cos(yaw), -CHASE_BACK * np.sin(yaw), -CHASE_UP])
        # Body pitch down, no camera offset: c_e is the body->camera rotation
        # and a free camera has none.
        q_cam = euler_to_quat(
            jnp.float32(0.0), jnp.float32(-np.deg2rad(CHASE_PITCH)), jnp.float32(yaw))
        gates = np.asarray(render_mask(
            jnp.asarray(eye, jnp.float32), q_cam, jnp.zeros(3), track, h=res, w=res))
        R = np.asarray(camera_rotation(q_cam, jnp.zeros(3)))
        rays = np.asarray(pixel_rays(res, res))

    # The floor is real: `track.ground_collision` ends an episode on it, so it
    # is part of the simulated world and belongs in a picture of that world.
    # Intersect every pixel ray with the plane z = 0 (NED, so altitude is -z).
    world = rays @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(world[..., 2] > 1e-6, -eye[2] / world[..., 2], np.inf)
    floor = np.isfinite(t) & (t > 0) & (t < GROUND_RANGE)
    # Only floor pixels have a hit point; zero the rest so the grid arithmetic
    # below stays finite instead of computing inf - inf on the sky.
    hit = eye[None, None, :] + np.where(floor, t, 0.0)[..., None] * world

    # A one-metre grid on the floor gives the perspective and the motion
    # something to read against.  Widen the line with distance so it does not
    # alias into noise near the horizon.
    off = np.minimum(
        np.abs(hit[..., 0] - np.round(hit[..., 0])),
        np.abs(hit[..., 1] - np.round(hit[..., 1])),
    )
    grid = floor & (off < 0.02 * (1.0 + 0.35 * np.clip(t, 0, GROUND_RANGE)))

    img = np.ones((res, res, 3), np.float32) * _rgb(SKY)
    img[floor] = _rgb(FLOOR)
    img[grid] = _rgb(FLOOR_GRID)
    img[gates > 0.5] = _rgb(INK)

    d = R.T @ (p - eye)  # world -> camera-forward (x fwd, y right, z down)
    fx, fy, cx, cy = intrinsics(res, res)
    uv = (cx + fx * d[1] / d[0], cy + fy * d[2] / d[0]) if d[0] > 1e-3 else None
    return img, uv


def _draw_chase(ax, ep, i, track, res=CHASE_RES):
    img, uv = chase_view(ep["p"][i], ep["q"][i], track, res=res)
    ax.imshow(img, interpolation="nearest")
    if uv is not None and 0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]:
        ax.plot(uv[0], uv[1], "o", ms=9, color=TRUTH,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
    ax.set_title("chase camera (simulator render)", color=INK, fontsize=9, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#dddddd")


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


def render(ep, track, out_stem, stride, fps, chase_res=CHASE_RES):
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

    # Subsample the trail that is redrawn every frame.  It grows with `i`, so
    # drawing every simulated step makes the whole render quadratic in episode
    # length -- a 5-lap flight is 3000+ steps and spends minutes on trail
    # segments thinner than a pixel.  `TRAIL_MAX` segments is indistinguishable.
    TRAIL_MAX = 400

    frames = []
    for i in range(0, len(p), stride):
        ts = max(1, (i + 1) // TRAIL_MAX)
        fig = plt.figure(figsize=(15.0, 7.4))
        fig.patch.set_facecolor(GROUND)
        axm = fig.add_subplot(2, 3, 1)
        axc = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        axt = fig.add_subplot(2, 3, 4)
        axs_ = fig.add_subplot(2, 3, 5)
        axi = fig.add_subplot(2, 3, 6)

        axm.imshow(ep["mask"][i], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axm.set_title("what the policy sees (64x64)", color=INK, fontsize=9, pad=6)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_color("#dddddd")

        _draw_chase(axc, ep, i, track, chase_res)

        # third person: gates, the trail so far, and the drone with its camera axis
        _setup3d(ax3, bounds3d)
        _draw_gates3d(ax3, rings)
        if i > 1:
            seg = np.stack([xs[: i + 1: ts], ys[: i + 1: ts], alt[: i + 1: ts]], -1)
            pts = seg.reshape(-1, 1, 3)
            lc3 = Line3DCollection(
                np.concatenate([pts[:-1], pts[1:]], axis=1),
                cmap=cmap, norm=plt.Normalize(0, vmax), linewidth=1.8)
            lc3.set_array(speed[: i + 1: ts])
            ax3.add_collection3d(lc3)
        a = axis[i]
        ax3.plot([xs[i]], [ys[i]], [alt[i]], "o", ms=7,
                 color=cmap(speed[i] / vmax), markeredgecolor="white", markeredgewidth=1.4)
        ax3.plot([xs[i], xs[i] + a[0]], [ys[i], ys[i] + a[1]],
                 [alt[i], alt[i] - a[2]], color=INK, lw=1.4)

        for ax, (h, v), lims, labels, title in (
            (axt, (ys[: i + 1: ts], xs[: i + 1: ts]), (xlim, ylim), ("Y [m]", "X [m]"), "top-down"),
            (axs_, (ys[: i + 1: ts], alt[: i + 1: ts]), (xlim, zlim), ("Y [m]", "altitude [m]"), "side"),
        ):
            _setup(ax, *labels, title)
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal")
            if len(h) > 1:
                pts = np.array([h, v]).T.reshape(-1, 1, 2)
                lc = LineCollection(
                    np.concatenate([pts[:-1], pts[1:]], axis=1),
                    cmap=cmap, norm=plt.Normalize(0, vmax), linewidth=2)
                lc.set_array(speed[: i + 1: ts])
                ax.add_collection(lc)
        _draw_gates(axt, axs_, segs)

        axt.arrow(ys[i], xs[i], a[1] * 0.7, a[0] * 0.7, head_width=0.16, color=INK, lw=1.2)
        axs_.arrow(ys[i], alt[i], a[1] * 0.7, -a[2] * 0.7, head_width=0.16, color=INK, lw=1.2)
        for ax, h, v in ((axt, ys[i], xs[i]), (axs_, ys[i], alt[i])):
            ax.plot(h, v, "o", ms=6, color=cmap(speed[i] / vmax),
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        axi.axis("off")
        lines = [
            f"t            {i * 0.011:6.2f} s",
            f"speed        {speed[i]:6.2f} m/s",
            f"altitude     {alt[i]:6.2f} m",
            f"gates passed {ep['gates'][i]:6d}",
            "",
            f"flight ended {'in a crash' if ep['crashed'] else 'intact'}",
            f"after        {ep['duration']:6.2f} s",
        ]
        axi.text(0.02, 0.97, "\n".join(lines), transform=axi.transAxes,
                 va="top", ha="left", family="monospace", fontsize=10, color=INK)

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

    # Three panels, not two: the paper's top-down and side plots, plus one
    # frame of the simulator's own render so the figure shows the environment
    # and not only a plot of it.  Mid-flight, which is more informative than
    # the moment of the crash.
    fig, (axc, axt, axs_) = plt.subplots(1, 3, figsize=(16, 4.6))
    _draw_chase(axc, ep, len(p) // 2, track)
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
    ap.add_argument("--logdir", type=pathlib.Path,
                    help="a training run to draw.  Not needed with --expert.")
    ap.add_argument("--expert", action="store_true",
                    help="draw the demonstration controller instead of a learned "
                         "policy.  Needs no checkpoint -- it reads the true state.")
    ap.add_argument("--v-nom", type=float, default=2.0, help="--expert commanded speed, m/s")
    ap.add_argument("--lookahead", type=float, default=0.8, help="--expert carrot distance, m")
    ap.add_argument("--alt-floor", type=float, default=1.8, help="--expert altitude floor, m")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--stride", type=int, default=3, help="render every Nth sim step")
    ap.add_argument("--chase-res", type=int, default=CHASE_RES,
                    help="chase-camera resolution, px.  The ray trace is the cost of a "
                         "GIF, and it is quadratic in this; 128 is fine for a contact sheet.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--track", default=None,
        help="fly a track other than the one trained on, e.g. --track big "
             "for a zero-shot test of the paper's flight-plan generalisation claim")
    args = ap.parse_args()

    if args.expert:
        track_name = args.track or "inverted_loop"
        out = args.out or pathlib.Path(f"expert_video_{track_name}")
        out.mkdir(parents=True, exist_ok=True)
        agent = None
    else:
        if args.logdir is None:
            sys.exit("--logdir is required unless --expert is given")
        logdir = args.logdir.expanduser()
        if not (logdir / "config.yaml").exists():
            sys.exit(f"no config.yaml in {logdir}")
        out = args.out or (logdir / "video")
        out.mkdir(parents=True, exist_ok=True)

        import ruamel.yaml as yaml

        task = yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())["task"]
        track_name = args.track or task.split("_", 1)[1]
        if args.track:
            out = args.out or (logdir / f"video_{args.track}")
            out.mkdir(parents=True, exist_ok=True)
            print(f"zero-shot: policy trained on {task.split('_', 1)[1]}, flying {track_name}")

    from skydreamer.embodied_env import SkyDreamer

    env = SkyDreamer(track_name, max_steps=args.laps * 1200, seed=args.seed)
    if not args.expert:
        from evaluate import build_agent
        agent, _ = build_agent(logdir, env)

    expert_cfg = None
    if args.expert:
        from skydreamer import expert as _ex
        expert_cfg = dict(v_nom=args.v_nom, lookahead=args.lookahead,
                          alt_floor=args.alt_floor,
                          gains=_ex.Gains(k_vel=3.5, max_tilt_deg=35.0))
        print(f"demonstration controller: v_nom={args.v_nom} m/s on {track_name}")

    # Pull the track onto the host once, here, rather than letting the drawing
    # code call np.asarray on device arrays.  DreamerV3 sets
    # jax_transfer_guard='disallow' process-wide; on CPU a device-to-host read
    # is free and the guard never fires, so this is invisible until it runs on
    # a GPU -- which is exactly how it got here.  Everything downstream is
    # plain numpy and matplotlib, so it has no business touching a device.
    import jax

    with jax.transfer_guard("allow"):
        track = jax.tree.map(np.asarray, env.cfg.track)

    print(f"flying {args.episodes} episodes on {track_name}, {args.laps} laps each", flush=True)

    for i in range(args.episodes):
        ep = fly(agent, env.cfg, args.seed + i, args.laps, expert=expert_cfg)
        stem = out / f"flight_{i}"
        png = still(ep, track, out / f"figure_{i}.png")
        gif = render(ep, track, stem, args.stride, args.fps, args.chase_res)
        print(f"  episode {i}: {ep['gates'][-1]} gates, {ep['duration']:.2f} s, "
              f"peak {ep['speed'].max():.1f} m/s"
              f"{' (crashed)' if ep['crashed'] else ''}", flush=True)
        print(f"    {gif.name}  {png.name}", flush=True)

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
