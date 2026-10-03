"""Fly the demonstration controller, score every episode, and keep the good ones.

Two jobs, one rollout:

  --scan     how often the controller completes N laps, as a histogram
  --render   GIFs of representative episodes, bucketed by gates passed
  --collect  write successful episodes to a DreamerV3 replay directory

The rollout is jitted and vmapped over seeds, which is the whole point: driving
the environment from a Python loop costs minutes per 5-lap episode, and this
needs thousands of them.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from skydreamer import env as E
from skydreamer import expert as ex
from skydreamer import expert_path as ep
from skydreamer import track as trk
from skydreamer.params import NOMINAL
from skydreamer.render import camera_rotation

TRACKS = {"inverted_loop": trk.inverted_loop, "big": trk.big_track,
          "ladder_inverted_loop": trk.ladder_inverted_loop}


def reference(track, v_nom, margin=0.35):
    r = ep.reference(track, NOMINAL, margin=margin, v_cap=v_nom * 1.4,
                     iters=1, omega_max=1e9)
    R = {k: jnp.asarray(r[k], jnp.float32) for k in ("p", "tangent", "v")}
    R["ds"] = float(r["ds"])
    return R


def rollout(cfg, R, gains, v_nom, lookahead, alt_floor, steps, record=False):
    """One episode from a key.  `record` also keeps what the drawing needs."""
    def one(key):
        st, _ = E.reset(key, cfg)
        c = ex.track_init(R, st.s.p)

        def body(carry, _):
            st, c, done = carry
            c2, u = ex.control(c, st.s, st.params, R, v_nom, lookahead, alt_floor, gains)
            st2, obs, rew, term, trunc = E.step(st, u, cfg)
            # Freeze on the *previous* done so the terminating state, which
            # carries the cause, is the one that survives.
            st2 = jax.tree.map(lambda a, b: jnp.where(done, b, a), st2, st)
            out = dict(gates=(st2.plane - st2.plane0) // trk.PLANES_PER_GATE,
                       done=done | term | trunc, cause=st2.term_cause,
                       reward=jnp.where(done, 0.0, rew))
            if record:
                out.update(p=st.s.p, q=st.s.q, v=st.s.v, mask=st.mask_buf[0],
                           action=u, c_e=st.c_e)
            return (st2, c2, done | term | trunc), out

        _, o = jax.lax.scan(body, (st, c, jnp.bool_(False)), None, length=steps)
        return o
    return jax.jit(jax.vmap(one))


def score(o):
    """Gates passed and termination cause at the end of each episode."""
    d = np.asarray(o["done"])
    first = np.argmax(d, 1)
    alive = np.where(d.any(1), first, d.shape[1])
    i = np.minimum(first, d.shape[1] - 1)
    n = np.arange(len(d))
    return (np.asarray(o["gates"])[n, i], np.asarray(o["cause"])[n, i], alive)


BUCKETS = [("15+ (5랩 완주)", 15, 10 ** 9), ("10-14", 10, 15),
           ("5-9", 5, 10), ("1-4", 1, 5), ("0 (실패)", 0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="inverted_loop")
    ap.add_argument("--v-nom", type=float, default=2.5)
    ap.add_argument("--lookahead", type=float, default=0.8)
    ap.add_argument("--alt-floor", type=float, default=1.8)
    ap.add_argument("--k-vel", type=float, default=3.5)
    ap.add_argument("--max-tilt", type=float, default=35.0)
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--per-bucket", type=int, default=5)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--chase-res", type=int, default=128)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("expert_demo"))
    args = ap.parse_args()

    track = TRACKS[args.track]()
    steps = args.laps * 1200
    cfg = E.EnvConfig(track=track, max_steps=steps)
    R = reference(track, args.v_nom)
    gains = ex.Gains(k_vel=args.k_vel, max_tilt_deg=args.max_tilt)
    need = args.laps * track.n_gates

    fast = rollout(cfg, R, gains, args.v_nom, args.lookahead, args.alt_floor, steps)

    # --- scan.  Seeds are `jax.random.key(i)` so a seed names an episode.
    gates = np.zeros(args.seeds, np.int32)
    causes = np.zeros(args.seeds, np.int32)
    for s in range(0, args.seeds, args.batch):
        idx = np.arange(s, min(s + args.batch, args.seeds))
        o = fast(jax.vmap(jax.random.key)(jnp.asarray(idx)))
        g, c, _ = score(o)
        gates[idx], causes[idx] = g, c
        print(f"  scanned {idx[-1] + 1}/{args.seeds}", flush=True)

    print(f"\n{args.track}, {args.laps} laps = {need} gates, v_nom {args.v_nom}")
    print(f"{'bucket':>16} {'n':>5} {'%':>7}   seeds")
    picks = {}
    for name, lo, hi in BUCKETS:
        sel = np.where((gates >= lo) & (gates < hi))[0]
        picks[name] = sel[: args.per_bucket]
        print(f"{name:>16} {len(sel):5d} {100 * len(sel) / args.seeds:6.1f}%   "
              f"{sel[:args.per_bucket].tolist()}")
    best = int(np.argmax(gates))
    print(f"\nbest: seed {best} -> {gates[best]} gates")

    if not args.render:
        return 0

    # --- render.  Re-fly only the chosen seeds, this time recording.
    import visualize as V

    rec = rollout(cfg, R, gains, args.v_nom, args.lookahead, args.alt_floor,
                  steps, record=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with jax.transfer_guard("allow"):
        track_np = jax.tree.map(np.asarray, cfg.track)

    jobs = [(f"{name.split()[0]}_seed{int(s)}", int(s))
            for name, sel in picks.items() for s in sel]
    jobs.append((f"best_seed{best}", best))
    for tag, seed in jobs:
        o = rec(jax.vmap(jax.random.key)(jnp.asarray([seed])))
        o = jax.tree.map(lambda x: np.asarray(x)[0], o)
        d = o["done"]
        n = int(np.argmax(d)) if d.any() else len(d)
        n = max(n, 2)
        eps = dict(p=o["p"][:n], q=o["q"][:n], v=o["v"][:n],
                   mask=o["mask"][:n].astype(np.float32),
                   speed=np.linalg.norm(o["v"][:n], axis=1),
                   gates=o["gates"][:n], decoded=np.zeros((0, 3)),
                   crashed=bool(o["cause"][n - 1] != 0),
                   duration=n / 90.0)
        eps["axis"] = np.stack([
            np.asarray(camera_rotation(jnp.asarray(q), jnp.asarray(c)) @ jnp.array([1.0, 0, 0]))
            for q, c in zip(o["q"][:n], o["c_e"][:n])])
        stem = args.out / tag
        png = V.still(eps, track_np, args.out / f"{tag}.png")
        gif = V.render(eps, track_np, stem, args.stride, 30, args.chase_res)
        print(f"  {tag}: {int(o['gates'][n - 1])} gates, {n / 90:.2f} s, "
              f"{'crashed' if eps['crashed'] else 'ok'}  ->  {gif.name}", flush=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
