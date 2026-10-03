"""Phase B: fly the demonstration controller and write the good episodes into a
DreamerV3 replay directory.

The format is not reproduced by hand.  This builds the environment and the
replay with DreamerV3's own factories (`main.make_env`, `main.make_replay`) and
drives them with `embodied.Driver`, so what lands on disk is byte-identical in
structure to what a training run writes.  The only departure is *which* steps
get added: the driver's transitions are buffered for the length of an episode
and handed to `replay.add` only if that episode passed enough gates.

Why filter rather than make the controller reliable: under the full Table III
disturbances the controller completes 5 laps about 5% of the time, and the
remaining failures are a flat per-gate hazard rather than a fixable trajectory
bug (docs/deviations.md).  It does not matter.  A scan runs 256 episodes of
6000 steps in six seconds, so the demonstrations that matter are a few minutes
of CPU away, and a demonstration that crashed teaches the critic the thing it
already believes.

    python scripts/collect_demos.py --config <run>/config.yaml --out demos \\
        --episodes 4000 --min-gates 15

The resulting `demos/replay/` is seeded into training by pointing a run's
replay directory at it, or by copying its chunks in before the run starts.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

DEMO_MANIFEST = ".demo-chunks"

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "dreamerv3"))


def _context_zeros(config, wrapped_env, np):
    """Zero-filled values for whatever `replay_context` adds to the agent's
    expected key set.  Built from a constructed Agent so a config change in
    `dyn.rssm` cannot silently desynchronise them."""
    import elements
    import jax
    from dreamerv3.agent import Agent

    # Constructing an Agent turns on `jax_transfer_guard='disallow'` for the
    # whole process, and everything after this point is a host-driven rollout
    # that transfers constantly.  Put it back.
    guard = jax.config.jax_transfer_guard

    obs = {k: v for k, v in wrapped_env.obs_space.items()
           if not k.startswith("log/")}
    act = {k: v for k, v in wrapped_env.act_space.items() if k != "reset"}
    agent = Agent(obs, act, elements.Config(
        **config.agent, logdir=str(config.logdir), seed=config.seed,
        jax=config.jax, batch_size=config.batch_size,
        batch_length=config.batch_length, replay_context=config.replay_context,
        report_length=config.report_length, replica=config.replica,
        replicas=config.replicas))
    skip = set(obs) | set(act) | {"consec", "stepid"}
    out = {k: np.zeros(v.shape, v.dtype)
           for k, v in agent.spaces.items() if k not in skip}
    jax.config.update("jax_transfer_guard", guard)
    return out


def _recorder(cfg, R, gains, args, jnp):
    """Jitted, vmapped rollout that records exactly what the driver would have
    written: each observation paired with the action chosen *from* it.

    This exists because the Driver is 18 s per 2000-step episode -- every step
    crosses the host/device boundary -- which is two and a half hours for 500
    demonstrations.  The same rollout vmapped and jitted is seconds.  Only the
    data path is skipped; the chunks are still written by `replay.add`, so the
    on-disk format is still theirs and not a reimplementation.
    """
    import jax

    from skydreamer import env as E
    from skydreamer import expert as ex
    from skydreamer import track as trk

    def one(key):
        st, obs = E.reset(key, cfg)
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
        init = (st, obs, jnp.float32(0.0), jnp.bool_(True), jnp.bool_(False),
                jnp.bool_(False), ex.track_init(R, st.s.p), jnp.bool_(False))

        def body(carry, _):
            st, obs, rew, first, last, term, c, done = carry
            c2, u = ex.control(c, st.s, st.params, R, args.v_nom,
                               args.lookahead, args.alt_floor, gains)
            out = dict(obs, action=2.0 * u - 1.0, reward=rew, is_first=first,
                       is_last=last, is_terminal=term,
                       gates=(st.plane - st.plane0) // trk.PLANES_PER_GATE,
                       cause=st.term_cause)
            st2, obs2, rew2, term2, trunc2 = E.step(st, u, cfg)
            obs2 = {k: v for k, v in obs2.items() if not k.startswith("log/")}
            fin = term2 | trunc2
            nxt = (st2, obs2, rew2, jnp.bool_(False), fin, term2, c2, done | last)
            nxt = jax.tree.map(lambda a, b: jnp.where(done, b, a), nxt, carry)
            return nxt, out

        _, o = jax.lax.scan(body, init, None, length=cfg.max_steps + 1)
        return o

    return jax.jit(jax.vmap(one))


def _screener(cfg, R, gains, args, jnp):
    """Jitted, vmapped scorer: keys in, gates passed out."""
    import jax

    from skydreamer import env as E
    from skydreamer import expert as ex
    from skydreamer import track as trk

    def one(key):
        st, _ = E.reset(key, cfg)
        c = ex.track_init(R, st.s.p)

        def body(carry, _):
            st, c, done = carry
            c2, u = ex.control(c, st.s, st.params, R, args.v_nom,
                               args.lookahead, args.alt_floor, gains)
            st2, _, _, term, trunc = E.step(st, u, cfg)
            st2 = jax.tree.map(lambda a, b: jnp.where(done, b, a), st2, st)
            return (st2, c2, done | term | trunc), dict(
                done=done | term | trunc, cause=st2.term_cause,
                gates=(st2.plane - st2.plane0) // trk.PLANES_PER_GATE)

        _, o = jax.lax.scan(body, (st, c, jnp.bool_(False)), None,
                            length=cfg.max_steps)
        d = o["done"]
        i = jnp.minimum(jnp.argmax(d), d.shape[0] - 1)
        return o["gates"][i], o["cause"][i]

    return jax.jit(jax.vmap(one))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path,
                    help="config.yaml from an existing run.  Omit it on a fresh "
                         "checkout: the same config is then built from "
                         "dreamerv3's configs.yaml the way scripts/train.py "
                         "does, so a server that has only just cloned the repo "
                         "can collect demonstrations before its first run.")
    ap.add_argument("--preset", default="small", choices=("small", "big"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("demos"))
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--min-gates", type=int, default=5,
                    help="floor on gates passed.  Note that a *training* episode "
                         "is truncated at config.env.skydreamer.max_steps (2000 "
                         "steps = 22.2 s), and at this controller's pace five "
                         "laps does not fit in that -- about 7 gates do.")
    ap.add_argument("--allow-crashed", action="store_true",
                    help="also keep episodes that ended in a crash.  Off by "
                         "default: an episode that reaches the step limit ends "
                         "with is_terminal False, so the critic bootstraps from "
                         "it instead of seeing a zero, which is the whole point.")
    ap.add_argument("--max-keep", type=int, default=0,
                    help="stop once this many episodes have been kept (0 = no limit)")
    ap.add_argument("--v-nom", type=float, default=2.5)
    ap.add_argument("--lookahead", type=float, default=0.8)
    ap.add_argument("--alt-floor", type=float, default=1.8)
    ap.add_argument("--k-vel", type=float, default=3.5)
    ap.add_argument("--max-tilt", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import elements
    import embodied
    import jax.numpy as jnp
    import ruamel.yaml as yaml
    from dreamerv3 import main as dv3main

    from skydreamer import expert as ex
    from skydreamer import expert_path as ep
    from skydreamer import track as trk
    from skydreamer.params import NOMINAL

    if args.config:
        config = elements.Config(yaml.YAML(typ="safe").load(args.config.read_text()))
    else:
        names = ["defaults",
                 "skydreamer" if args.preset == "small" else "skydreamer_big",
                 "size12m"]
        folder = ROOT / "third_party" / "dreamerv3" / "dreamerv3"
        allcfg = yaml.YAML(typ="safe").load((folder / "configs.yaml").read_text())
        config = elements.Config(allcfg["defaults"])
        for name in names[1:]:
            config = config.update(allcfg[name])
        print(f"config built from {' + '.join(names)}")
    args.out.mkdir(parents=True, exist_ok=True)
    # make_replay puts the directory under config.logdir, so point that here.
    config = config.update({"logdir": str(args.out.resolve())})

    replay = dv3main.make_replay(config, "replay", mode="train")
    driver = embodied.Driver([lambda: dv3main.make_env(config, 0)], parallel=False)

    # Reach through the wrappers to the simulator itself.  The controller is
    # privileged by design -- it reads the true state and the episode's true
    # coefficients -- and `policy(carry, obs)` only ever sees `o_t`.
    # Unwrap to the simulator itself.  `hasattr` is useless here: embodied's
    # wrappers forward every unknown attribute inwards, so the outermost one
    # answers to `_state` as readily as the env that owns it.
    env = driver.envs[0]
    while hasattr(env, "env"):
        env = env.env
    env._build()   # `_cfg` and `_rng` do not exist until the first build
    track_name = config.task.split("_", 1)[1]
    track = {"inverted_loop": trk.inverted_loop, "big": trk.big_track,
             "ladder_inverted_loop": trk.ladder_inverted_loop}[track_name]()
    n_gates = track.n_gates

    r = ep.reference(track, NOMINAL, margin=0.35, v_cap=args.v_nom * 1.4,
                     iters=1, omega_max=1e9)
    R = {k: jnp.asarray(r[k], jnp.float32) for k in ("p", "tangent", "v")}
    R["ds"] = float(r["ds"])
    gains = ex.Gains(k_vel=args.k_vel, max_tilt_deg=args.max_tilt)

    # The agent also stores its RSSM carry in replay when `replay_context > 0`
    # (`Agent.policy` puts enc/dyn/dec entries in `outs`), and `Agent.train`
    # asserts the batch carries exactly `self.spaces`.  A controller has no such
    # carry, so fill those keys with zeros: they are a cache that lets a sampled
    # sequence skip re-deriving the latent state for its first `replay_context`
    # steps, the RSSM re-converges over those steps without it, and the agent
    # overwrites them the first time it trains on the chunk.  The shapes come
    # from the agent rather than from the config so they cannot drift.
    extra = _context_zeros(config, driver.envs[0], np)
    if extra:
        print(f"  filling replay-context keys with zeros: {sorted(extra)}")

    def policy(carry, obs, **kw):
        st = env._state
        if carry is None or bool(np.asarray(obs["is_first"])[0]):
            carry = ex.track_init(R, st.s.p)
        carry, u = ex.control(carry, st.s, st.params, R, args.v_nom,
                              args.lookahead, args.alt_floor, gains)
        # NormalizeAction maps [-1, 1] onto the action space, so the policy
        # side of that wrapper is where the demonstration has to live.
        a = np.asarray(2.0 * u - 1.0, np.float32)[None]
        return carry, {"action": a}, {}

    # --- screen first.  Replaying an episode through the real Driver costs
    # seconds; the jitted, vmapped rollout scores 256 of them in six.  Only
    # about one episode in twenty is worth keeping, so screen with the fast
    # path and hand the Driver only the seeds that pay.
    #
    # The two paths have to agree on which episode a seed names.  The wrapper
    # holds `_rng` and draws each episode as `split(_rng)`, so setting
    # `_rng = key(s)` before a reset makes that episode `split(key(s))[1]` --
    # which is exactly the key the screen evaluates.
    import jax
    buf: list[dict] = []
    stats = collections.Counter()
    hist = collections.Counter()
    kept = 0

    def on_step(tran, worker):
        nonlocal buf, kept
        buf.append({k: np.asarray(v) for k, v in tran.items()})
        if not tran["is_last"]:
            return
        # The simulator's own accounting, read at the terminal step.  The
        # wrapper resets at the *start* of the next call, so `_state` here is
        # still the state the episode ended in.
        gates = int((env._state.plane - env._state.plane0) // trk.PLANES_PER_GATE)
        hist[min(gates, 30)] += 1
        stats["episodes"] += 1
        crashed = int(np.asarray(env._state.term_cause)) != 0
        if gates >= args.min_gates and (args.allow_crashed or not crashed):
            for t in buf:
                replay.add(t, worker)
            kept += 1
            stats["kept"] += 1
            stats["steps"] += len(buf)
        buf = []

    driver.on_step(on_step)

    def episode_keys(seeds):
        """The key the wrapper will actually draw for a seed, `split(key(s))[1]`.
        Built with a list comprehension, not vmap: `jax.random.key` needs a
        concrete seed and a traced one fails."""
        return jnp.stack([jax.random.split(jax.random.key(int(s)))[1]
                          for s in seeds])

    print(f"track {track_name}, {n_gates} gates/lap, keeping episodes with "
          f">= {args.min_gates} gates")
    fast = _screener(env._cfg, R, gains, args, jnp)

    # Screen first, then replay.  Keeping these as two phases matters: the
    # loop that interleaved them stopped as soon as the seed counter reached
    # `--episodes` and never replayed anything.
    good, screened = [], 0
    while screened < args.episodes:
        idx = np.arange(screened, min(screened + 512, args.episodes))
        g, c = fast(episode_keys(idx))
        ok = np.asarray(g) >= args.min_gates
        if not args.allow_crashed:
            ok &= np.asarray(c) == 0          # 0 = reached the step limit
        good += [int(i) for i, k in zip(idx, ok) if k]
        screened += len(idx)
        print(f"  screened {screened}/{args.episodes}, {len(good)} worth "
              f"replaying ({100 * len(good) / screened:.1f}%)", flush=True)
    if args.max_keep:
        good = good[: args.max_keep]

    rec = _recorder(env._cfg, R, gains, args, jnp)
    print(f"\nrecording {len(good)} episodes")
    CH = 32                                   # seeds per vmapped batch
    for base in range(0, len(good), CH):
        seeds = good[base: base + CH]
        o = jax.tree.map(np.asarray, rec(episode_keys(seeds)))
        for e in range(len(seeds)):
            last = np.asarray(o["is_last"][e])
            n = int(np.argmax(last)) + 1 if last.any() else len(last)
            gates = int(o["gates"][e][n - 1])
            hist[min(gates, 30)] += 1
            stats["episodes"] += 1
            for t in range(n):
                step = {k: v[e][t] for k, v in o.items()
                        if k not in ("gates", "cause")}
                step.update(extra)
                replay.add(step, 0)
            stats["kept"] += 1
            stats["steps"] += n
        print(f"  {stats['kept']:5d} kept, {stats['steps']:8d} steps", flush=True)

    # Wait for the chunk writes.  They are submitted to a thread pool and
    # `save_wait` is off by default, so the manifest below would glob an empty
    # directory and protect nothing.
    replay.save_wait = True
    replay.save()
    # Name the chunks so `prune_replay.py` can refuse to delete them.  It drops
    # the oldest files first, and demonstrations are written before training
    # starts, so without this they are exactly what it reclaims.
    rdir = args.out / "replay"
    names = sorted(x.name for x in rdir.glob("*.npz"))
    (rdir / DEMO_MANIFEST).write_text("\n".join(names) + "\n")

    print(f"\nkept {stats['kept']} / {stats['episodes']} replayed episodes "
          f"({100 * stats['kept'] / max(stats['episodes'], 1):.1f}%), "
          f"{stats['steps']} transitions")
    print("gates passed:")
    for g in sorted(hist):
        print(f"  {g:3d}: {hist[g]:5d}")
    print(f"\nwrote {rdir}  ({len(names)} chunks, manifest {DEMO_MANIFEST})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
