#!/usr/bin/env python3
"""Fly the CTBA demonstration controller (ctba.expert) and write the clean,
long episodes into a DreamerV3 replay directory (curriculum stage c1).

As skydreamer's scripts/collect_demos.py: screen many seeds with a jitted,
vmapped rollout, then record the good ones with the same rollout and hand each
step to DreamerV3's own `replay.add`, so the chunks on disk are DreamerV3's
format.  Each stored step is the observation the policy would have seen, the
action chosen from it (in the policy's [-1, 1]^4 space), and the reward and
flags that arrived with that observation.

    python scripts/collect_demos.py --out demos_ctba [--track team44up]
        [--episodes 4000] [--min-gates 8] [--v-nom 2.5]
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
SKYDREAMER = pathlib.Path(os.environ.get("SKYDREAMER", HERE.parent)).expanduser()
DV3 = SKYDREAMER / "third_party" / "dreamerv3"
sys.path.insert(0, str(DV3))
sys.path.insert(0, str(HERE))

DEMO_MANIFEST = ".demo-chunks"


def recorder(ccfg, ref, args):
    import jax
    import jax.numpy as jnp

    from ctba import env as E
    from ctba import expert as X

    def one(key):
        st, obs = E.reset(key, ccfg)
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
        init = (st, obs, jnp.float32(0.0), jnp.bool_(True), jnp.bool_(False), jnp.bool_(False),
                X.init(ref, st.s.p), jnp.bool_(False))

        def body(carry, _):
            st, obs, rew, first, last, term, c, done = carry
            c2, a = X.control(c, st, ref, args.v_nom, args.lookahead, args.alt_floor, X.Gains())
            out = dict(obs, action=a, reward=rew, is_first=first, is_last=last, is_terminal=term,
                       gates=E.gates_passed(st), cause=st.term_cause)
            st2, obs2, rew2, term2, trunc2 = E.step(st, a, ccfg)
            obs2 = {k: v for k, v in obs2.items() if not k.startswith("log/")}
            nxt = (st2, obs2, rew2, jnp.bool_(False), term2 | trunc2, term2, c2, done | last)
            nxt = jax.tree.map(lambda x, y: jnp.where(done, y, x), nxt, carry)
            return nxt, out

        _, o = jax.lax.scan(body, init, None, length=ccfg.base.max_steps + 1)
        return o

    return jax.jit(jax.vmap(one))


def context_zeros(config):
    if not config.replay_context:
        return {}
    r = config.agent.dyn.rssm
    return {"dyn/deter": np.zeros((r.deter,), np.float32),
            "dyn/stoch": np.zeros((r.stoch, r.classes), np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("demos_ctba"))
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--stage", default="c1")
    ap.add_argument("--episodes", type=int, default=4000, help="seeds to screen")
    ap.add_argument("--min-gates", type=int, default=8, help="keep episodes passing at least this many")
    ap.add_argument("--max-keep", type=int, default=0)
    ap.add_argument("--v-nom", type=float, default=2.5)
    ap.add_argument("--lookahead", type=float, default=0.8)
    ap.add_argument("--alt-floor", type=float, default=0.6)
    ap.add_argument("--size", default="size12m")
    args = ap.parse_args()

    import elements
    import jax
    import jax.numpy as jnp
    import ruamel.yaml as yaml
    from dreamerv3 import main as dv3main

    from ctba import expert as X
    from ctba import tasks

    allcfg = yaml.YAML(typ="safe").load((DV3 / "dreamerv3" / "configs.yaml").read_text())
    if "ctba" not in allcfg:
        sys.exit("dreamerv3 has no `ctba` config: run scripts/patch_dreamerv3.py first")
    config = elements.Config(allcfg["defaults"]).update(allcfg["ctba"]).update(allcfg[args.size])
    args.out.mkdir(parents=True, exist_ok=True)
    task = f"{args.stage}_{args.track}"
    config = config.update({"logdir": str(args.out.resolve()), "task": f"ctba_{task}"})
    replay = dv3main.make_replay(config, "replay", mode="train")
    extra = context_zeros(config)

    with jax.default_device(jax.devices("cpu")[0]):
        ccfg = tasks.make_config(task, max_steps=config.env.ctba.max_steps)
        ref = X.reference(ccfg.base.track, args.v_nom)
        screen = X.screener(ccfg, ref, args.v_nom, args.lookahead, args.alt_floor, X.Gains())
        rec = recorder(ccfg, ref, args)

        def keys(seeds):
            # the key the embodied wrapper draws for seed s: split(key(s))[1]
            return jnp.stack([jax.random.split(jax.random.key(int(s)))[1] for s in seeds])

        good, done = [], 0
        while done < args.episodes:
            idx = np.arange(done, min(done + 512, args.episodes))
            g, c = screen(keys(idx))
            ok = (np.asarray(g) >= args.min_gates) & (np.asarray(c) == 0)
            good += [int(i) for i, k in zip(idx, ok) if k]
            done += len(idx)
            print(f"  screened {done}/{args.episodes}: {len(good)} kept ({100 * len(good) / done:.1f}%)", flush=True)
        if args.max_keep:
            good = good[: args.max_keep]

        stats = collections.Counter()
        for b in range(0, len(good), 32):
            seeds = good[b: b + 32]
            o = jax.tree.map(np.asarray, rec(keys(seeds)))
            for e in range(len(seeds)):
                last = o["is_last"][e]
                n = int(np.argmax(last)) + 1 if last.any() else len(last)
                for t in range(n):
                    step = {k: v[e][t] for k, v in o.items() if k not in ("gates", "cause")}
                    step.update(extra)
                    replay.add(step, 0)
                stats["episodes"] += 1
                stats["steps"] += n
            print(f"  recorded {stats['episodes']} episodes, {stats['steps']} steps", flush=True)

    replay.save_wait = True
    replay.save()
    rdir = args.out / "replay"
    names = sorted(x.name for x in rdir.glob("*.npz"))
    (rdir / DEMO_MANIFEST).write_text("\n".join(names) + "\n")
    print(f"wrote {rdir}: {stats['episodes']} episodes, {stats['steps']} steps, {len(names)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
