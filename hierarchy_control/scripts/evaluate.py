#!/usr/bin/env python3
"""Score a CTBA checkpoint in simulation on the real test: stage `ce` (the
whole airframe against the real opening), every flight launched from the floor
before G1, the plant randomized as in training.

Each flight is also checked after the fact with skydreamer.hardware.check_flight
on our CAD footprint (props, arms, shell, stands, camera): every point that
crosses a gate plane must do so inside the opening.  Success = the simulator's
own verdict AND that check, for `--laps` laps.

    python scripts/evaluate.py --logdir <run> [--episodes 200] [--laps 1] [--track team44up]
                               [--stage ceg]   (ceg: masks through StochGAN, $STOCHGAN_WEIGHTS)
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
SKYDREAMER = pathlib.Path(os.environ.get("SKYDREAMER", HERE.parent)).expanduser()
sys.path.insert(0, str(SKYDREAMER / "third_party" / "dreamerv3"))
sys.path.insert(0, str(HERE))

CAUSES = {0: "survived", 1: "hit a gate", 2: "hit the floor", 3: "diverged", 4: "wall or ceiling"}


def _sd_eval():
    spec = importlib.util.spec_from_file_location("sd_evaluate", SKYDREAMER / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rollout(agent, ccfg, episodes, laps, seed, chunk=64):
    import jax
    import jax.numpy as jnp

    from ctba import airframe, env as E, params as P
    from skydreamer import hardware

    reset_b, step_b = E.batched(ccfg)
    tr = ccfg.base.track
    n_gates = tr.n_gates
    with jax.transfer_guard("allow"):
        pos, yaw, vis = np.asarray(tr.pos), np.asarray(tr.yaw), np.asarray(tr.visible)
    gates_def = [(pos[g], float(yaw[g]), bool(vis[g])) for g in range(n_gates)]
    pts = airframe.footprint()
    with jax.transfer_guard("allow"):
        inner_half = float(np.asarray(tr.inner)[0]) / 2
    out = []
    with jax.transfer_guard("allow"):
        done_total = 0
        while done_total < episodes:
            b = min(chunk, episodes - done_total)
            st, obs = reset_b(jax.random.split(jax.random.key(seed + done_total), b))
            carry = agent.init_policy(b)
            alive, finished = np.ones(b, bool), np.zeros(b, bool)
            steps = np.zeros(b, int)
            vmax = np.zeros(b)
            first = np.ones(b, bool)
            tp, tq = [], []
            for t in range(ccfg.base.max_steps):
                o = {k: (np.asarray(v).astype(np.uint8) if k == "mask" else np.asarray(v).astype(np.float32))
                     for k, v in obs.items() if not k.startswith("log/")}
                o.update(reward=np.zeros(b, np.float32), is_first=first.copy(),
                         is_last=np.zeros(b, bool), is_terminal=np.zeros(b, bool))
                carry, act, _ = agent.policy(carry, o, mode="eval")
                first[:] = False
                a = jnp.asarray(np.asarray(act["action"], np.float32))
                live = alive & ~finished
                st2, obs2, _, term, _ = step_b(st, a)
                hold = jnp.asarray(~live)
                sel = lambda x, y: jnp.where(hold.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)  # noqa: E731
                st = jax.tree.map(sel, st, st2)
                obs = jax.tree.map(sel, obs, obs2)
                p, q, v = np.asarray(st.s.p), np.asarray(st.s.q), np.asarray(st.s.v)
                tp.append(p); tq.append(q)
                gates = np.asarray(E.gates_passed(st))
                terminated = np.asarray(term) & live
                vmax = np.where(live, np.maximum(vmax, np.linalg.norm(v, axis=-1)), vmax)
                steps = np.where(live, t + 1, steps)
                finished |= live & ~terminated & (gates >= laps * n_gates)
                alive &= ~terminated
                if (finished | ~alive).all():
                    break
            causes = np.asarray(st.term_cause)
            gates = np.asarray(E.gates_passed(st))
            tp, tq = np.stack(tp, 1), np.stack(tq, 1)
            for i in range(b):
                ok_phys, ev = hardware.check_flight(tp[i], tq[i], gates_def, int(steps[i]), P.CONTROL_DT,
                                                    opening=inner_half, points=pts)
                bad = [e for e in ev if not e[4]]
                out.append(dict(success=bool(finished[i]) and ok_phys, sim_success=bool(finished[i]),
                                cause=int(causes[i]), gates=int(gates[i]), duration=float(steps[i] * P.CONTROL_DT),
                                vmax=float(vmax[i]),
                                phys_fail=None if not bad else dict(gate=bad[0][1], off=bad[0][3])))
            done_total += b
            print(f"  {done_total}/{episodes}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=pathlib.Path, required=True)
    ap.add_argument("--track", default="team44up")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage", default="ce", choices=["ce", "ceg"],
                    help="ce: rendered masks (the curriculum's test); ceg: the same through StochGAN")
    args = ap.parse_args()

    import jax
    from ctba import tasks
    from ctba.embodied import Ctba

    task = f"{args.stage}_{args.track}"
    env = Ctba(task)
    agent, cfg = _sd_eval().build_agent(args.logdir, env)
    # DreamerV3 pins jax_platforms to its `jax.platform` (cuda on the server), so the
    # cpu backend may not exist once the agent is built; the config is cheap anywhere.
    try:
        dev = jax.devices("cpu")[0]
    except RuntimeError:
        dev = jax.devices()[0]
    with jax.default_device(dev), jax.transfer_guard("allow"):
        ccfg = tasks.make_config(task, max_steps=cfg.env.ctba.max_steps)
        ccfg = ccfg._replace(base=ccfg.base._replace(train_fraction=0.0))   # every flight from the floor
    recs = rollout(agent, ccfg, args.episodes, args.laps, args.seed)

    n = len(recs)
    succ = np.mean([r["success"] for r in recs])
    sim = np.mean([r["sim_success"] for r in recs])
    causes = collections.Counter(CAUSES[r["cause"]] for r in recs if not r["sim_success"])
    phys = collections.Counter(f"G{r['phys_fail']['gate'] + 1}" for r in recs if r["phys_fail"])
    se = np.sqrt(succ * (1 - succ) / n)
    lines = [
        f"CTBA evaluation: {args.logdir}  task {task}  {args.laps} lap(s)  {n} flights (seed {args.seed})",
        f"  success (sim and airframe check): {100 * succ:.1f}% +- {100 * 1.96 * se:.1f}",
        f"  success in the simulator alone:   {100 * sim:.1f}%",
        f"  sim failures: {dict(causes)}",
        f"  airframe outside the opening at: {dict(phys)}",
        f"  gates passed mean {np.mean([r['gates'] for r in recs]):.1f}, top speed p50 "
        f"{np.median([r['vmax'] for r in recs]):.1f} m/s",
    ]
    text = "\n".join(lines)
    print(text)
    name = f"evaluation_ctba_{args.laps}lap{'' if args.stage == 'ce' else '_' + args.stage}{args.tag}"
    (args.logdir / f"{name}.txt").write_text(text + "\n")
    (args.logdir / f"{name}.json").write_text(json.dumps(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
