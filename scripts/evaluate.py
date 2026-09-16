#!/usr/bin/env python3
"""Evaluate a trained SkyDreamer policy in simulation and score it against the paper.

Reports, per track:

  success rate        fraction of episodes completing `--laps` laps without a
                      crash.  Paper Table IV: 100% over 25 laps per track.
  lap time            paper Table IV splits lap 1 from laps 2-5, so we do too.
  gate error          distance from the gate centre at the crossing plane
                      (the ADR literature's "mean gate error").
  gate margin         smallest clearance to the gate edge over the flight --
                      how close it came to hitting something.
  max speed / accel   paper: 13 m/s and 6 g on the small tracks.
  decode error        |decoded state - ground truth|, which is the paper's
                      interpretability claim (Figures 4 and 5).  Needs
                      `--agent.decode_in_policy True`, which this script sets.

Episodes run batched in the simulator, so a few hundred is cheap -- far more
than the 25 real laps the paper could fly, and in line with the 1000-rollout
protocol used by Ferede et al. (arXiv:2504.21586).

Usage
  python scripts/evaluate.py --logdir ~/logdir/sd/run1
  python scripts/evaluate.py --logdir ~/logdir/sd/run1 --episodes 200 --laps 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DV3 = ROOT / "third_party" / "dreamerv3"
if DV3.exists():
    sys.path.insert(0, str(DV3))

# Paper Table IV, for the printed comparison.
PAPER = {
    "inverted_loop": dict(success=1.0, lap1=3.37, lap2_5=3.25, speed=13.0, accel=6.0),
    "ladder_inverted_loop": dict(success=1.0, lap1=3.78, lap2_5=3.62, speed=13.0, accel=6.0),
}


def build_agent(logdir: pathlib.Path, env):
    """Reconstruct the agent from the run's own config.yaml and load its weights.

    Mirrors `dreamerv3.main.make_agent`: the agent's spaces come from the
    *wrapped* env, and its config is `config.agent` plus a handful of top-level
    keys it also reads."""
    import elements
    import ruamel.yaml as yaml
    from dreamerv3.agent import Agent
    from dreamerv3.main import wrap_env

    cfg = elements.Config(yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text()))
    # The decoder is skipped during policy by default; we need it for the
    # decoded-state metrics that test the paper's interpretability claim.
    cfg = cfg.update({"agent": {"decode_in_policy": True}})

    wrapped = wrap_env(env, cfg)
    obs_space = {k: v for k, v in wrapped.obs_space.items() if not k.startswith("log/")}
    act_space = {k: v for k, v in wrapped.act_space.items() if k != "reset"}

    agent = Agent(
        obs_space,
        act_space,
        elements.Config(
            **cfg.agent,
            logdir=str(logdir),
            seed=cfg.seed,
            jax=cfg.jax,
            batch_size=cfg.batch_size,
            batch_length=cfg.batch_length,
            replay_context=cfg.replay_context,
            report_length=cfg.report_length,
            replica=cfg.replica,
            replicas=cfg.replicas,
        ),
    )
    cp = elements.Checkpoint(str(logdir / "ckpt"))
    cp.agent = agent
    cp.load(keys=["agent"])
    return agent, cfg


def rollout(agent, cfg_env, episodes: int, laps: int, seed: int, chunk: int = 64):
    """Drive the JAX env directly rather than through embodied's driver, so the
    ground-truth `EnvState` stays visible alongside the policy's own beliefs.

    Everything runs under one transfer guard.  This function is a host-side
    driver by nature -- it reads ground truth out of the env state and pushes
    actions back every step -- and DreamerV3 sets
    `jax_transfer_guard='disallow'` process-wide.  On GPU each of those reads is
    a real device-to-host transfer; on CPU none of them are, so a *missing*
    guard is invisible to the CPU smoke test and only explodes on the rented
    machine.  Hence one guard around the whole thing rather than a scattering of
    them that has to be kept complete by hand."""
    import jax

    with jax.transfer_guard("allow"):
        return _rollout(agent, cfg_env, episodes, laps, seed, chunk)


def _rollout(agent, cfg_env, episodes: int, laps: int, seed: int, chunk: int = 64):
    import jax
    import jax.numpy as jnp

    from skydreamer.dynamics import CONTROL_DT
    from skydreamer.env import batched, gates_passed
    from skydreamer.track import gate_rotation, plane_pose

    reset_b, step_b = batched(cfg_env)
    n_gates = cfg_env.track.n_gates
    target_gates = laps * n_gates

    records = []
    # Decode error is a per-timestep quantity over every still-flying drone, so
    # it is accumulated globally rather than attributed to individual episodes.
    decode_all = {"p_w": [], "v_w": []}
    done_total = 0
    while done_total < episodes:
        b = min(chunk, episodes - done_total)
        with jax.transfer_guard("allow"):
            st, obs = reset_b(jax.random.split(jax.random.key(seed + done_total), b))
        carry = agent.init_policy(b)

        alive = np.ones(b, bool)
        finished = np.zeros(b, bool)
        steps = np.zeros(b, int)
        gate_times = [[] for _ in range(b)]
        gate_errs = [[] for _ in range(b)]
        margin = np.full(b, np.inf)
        vmax = np.zeros(b)
        amax = np.zeros(b)
        decode_err = {k: [] for k in ("p_w", "v_w")}
        prev_gates = np.zeros(b, int)
        prev_v = np.zeros((b, 3))
        is_first = np.ones(b, bool)

        for t in range(cfg_env.max_steps):
            with jax.transfer_guard("allow"):
                o = {k: np.asarray(v) for k, v in obs.items()}
            o["mask"] = (o["mask"] > 0.5).astype(np.uint8) * 255
            o = {k: (v.astype(np.float32) if v.dtype != np.uint8 else v) for k, v in o.items()}
            o.update(
                reward=np.zeros(b, np.float32),
                is_first=is_first.copy(),
                is_last=np.zeros(b, bool),
                is_terminal=np.zeros(b, bool),
            )
            carry, act, outs = agent.policy(carry, o, mode="eval")
            is_first[:] = False

            # embodied's NormalizeAction wrapper maps the policy's tanh output
            # from [-1, 1] onto the action space.  Driving the env directly
            # means doing that ourselves; skipping it halves every command.
            a = np.asarray(act["action"], np.float32)
            u = np.clip((a + 1.0) / 2.0, 0.0, 1.0)

            for key in ("p_w", "v_w"):
                rk = f"recon_info_{key}"
                if rk in outs:
                    truth = np.asarray(getattr(st.s, "p" if key == "p_w" else "v"))
                    err = np.linalg.norm(np.asarray(outs[rk]) - truth, axis=-1)
                    decode_err[key].append(err[alive].copy())

            with jax.transfer_guard("allow"):
                st, obs, rew, term, trunc = step_b(st, jnp.asarray(u))
                p = np.asarray(st.s.p)
                v = np.asarray(st.s.v)
                gates = np.asarray(gates_passed(st))
                terminated = np.asarray(term)
                plane = np.asarray(st.plane)
                d_g = np.asarray(st.d_g)
                t_g = np.asarray(st.t_g)

            speed = np.linalg.norm(v, axis=-1)
            # Specific force, i.e. what an accelerometer would read: the paper's
            # "6 g" includes the 1 g the drone holds up against gravity.
            acc = np.linalg.norm((v - prev_v) / CONTROL_DT - np.array([0, 0, 9.81]), axis=-1)
            prev_v = v

            live = alive & ~finished
            vmax = np.where(live, np.maximum(vmax, speed), vmax)
            amax = np.where(live & (t > 2), np.maximum(amax, acc / 9.81), amax)
            steps = np.where(live, t + 1, steps)

            # gate crossing: `gates_passed` ticks once the post-gate is cleared
            crossed = live & (gates > prev_gates)
            if crossed.any():
                with jax.transfer_guard("allow"):
                    centre, yaw, _ = jax.vmap(plane_pose, in_axes=(None, 0, 0))(
                        cfg_env.track, plane - 1, t_g
                    )
                    local = np.asarray(
                        jnp.einsum("bij,bj->bi", gate_rotation(yaw), jnp.asarray(p) - centre)
                    )
                off = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
                for i in np.nonzero(crossed)[0]:
                    gate_times[i].append((t + 1) * CONTROL_DT)
                    gate_errs[i].append(float(np.hypot(local[i, 1], local[i, 2])))
                    margin[i] = min(margin[i], float(d_g[i] - off[i]))
            prev_gates = gates

            finished |= live & (gates >= target_gates)
            alive &= ~terminated
            if (finished | ~alive).all():
                break

        for i in range(b):
            records.append(
                dict(
                    success=bool(finished[i]),
                    gates=int(prev_gates[i]),
                    duration=float(steps[i] * CONTROL_DT),
                    gate_times=gate_times[i],
                    gate_err=gate_errs[i],
                    margin=float(margin[i]) if np.isfinite(margin[i]) else None,
                    vmax=float(vmax[i]),
                    amax=float(amax[i]),
                )
            )
        for k, v in decode_err.items():
            if v:
                decode_all[k].append(np.concatenate(v))
        done_total += b
        print(f"  {done_total}/{episodes} episodes", flush=True)
    decode = {k: float(np.mean(np.concatenate(v))) for k, v in decode_all.items() if v}
    return records, decode


def summarize(records, decode: dict, n_gates: int, laps: int):
    ok = [r for r in records if r["success"]]
    out = {
        "episodes": len(records),
        "success_rate": len(ok) / max(1, len(records)),
        "mean_gates_passed": float(np.mean([r["gates"] for r in records])),
    }
    if ok:
        lap_times = []
        for r in ok:
            t = r["gate_times"]
            splits = [t[(i + 1) * n_gates - 1] for i in range(laps) if (i + 1) * n_gates - 1 < len(t)]
            lap_times.append([splits[0]] + list(np.diff(splits)))
        lap_times = np.array([x for x in lap_times if len(x) == laps])
        if len(lap_times):
            out["lap1_s"] = float(lap_times[:, 0].mean())
            out["lap1_std"] = float(lap_times[:, 0].std())
            if laps > 1:
                out["lap2plus_s"] = float(lap_times[:, 1:].mean())
                out["lap2plus_std"] = float(lap_times[:, 1:].std())
        out["gate_error_m"] = float(np.mean([np.mean(r["gate_err"]) for r in ok if r["gate_err"]]))
        margins = [r["margin"] for r in ok if r["margin"] is not None]
        out["min_gate_margin_m"] = float(np.min(margins)) if margins else None
        out["max_speed_ms"] = float(np.max([r["vmax"] for r in ok]))
        out["max_accel_g"] = float(np.max([r["amax"] for r in ok]))
    if decode:
        out["decode_error"] = decode
    return out


def report(name: str, s: dict) -> str:
    p = PAPER.get(name, {})
    L = [f"\n=== {name} ===", f"episodes            {s['episodes']}"]

    def row(label, got, want, fmt="{:.2f}"):
        g = fmt.format(got) if got is not None else "n/a"
        w = f"   (paper {fmt.format(want)})" if want is not None else ""
        L.append(f"{label:<20}{g}{w}")

    row("success rate", s["success_rate"], p.get("success"), "{:.1%}")
    row("lap 1 [s]", s.get("lap1_s"), p.get("lap1"))
    row("laps 2+ [s]", s.get("lap2plus_s"), p.get("lap2_5"))
    row("max speed [m/s]", s.get("max_speed_ms"), p.get("speed"))
    row("max accel [g]", s.get("max_accel_g"), p.get("accel"))
    row("mean gate error [m]", s.get("gate_error_m"), None, "{:.3f}")
    row("min gate margin [m]", s.get("min_gate_margin_m"), None, "{:.3f}")
    if "decode_error" in s:
        for k, v in s["decode_error"].items():
            L.append(f"{'decode err ' + k:<20}{v:.3f}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--track", default=None, help="default: whatever the run trained on")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    if not (logdir / "config.yaml").exists():
        sys.exit(f"no config.yaml in {logdir} -- is that a finished run?")

    import ruamel.yaml as yaml

    task = yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())["task"]
    track = args.track or task.split("_", 1)[1]

    from skydreamer.embodied_env import SkyDreamer

    env = SkyDreamer(track, max_steps=args.laps * 1200, seed=args.seed)
    agent, _ = build_agent(logdir, env)

    print(f"evaluating {track}: {args.episodes} episodes x {args.laps} laps", flush=True)
    records, decode = rollout(agent, env.cfg, args.episodes, args.laps, args.seed)
    summary = summarize(records, decode, env.cfg.track.n_gates, args.laps)
    text = report(track, summary)
    print(text)

    out = args.out or (logdir / "evaluation.json")
    out.write_text(json.dumps({"track": track, "laps": args.laps, **summary}, indent=2))
    (logdir / "evaluation.txt").write_text(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
