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


# Which distribution the evaluation episodes are drawn from.
#
# `paper`   Table III's *evaluation* column, every episode from the start gate.
#           This is what Table IV describes, and it is the default.
# `strict`  The same, but with the gate window forced to the *training* value
#           (d_g = 0.8 m instead of 1.0 m).  Our track's inner gate is 1.5 m, so
#           its true half-width is 0.75 m and 1.0 m is more lenient than the real
#           gate; this is the number to watch before trusting a policy outdoors.
# `mixed`   `EnvConfig`'s own default, train_fraction = 0.7: 70% of episodes come
#           from the *training* column and start in front of a random gate.  It is
#           the schedule section III-A describes for *training*, and until this
#           flag existed it was what we evaluated on by accident -- which is why
#           early evaluations read ~20 points below the paper's protocol.  Kept as
#           a stress test, because it contains start states the paper never
#           evaluates (standing still in mid-air in front of the virtual gate).
PROTOCOLS = {
    "paper": dict(train_fraction=0.0, d_g=None),
    "strict": dict(train_fraction=0.0, d_g=0.8),
    "mixed": dict(train_fraction=0.7, d_g=None),
    # `physical` Paper start and Table III evaluation column, but the sim's own gate
    #           window is switched off (d_g = 3 m) and the flight is scored after the
    #           fact against our real gate: every airframe point (props, arms, camera)
    #           that crosses a real gate's plane must do so inside the 1.5 m opening,
    #           the virtual gate needs the centre within 1 m (the paper's d_g = 1.0),
    #           and MonoRace's rule applies -- crossing the plane outside the opening
    #           is a crash.  The policy never observes d_g, so the flight itself is
    #           unchanged; only the verdict differs.  Combine with `--hw` to also
    #           give the simulator our gate frame, camera offset and thrust-to-weight.
    #           The sim window is 1.0 m, not off: `hardware.check_flight` leaves a
    #           centre crossing more than `near` (1.4 m) out to another gate, so at
    #           3 m a flight could go past a gate 1.4-3 m to the side, the sim would
    #           count the gate and the check would not see it.  Any centre beyond
    #           ~0.45 m fails the 0.6 m opening anyway, so 1.0 drops no real pass.
    "physical": dict(train_fraction=0.0, d_g=1.0, physical=True),
}

# `--hw`: the simulator moved to our drone and gate (skydreamer/hardware.py).
HW_FULL = dict(twr=4.1, gate_outer=2.1, cam_offset=0.10, fov=(114.6, 92.15))


def apply_hardware(cfg_env, twr=None, gate_outer=None, cam_offset=None, fov=None):
    """Return `cfg_env` with our gate frame and camera offset, and scale the
    simulator's nominal thrust-to-weight to `twr` (process-global: NOMINAL is read
    when the env is traced, so call this before the first step)."""
    import jax.numpy as jnp

    from skydreamer import hardware
    if twr is not None:
        hardware.apply_dynamics(twr)
    if gate_outer is not None:
        import jax

        tr = cfg_env.track
        with jax.transfer_guard("allow"):
            outer = jnp.full(tr.outer.shape, float(gate_outer), tr.outer.dtype)
        cfg_env = cfg_env._replace(track=tr._replace(outer=outer))
    if cam_offset is not None:
        cfg_env = cfg_env._replace(cam_offset=float(cam_offset))
    if fov is not None:
        cfg_env = cfg_env._replace(fov=tuple(fov))
    return cfg_env


def protocol_config(cfg_env, name: str):
    """`(cfg, forced_d_g)` for one of `PROTOCOLS`."""
    spec = PROTOCOLS[name]
    return cfg_env._replace(train_fraction=spec["train_fraction"]), spec["d_g"]


def _tilt_deg(q) -> float:
    """Angle between body z and world z, from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def rollout(agent, cfg_env, episodes: int, laps: int, seed: int, chunk: int = 64,
            force_d_g=None, physical: bool = False, dump: bool = False):
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
        return _rollout(agent, cfg_env, episodes, laps, seed, chunk, force_d_g, physical, dump)


# Decode error over the whole flight conflates two different things: how well
# the world model estimates state, and how far off-distribution the drone ends
# up once it starts going wrong.  The first `EARLY_STEPS` are before anything
# has gone wrong, so comparing the two separates them.
EARLY_STEPS = 20

# EnvState.term_cause
CAUSES = {0: "survived", 1: "hit a gate", 2: "hit the ground", 3: "diverged",
          4: "hit a wall or the ceiling"}


def _agent_obs(obs: dict) -> dict:
    """The observation in exactly the dtypes embodied_env.py hands the agent.

    `mask` passes through untouched: the env already emits it as uint8, and
    when PACK_MASK is on those bytes are bit-packed pixels that the agent
    unpacks itself.  This used to threshold it -- correct back when the env
    emitted a float mask, and quietly destructive afterwards, because
    `(byte > 0.5) * 255` turns every packed byte holding a single gate pixel
    into 0xFF, i.e. eight lit pixels in a row.  The policy then flies on a
    grossly dilated image and crashes on a track it was trained to fly.

    `log/*` is dropped for the same reason embodied's driver drops it: it is
    diagnostics, the agent asserts it never arrives, and driving the env
    directly bypasses the driver that would have removed it.
    """
    return {
        k: (v.astype(np.uint8) if k == "mask" else v.astype(np.float32))
        for k, v in obs.items()
        if not k.startswith("log/")
    }


def _hold(mask, frozen, moved):
    """Per-episode select between two pytrees: keep `frozen` where `mask`."""
    import jax
    import jax.numpy as jnp

    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
        frozen,
        moved,
    )


def _rollout(agent, cfg_env, episodes: int, laps: int, seed: int, chunk: int = 64,
             force_d_g=None, physical: bool = False, dump: bool = False):
    import jax
    import jax.numpy as jnp

    from skydreamer.dynamics import CONTROL_DT
    from skydreamer.env import batched, gates_passed
    from skydreamer.dynamics import quat_to_euler
    from skydreamer.track import floor_penalty, gate_rotation, plane_pose

    reset_b, step_b = batched(cfg_env)
    n_gates = cfg_env.track.n_gates
    target_gates = laps * n_gates

    records = []
    # Decode error is a per-timestep quantity over every still-flying drone, so
    # it is accumulated globally rather than attributed to individual episodes.
    decode_all = {"p_w": [], "v_w": []}
    decode_early_all = {"p_w": [], "v_w": []}
    done_total = 0
    while done_total < episodes:
        b = min(chunk, episodes - done_total)
        with jax.transfer_guard("allow"):
            st, obs = reset_b(jax.random.split(jax.random.key(seed + done_total), b))
            if force_d_g is not None:
                st = st._replace(d_g=jnp.full_like(st.d_g, force_d_g))
        carry = agent.init_policy(b)
        # Thrust-to-weight of each airframe: takeoff deaths track it (docs/deviations.md).
        with jax.transfer_guard("allow"):
            twr_b = np.asarray(st.params.k_w * 4 * st.params.w_max ** 2 / 9.81)
            # What each flight was dealt: the same (seed, episodes) deals every
            # checkpoint the same flights, so two checkpoints can be compared
            # flight by flight, and a flight that fails under every checkpoint
            # can be traced to its draw (scripts/diagnose_failures.py).
            params_b = {k: np.asarray(v) for k, v in st.params._asdict().items()}
            p0_b = np.asarray(st.s.p)
            yaw0_b = np.degrees(np.asarray(quat_to_euler(st.s.q))[:, 2])

        alive = np.ones(b, bool)
        finished = np.zeros(b, bool)
        steps = np.zeros(b, int)
        gate_times = [[] for _ in range(b)]
        gate_errs = [[] for _ in range(b)]
        # For the per-episode dump: where each gate was crossed (gate frame, at
        # its post plane) and how fast, and the state the flight ended in.
        crossings = [[] for _ in range(b)]
        end = [None] * b
        margin = np.full(b, np.inf)
        # track.floor_penalty at its default weight, summed over the flight, whether
        # or not the run trained with it: where a flight sank toward the floor.
        floor_pen = np.zeros(b)
        vmax = np.zeros(b)
        amax = np.zeros(b)
        decode_err = {k: [] for k in ("p_w", "v_w")}
        decode_early = {k: [] for k in ("p_w", "v_w")}
        trace_p, trace_q, trace_v, trace_u, trace_dp = [], [], [], [], []
        prev_gates = np.zeros(b, int)
        prev_v = np.zeros((b, 3))
        is_first = np.ones(b, bool)

        for t in range(cfg_env.max_steps):
            with jax.transfer_guard("allow"):
                o = {k: np.asarray(v) for k, v in obs.items()}
            o = _agent_obs(o)
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

            live = alive & ~finished
            # Decoded and true position at the same instant, kept for the gate
            # crossing below: does the agent think it is where it is, at the gate?
            with jax.transfer_guard("allow"):
                p_pre = np.asarray(st.s.p)
            dec_p = np.asarray(outs["recon_info_p_w"]) if "recon_info_p_w" in outs else None
            for key in ("p_w", "v_w"):
                rk = f"recon_info_{key}"
                if rk in outs:
                    truth = np.asarray(getattr(st.s, "p" if key == "p_w" else "v"))
                    err = np.linalg.norm(np.asarray(outs[rk]) - truth, axis=-1)
                    decode_err[key].append(err[live].copy())
                    if t < EARLY_STEPS:
                        decode_early[key].append(err[live].copy())

            with jax.transfer_guard("allow"):
                st_next, obs_next, rew, term, trunc = step_b(st, jnp.asarray(u))
                # An episode that has crashed or finished must stop being
                # integrated.  embodied's driver resets an env on `is_last`;
                # driving the env directly means holding the state still
                # ourselves.  Without this a drone that hit a gate on step 100
                # keeps being flown for the remaining 1900 steps -- the
                # divergence guard terminates it again every step, to nobody's
                # benefit, while its body rates grow without bound until they
                # overflow to NaN and the agent's finiteness assert fires.  The
                # batch runs to the *slowest* episode, so one early crash is
                # enough; only a policy good enough to keep episodes alive that
                # long ever gets there, which is why this survived every smoke
                # test and appeared on the 17M checkpoint.
                hold = jnp.asarray(~live)
                st = _hold(hold, st, st_next)
                obs = _hold(hold, obs, obs_next)
                p = np.asarray(st.s.p)
                v = np.asarray(st.s.v)
                gates = np.asarray(gates_passed(st))
                terminated = np.asarray(term) & live
                plane = np.asarray(st.plane)
                d_g = np.asarray(st.d_g)
                t_g = np.asarray(st.t_g)

            q = np.asarray(st.s.q)
            if cfg_env.floor is not None:
                with jax.transfer_guard("allow"):
                    fp = np.asarray(floor_penalty(st.s.p, st.s.v, quat_to_euler(st.s.q), *cfg_env.floor))
                floor_pen += np.where(live, fp, 0.0)
            if physical or dump:
                trace_p.append(p.copy())
                trace_q.append(q.copy())
                trace_v.append(v.copy())
            if dump:
                trace_u.append(u.copy())
                trace_dp.append(np.full_like(p_pre, np.nan) if dec_p is None else dec_p.copy())
            for i in np.nonzero(terminated)[0]:
                end[i] = dict(step=t + 1, alt=float(-p[i, 2]), p=p[i].tolist(), v=v[i].tolist(),
                              tilt_deg=_tilt_deg(q[i]))

            speed = np.linalg.norm(v, axis=-1)
            # Specific force, i.e. what an accelerometer would read: the paper's
            # "6 g" includes the 1 g the drone holds up against gravity.
            acc = np.linalg.norm((v - prev_v) / CONTROL_DT - np.array([0, 0, 9.81]), axis=-1)
            prev_v = v

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
                    # The step before the crossing, in the same gate frame: the
                    # truth and what the world model decodes from its latent.
                    local_pre = np.asarray(
                        jnp.einsum("bij,bj->bi", gate_rotation(yaw), jnp.asarray(p_pre) - centre)
                    )
                    local_dec = None if dec_p is None else np.asarray(
                        jnp.einsum("bij,bj->bi", gate_rotation(yaw), jnp.asarray(dec_p) - centre)
                    )
                off = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
                for i in np.nonzero(crossed)[0]:
                    c = dict(gate=int((plane[i] - 1) // 3 % n_gates),
                             y=float(local[i, 1]), z=float(local[i, 2]),
                             speed=float(speed[i]), alt=float(-p[i, 2]),
                             vz=float(v[i, 2]),
                             lap=len(crossings[i]) // n_gates,
                             true_y=float(local_pre[i, 1]), true_z=float(local_pre[i, 2]))
                    if local_dec is not None:
                        c.update(dec_y=float(local_dec[i, 1]), dec_z=float(local_dec[i, 2]))
                    crossings[i].append(c)
                    gate_times[i].append((t + 1) * CONTROL_DT)
                    gate_errs[i].append(float(np.hypot(local[i, 1], local[i, 2])))
                    margin[i] = min(margin[i], float(d_g[i] - off[i]))
            prev_gates = gates

            # A strike on the final gate's post plane ticks `gates` to the target in the
            # very step that terminates the episode; counting it as finished scored a
            # crash as a completed run.  One episode in a hundred at a 0.8 m window.
            finished |= live & ~terminated & (gates >= target_gates)
            alive &= ~terminated
            if (finished | ~alive).all():
                break

        with jax.transfer_guard("allow"):
            causes = np.asarray(st.term_cause)
        phys = [None] * b
        if physical:
            from skydreamer import hardware

            tp, tq = np.stack(trace_p, 1), np.stack(trace_q, 1)
            tr = cfg_env.track
            gates_def = [(np.asarray(tr.pos[g]), float(tr.yaw[g]), bool(tr.visible[g]))
                         for g in range(tr.n_gates)]
            for i in range(b):
                phys[i] = hardware.check_flight(tp[i], tq[i], gates_def, int(steps[i]),
                                                CONTROL_DT, opening=float(tr.inner[0]) / 2)
        for i in range(b):
            sim_ok = bool(finished[i])
            ok_phys = True if phys[i] is None else phys[i][0]
            bad = [] if phys[i] is None else [e for e in phys[i][1] if not e[4]]
            records.append(
                dict(
                    success=sim_ok and ok_phys,
                    sim_success=sim_ok,
                    twr=float(twr_b[i]),
                    phys_fail=(None if not bad else dict(step=bad[0][0], gate=bad[0][1],
                                                         off=bad[0][3])),
                    phys_events=[(e[1], e[2], e[3]) for e in (phys[i][1] if phys[i] else [])],
                    cause=int(causes[i]),
                    gates=int(prev_gates[i]),
                    duration=float(steps[i] * CONTROL_DT),
                    gate_times=gate_times[i],
                    gate_err=gate_errs[i],
                    crossings=crossings[i],
                    end=end[i],
                    margin=float(margin[i]) if np.isfinite(margin[i]) else None,
                    floor_pen=float(floor_pen[i]),
                    vmax=float(vmax[i]),
                    amax=float(amax[i]),
                    seed=int(seed),
                    index=int(done_total + i),
                    start=dict(p=p0_b[i].tolist(), yaw_deg=float(yaw0_b[i])),
                    params={k: (float(v[i]) if v[i].ndim == 0 else v[i].tolist())
                            for k, v in params_b.items()},
                )
            )
            if dump:
                n = int(steps[i])
                # Step t: p, v, q after it; u the command flown during it; dp the
                # position the world model decoded before it (i.e. of p[t-1]).
                records[-1]["_traj"] = {k: np.stack(tr, 1)[i, :n].astype(np.float32)
                                        for k, tr in (("p", trace_p), ("v", trace_v), ("q", trace_q),
                                                      ("u", trace_u), ("dp", trace_dp))}
        for k, v in decode_err.items():
            if v:
                decode_all[k].append(np.concatenate(v))
        for k, v in decode_early.items():
            if v:
                decode_early_all[k].append(np.concatenate(v))
        done_total += b
        print(f"  {done_total}/{episodes} episodes", flush=True)
    decode = {k: float(np.mean(np.concatenate(v))) for k, v in decode_all.items() if v}
    decode["early"] = {
        k: float(np.mean(np.concatenate(v))) for k, v in decode_early_all.items() if v
    }
    return records, decode


def summarize(records, decode: dict, n_gates: int, laps: int, virtual=()):
    """Two layers.

    The paper-comparison layer keeps the paper's definitions exactly: Table IV
    reports lap times and speeds for flights that *completed*, so those stay
    restricted to successful episodes and read n/a when there are none.

    The diagnostic layer describes every episode, successful or not.  A run
    that fails needs this most and had it least: the first evaluation of the
    finished 17M checkpoint printed n/a on six of seven rows, which says
    nothing about how it failed.  Speed, acceleration, gate error and gate
    margin are all perfectly well defined for a flight that ends in a crash.
    """
    ok = [r for r in records if r["success"]]
    n = max(1, len(records))
    out = {
        "episodes": len(records),
        "success_rate": len(ok) / n,
        "mean_gates_passed": float(np.mean([r["gates"] for r in records])),
    }

    # --- paper comparison: completed flights only --------------------------
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

    # --- diagnostics: every episode ----------------------------------------
    gates = np.array([r["gates"] for r in records])
    durations = np.array([r["duration"] for r in records])
    diag = {
        "gates_passed": {
            "mean": float(gates.mean()),
            "max": int(gates.max()),
            # How far it got, as a histogram.  A policy that always dies at the
            # same gate is a different problem from one that dies at random.
            "histogram": {int(g): int((gates == g).sum()) for g in np.unique(gates)},
        },
        "duration_s": {
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "min": float(durations.min()),
            "max": float(durations.max()),
        },
        "termination": {
            CAUSES.get(c, f"code {c}"): int(sum(1 for r in records if r["cause"] == c))
            for c in sorted({r["cause"] for r in records})
        },
    }
    speeds = np.array([r["vmax"] for r in records])
    accels = np.array([r["amax"] for r in records])
    diag["max_speed_ms_all"] = {"mean": float(speeds.mean()), "max": float(speeds.max())}
    diag["max_accel_g_all"] = {"mean": float(accels.mean()), "max": float(accels.max())}

    errs = [e for r in records for e in r["gate_err"]]
    if errs:
        diag["gate_error_m_all"] = {"mean": float(np.mean(errs)), "max": float(np.max(errs))}
    margins = [r["margin"] for r in records if r["margin"] is not None]
    if margins:
        diag["min_gate_margin_m_all"] = float(np.min(margins))
    if any(r.get("floor_pen") for r in records):
        groups = {"finished": [r for r in records if r["success"]],
                  "hit the ground": [r for r in records if r["cause"] == 2],
                  "other": [r for r in records if not r["success"] and r["cause"] != 2]}
        diag["floor_penalty"] = {k: (float(np.mean([r["floor_pen"] for r in v])), len(v))
                                 for k, v in groups.items() if v}
    out["diagnostics"] = diag

    # Success and ground deaths by airframe thrust-to-weight.  The takeoff failure the
    # policy shows is a function of it, so a flat aggregate can hide a weak band.
    if all("twr" in r for r in records):
        edges = [0.0, 3.5, 4.5, 5.5, 7.0, 99.0]
        tw = np.array([r["twr"] for r in records])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r, t in zip(records, tw) if lo <= t < hi]
            if sel:
                rows.append(dict(
                    twr=f"{lo:g}-{hi:g}" if hi < 99 else f">={lo:g}", n=len(sel),
                    success=float(np.mean([r["success"] for r in sel])),
                    gates=float(np.mean([r["gates"] for r in sel])),
                    ground_before_gate1=int(sum(1 for r in sel if r["cause"] == 2 and r["gates"] == 0))))
        out["by_twr"] = rows

    if any("phys_fail" in r and r.get("phys_events") for r in records):
        fails = [r for r in records if r["phys_fail"]]
        # An episode can fail the swept check *and* the sim; attribute by what came first.
        only_phys = [r for r in fails if r["sim_success"]]
        # Real (rendered) gates only.  This used to drop gate index 1, the inverted
        # loop's invisible gate -- and so dropped G2, the worst gate, on `team`.
        real = [e for r in records for e in r["phys_events"] if e[0] not in set(virtual)]
        out["physical"] = {
            "sim_finished_rate": float(np.mean([r["sim_success"] for r in records])),
            "failed_swept_check": len(fails),
            "failed_swept_check_but_sim_finished": len(only_phys),
            "fail_gate_histogram": {int(g): int(sum(1 for r in fails if r["phys_fail"]["gate"] == g))
                                    for g in sorted({r["phys_fail"]["gate"] for r in fails})},
            "real_gate_worst_point_offset_m": {
                q: float(np.percentile([e[2] for e in real], q)) for q in (50, 90, 99)
            } if real else None,
        }

    if decode:
        out["decode_error"] = decode
    gate_est = gate_estimate(records, n_gates)
    if gate_est:
        out["gate_estimate"] = gate_est
    return out


def gate_estimate(records, n_gates: int):
    """Per gate, where the drone truly crossed vs where the world model decoded it
    (gate frame, the step before the crossing).  A bias in `true` that `dec` shares
    is the policy's chosen line; a bias in `true` that `dec` does not see is an
    estimation error the policy cannot correct."""
    out = {}
    for g in range(n_gates):
        cs = [c for r in records for c in r.get("crossings", []) if c["gate"] == g and "dec_y" in c]
        if not cs:
            continue
        a = {k: np.array([c[k] for c in cs]) for k in ("true_y", "true_z", "dec_y", "dec_z")}
        ey, ez = a["dec_y"] - a["true_y"], a["dec_z"] - a["true_z"]
        out[g] = dict(
            n=len(cs),
            true_mean=[float(a["true_y"].mean()), float(a["true_z"].mean())],
            dec_mean=[float(a["dec_y"].mean()), float(a["dec_z"].mean())],
            true_sd=[float(a["true_y"].std()), float(a["true_z"].std())],
            err_mean=[float(ey.mean()), float(ez.mean())],
            err_sd=[float(ey.std()), float(ez.std())],
        )
    return out


def report(name: str, s: dict, protocol: str = "paper") -> str:
    p = PAPER.get(name, {})
    L = [f"\n=== {name} ===", f"protocol            {protocol}",
         f"episodes            {s['episodes']}"]

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

    if s.get("by_twr"):
        L.append("\n--- by thrust-to-weight of the airframe ---")
        for r in s["by_twr"]:
            L.append(f"  TWR {r['twr']:<8} n={r['n']:<4} success {r['success']:.1%}   "
                     f"gates {r.get('gates', float('nan')):5.2f}   "
                     f"ground deaths before gate 1: {r['ground_before_gate1']}")
    ph = s.get("physical")
    if ph:
        L.append(f"{'sim-only success':<20}{ph['sim_finished_rate']:.1%}   (no swept-footprint check)")
        L.append(f"{'swept-check fails':<20}{ph['failed_swept_check']}   of which sim-clean "
                 f"{ph['failed_swept_check_but_sim_finished']}   by gate {ph['fail_gate_histogram']}")
        w = ph["real_gate_worst_point_offset_m"]
        if w:
            L.append(f"{'worst point offset':<20}" + "   ".join(f"p{q} {v:.3f}" for q, v in w.items())
                     + f"   (opening half-width {ph.get('opening_half_width_m', 0.75):g})")

    d = s.get("diagnostics")
    if d:
        L.append("\n--- all episodes, successful or not ---")
        g = d["gates_passed"]
        L.append(f"{'gates passed':<20}mean {g['mean']:.2f}   max {g['max']}")
        hist = "  ".join(f"{k}:{v}" for k, v in sorted(g["histogram"].items()))
        L.append(f"{'  distribution':<20}{hist}")

        t = d["duration_s"]
        L.append(
            f"{'flight time [s]':<20}mean {t['mean']:.2f}   median {t['median']:.2f}"
            f"   min {t['min']:.2f}   max {t['max']:.2f}"
        )
        L.append(
            f"{'ended by':<20}"
            + "   ".join(f"{k} {v}" for k, v in d["termination"].items())
        )
        v = d["max_speed_ms_all"]
        L.append(f"{'max speed [m/s]':<20}mean {v['mean']:.2f}   max {v['max']:.2f}")
        a = d["max_accel_g_all"]
        L.append(f"{'max accel [g]':<20}mean {a['mean']:.2f}   max {a['max']:.2f}")
        if "gate_error_m_all" in d:
            e = d["gate_error_m_all"]
            L.append(f"{'gate error [m]':<20}mean {e['mean']:.3f}   max {e['max']:.3f}")
        if "min_gate_margin_m_all" in d:
            L.append(f"{'gate margin [m]':<20}min {d['min_gate_margin_m_all']:.3f}")
        if "floor_penalty" in d:
            L.append(f"{'floor penalty':<20}" + "   ".join(
                f"{k} {m:.2f} (n={n})" for k, (m, n) in d["floor_penalty"].items())
                + "   (mean per flight)")

    if "decode_error" in s:
        dec = s["decode_error"]
        early = dec.get("early", {})
        L.append("\n--- world model: decoded state vs ground truth [m, m/s] ---")
        # Split the window deliberately.  A model that is accurate early and
        # wrong later is estimating fine and being flown somewhere it has never
        # been; one that is wrong from the first step has a broken input.
        for k in ("p_w", "v_w"):
            if k not in dec:
                continue
            e = f"   first {EARLY_STEPS} steps {early[k]:.3f}" if k in early else ""
            L.append(f"{'decode err ' + k:<20}whole flight {dec[k]:.3f}{e}")
    ge = s.get("gate_estimate")
    if ge:
        L.append("\n--- at the gate: true vs decoded CoM, gate frame (y lateral, z vertical) [m] ---")
        L.append("  gate   n     true mean (y, z)   decoded mean (y, z)   true sd (y, z)   "
                 "dec-true mean (y, z)   dec-true sd (y, z)")
        for g, r in ge.items():
            f2 = lambda v: f"{v[0]:+.3f} {v[1]:+.3f}"
            L.append(f"  G{int(g) + 1:<4} {r['n']:<5} {f2(r['true_mean']):<18} {f2(r['dec_mean']):<21} "
                     f"{f2(r['true_sd']):<16} {f2(r['err_mean']):<22} {f2(r['err_sd'])}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=pathlib.Path)
    ap.add_argument("--track", default=None, help="default: whatever the run trained on")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="paper",
                    help="paper (default): Table III evaluation column from the start gate; "
                         "strict: paper but with the training gate window 0.8 m; "
                         "mixed: the 70/30 training mixture (the old accidental default)")
    ap.add_argument("--hw", action="store_true",
                    help="simulate our drone and gate: outer 2.1 m, camera 0.10 m forward, "
                         "thrust-to-weight 4.1 (override each with the flags below)")
    ap.add_argument("--hw-twr", type=float, default=None,
                    help="nominal thrust-to-weight (sim paper value 6.07)")
    ap.add_argument("--hw-gate-outer", type=float, default=None)
    ap.add_argument("--hw-cam-offset", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--air-start", action="store_true",
                    help="start mid-air as Table III, not on the floor (track.LAUNCH): "
                         "the protocol every run before 7e was scored under")
    ap.add_argument("--dump", action="store_true",
                    help="also write every flight's ground-truth p, v, q, its motor commands u "
                         "and decoded position dp to <stem>_traj.npz")
    ap.add_argument("--erosion", type=float, default=None,
                    help="chance the 1-pixel mask erosion is on (default: the env's, 0.5); "
                         "0 shows every frame as rendered")
    ap.add_argument("--dist-scale", default=None,
                    help="a,Ms,Mf,u: multipliers on the eval disturbances (slow acceleration, "
                         "slow and fast moment, fast motor command); same flights, for ablations")
    ap.add_argument("--gyro-noise", default=None,
                    help="lo,hi: gyro noise sigma range in rad/s (default 0.1,0.5); 0,0 = none")
    ap.add_argument("--rpm-noise", type=float, default=None,
                    help="RPM telemetry noise, fraction of w_max (default 0.01); 0 = none")
    args = ap.parse_args()

    logdir = args.logdir.expanduser()
    if not (logdir / "config.yaml").exists():
        sys.exit(f"no config.yaml in {logdir} -- is that a finished run?")

    import ruamel.yaml as yaml

    task = yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())["task"]
    from skydreamer.hardware import full_hw, keep_oracle, split_task

    # A run trained on our hardware (`hw_*`) is scored on our hardware: `--track big`
    # means `hw_big` there, not the paper's drone on the big track.  A run part-way
    # through the fine-tuning stages (`hw2_*`) is scored on the full profile, the target.
    trained_on = full_hw(task.split("_", 1)[1])
    track = args.track or trained_on
    if split_task(trained_on)[0] is not None:
        track = full_hw(track if split_task(track)[0] else "hw_" + track)
    track = keep_oracle(trained_on, track)

    from skydreamer.embodied_env import SkyDreamer

    env = SkyDreamer(track, max_steps=args.laps * 1200, seed=args.seed)
    agent, _ = build_agent(logdir, env)

    cfg_env, force_d_g = protocol_config(env.cfg, args.protocol)
    if args.air_start:
        cfg_env = cfg_env._replace(launch=None, air_start_alt=None)
    if args.erosion is not None:
        cfg_env = cfg_env._replace(erosion_prob=float(args.erosion))
    if args.dist_scale is not None:
        ds = tuple(float(x) for x in args.dist_scale.split(","))
        assert len(ds) == 4, "--dist-scale takes four numbers: a,Ms,Mf,u"
        cfg_env = cfg_env._replace(dist_scale=ds)
    if args.gyro_noise is not None:
        lo, hi = (float(x) for x in args.gyro_noise.split(","))
        cfg_env = cfg_env._replace(gyro_noise_range=(lo, hi))
    if args.rpm_noise is not None:
        cfg_env = cfg_env._replace(rpm_noise_frac=float(args.rpm_noise))
    if any(a is not None for a in (args.dist_scale, args.gyro_noise, args.rpm_noise)):
        print(f"ablation: dist_scale={cfg_env.dist_scale} gyro_noise={cfg_env.gyro_noise_range} "
              f"rpm_noise={cfg_env.rpm_noise_frac}", flush=True)
    hw =dict(HW_FULL) if args.hw else dict(twr=None, gate_outer=None, cam_offset=None, fov=None)
    for k in hw:
        v = getattr(args, f"hw_{k}", None)
        if v is not None:
            hw[k] = v
    if any(v is not None for v in hw.values()):
        cfg_env = apply_hardware(cfg_env, **hw)
        print(f"hardware overrides: {hw}", flush=True)
    print(f"evaluating {track} [{args.protocol}]: {args.episodes} episodes x {args.laps} laps",
          flush=True)
    records, decode = rollout(agent, cfg_env, args.episodes, args.laps, args.seed,
                              force_d_g=force_d_g, physical=PROTOCOLS[args.protocol].get("physical", False),
                              dump=args.dump)
    traj = [r.pop("_traj") for r in records] if args.dump else None
    vis = np.asarray(env.cfg.track.visible)
    summary = summarize(records, decode, env.cfg.track.n_gates, args.laps,
                        virtual=[int(g) for g in np.nonzero(~vis)[0]])
    if "physical" in summary:
        import jax

        with jax.transfer_guard("allow"):   # DreamerV3 disallows transfers process-wide
            summary["physical"]["opening_half_width_m"] = float(np.asarray(env.cfg.track.inner)[0]) / 2
    text = report(track, summary, args.protocol)
    print(text)

    # Scoring a run on a track it did not train on must not overwrite the
    # training-track result -- `--eval-only` and `--eval-only --big` both land
    # in the same logdir.  Same convention visualize.py uses for `video_big/`.
    stem = "evaluation" if track == trained_on else f"evaluation_{track}"
    if args.protocol != "paper":
        stem += f"_{args.protocol}"
    stem += args.tag
    out = args.out or (logdir / f"{stem}.json")
    out.write_text(json.dumps(
        {"track": track, "protocol": args.protocol, "laps": args.laps, "hardware": hw,
         "seed": args.seed, "episodes": args.episodes, **summary}, indent=2))
    (logdir / f"{stem}.txt").write_text(text)
    # Every episode, for working out *why* flights fail (crash state, where each
    # gate was crossed); summaries alone could not answer that for run 7b/7c.
    (logdir / f"{stem}_episodes.json").write_text(json.dumps(records))
    if traj is not None:
        np.savez_compressed(logdir / f"{stem}_traj.npz", **{
            f"{k}_{i}": t[k] for i, t in enumerate(traj) for k in ("p", "v", "q", "u", "dp")})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
