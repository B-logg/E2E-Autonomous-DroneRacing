#!/usr/bin/env python3
"""Fly a trained policy and record *why* each episode ends, with everything
the evaluation harness throws away: the initial condition, the state at death,
and what the policy did in its first second.

`evaluate.py` reports that 21% of episodes die within one gate.  It cannot say
which ones, because it never records how an episode began.  This does.

Protocols (what `EnvConfig.train_fraction` is set to):

  mixed   0.7, the default `evaluate.py` uses.  70% of episodes are drawn from
          Table III's *training* column and may start in front of any gate;
          30% from the evaluation column and always start at gate 0.
  eval    0.0.  Table III's evaluation column only, always from gate 0 -- the
          protocol the paper's Table IV describes.
  train   1.0.  Training column only, random start gate.

Output is one `.npz` per run, analysed by `scripts/analyse_start.py`.

    python scripts/diagnose_start.py --logdir <run> --protocol mixed --episodes 256
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "dreamerv3"))

PROTOCOLS = {"mixed": 0.7, "eval": 0.0, "train": 1.0}
U_STEPS = 150  # how many initial steps of motor commands to keep


def cpu_logdir(logdir: pathlib.Path) -> pathlib.Path:
    """A run directory the policy can be loaded from on a machine with no GPU.

    The run's config says `jax.platform: cuda`; everything else is kept.  The
    checkpoint is symlinked rather than copied."""
    import ruamel.yaml as yaml

    y = yaml.YAML(typ="safe")
    cfg = y.load((logdir / "config.yaml").read_text())
    cfg["jax"]["platform"] = "cpu"
    cfg["jax"]["prealloc"] = False
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="diag_"))
    with (tmp / "config.yaml").open("w") as f:
        yaml.YAML(typ="safe").dump(cfg, f)
    (tmp / "ckpt").symlink_to((logdir / "ckpt").resolve())
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=pathlib.Path, required=True)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="mixed")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--target-gates", type=int, default=15)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--track", default="inverted_loop")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    # --- interventions, for testing a suspected cause rather than observing it
    ap.add_argument("--z-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="override Table III's initial z_g range [m below the gate centre]. "
                         "Altitude = 1.35 - z_g, so 0.2 0.8 starts the drone at 0.55-1.15 m.")
    ap.add_argument("--v0", type=float, default=0.0,
                    help="start every episode moving this fast [m/s] along the start gate's "
                         "normal, instead of from rest")
    ap.add_argument("--burn-in", type=int, default=0,
                    help="hold the drone still for this many steps while the policy keeps "
                         "receiving observations (its actions are discarded), then release.  "
                         "Tests whether a settled belief, rather than the launch itself, is "
                         "what limits the first gate.  Death times include these steps.")
    ap.add_argument("--cam-pitch", type=float, default=None,
                    help="force the camera's upward tilt [deg] (Table III trains on 45-55)")
    ap.add_argument("--d-g", type=float, default=None,
                    help="force the gate collision half-window [m] (Table III: 0.8 train, 1.0 eval)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from evaluate import _agent_obs, _hold, build_agent
    from skydreamer.dynamics import CONTROL_DT, quat_to_euler
    from skydreamer.embodied_env import SkyDreamer
    from skydreamer.env import batched, gates_passed
    from skydreamer.params import NOMINAL
    from skydreamer.track import gate_rotation, plane_pose, to_gate_frame

    import skydreamer.env as envmod

    if args.z_range is not None:
        # `reset` reads these module globals at trace time, so patching them before
        # `batched(cfg)` builds the jitted function is enough.
        lo, hi = args.z_range
        envmod.TRAIN = envmod.TRAIN._replace(init_z_g=(lo, hi))
        envmod.EVAL = envmod.EVAL._replace(init_z_g=(lo, hi))

    work = cpu_logdir(args.logdir)
    env = SkyDreamer(args.track, max_steps=args.max_steps, seed=args.seed)
    agent, _ = build_agent(work, env)
    cfg = env.cfg._replace(train_fraction=PROTOCOLS[args.protocol],
                           max_steps=args.max_steps)
    reset_b, step_b = batched(cfg)
    w_max_nom = NOMINAL["w_max"]
    T = args.max_steps

    rec: dict[str, list] = {}

    def put(k, v):
        rec.setdefault(k, []).append(np.asarray(v))

    done = 0
    with jax.transfer_guard("allow"):
        while done < args.episodes:
            b = min(args.chunk, args.episodes - done)
            st, obs = reset_b(jax.random.split(jax.random.key(args.seed + done), b))

            if args.cam_pitch is not None:
                # Applies from the next render on; the first few frames in the image
                # delay buffer were drawn with the sampled extrinsics.
                st = st._replace(c_e=st.c_e.at[:, 1].set(np.deg2rad(args.cam_pitch)))
            if args.d_g is not None:
                st = st._replace(d_g=jnp.full_like(st.d_g, args.d_g))
            if args.v0:
                g_ = np.asarray(st.plane) // 3
                yaw_ = np.asarray(cfg.track.yaw)[g_]
                nrm = np.stack([np.cos(yaw_), np.sin(yaw_), np.zeros_like(yaw_)], -1)
                st = st._replace(s=st.s._replace(v=jnp.asarray(args.v0 * nrm, jnp.float32)))

            # ---- initial condition, recorded before anything moves
            gate0 = np.asarray(st.plane) // 3
            gpos = np.asarray(cfg.track.pos)[gate0]
            gyaw = np.asarray(cfg.track.yaw)[gate0]
            pg0 = np.asarray(jax.vmap(to_gate_frame)(
                st.s.p, jnp.asarray(gpos), jnp.asarray(gyaw)))
            eul0 = np.asarray(quat_to_euler(st.s.q))
            put("start_gate", gate0)
            put("p0", st.s.p)
            put("alt0", -np.asarray(st.s.p)[:, 2])
            put("pg0", pg0)
            put("euler0", eul0)
            put("rates0", st.s.omega_b)
            put("motor0", np.asarray(st.s.motor) / w_max_nom)
            put("d_g", st.d_g)
            put("t_g", st.t_g)
            put("c_e", st.c_e)
            put("gyro_sigma", st.gyro_sigma)
            put("bounds", st.bounds)
            put("params", jax.vmap(lambda p: p.as_vector())(st.params))
            put("tau", st.params.tau)
            put("k_w", st.params.k_w)
            put("w_max_ep", st.params.w_max)
            put("info_att0", np.asarray(obs["info_att"]))

            carry = agent.init_policy(b)
            alive = np.ones(b, bool)
            finished = np.zeros(b, bool)
            death_t = np.full(b, -1)
            first = np.ones(b, bool)
            gate_hist = np.zeros((T, b), np.int16)
            # Centre offset, max(|y|,|z|) in the gate frame, at every plane crossing.
            # The policy never observes `d_g`, so with a window wide enough that
            # nothing collides its trajectory does not depend on the window -- and
            # the success rate at *any* smaller window is the fraction of episodes
            # whose offsets all stay below it.  One run answers the whole sweep.
            off_cross = np.full((T, b), np.nan, np.float32)
            plane_cross = np.full((T, b), -1, np.int16)
            prev_plane = np.asarray(st.plane).copy()
            dec_err = np.full((150, b), np.nan, np.float32)   # world-model position error
            ser = {k: np.full((T, b), np.nan, np.float32)
                   for k in ("alt", "vz", "speed", "tilt", "rate_l1", "motor")}
            ucmd = np.full((U_STEPS, b, 4), np.nan, np.float32)

            for t in range(T):
                o = _agent_obs({k: np.asarray(v) for k, v in obs.items()})
                o.update(reward=np.zeros(b, np.float32), is_first=first.copy(),
                         is_last=np.zeros(b, bool), is_terminal=np.zeros(b, bool))
                carry, act, outs = agent.policy(carry, o, mode="eval")
                if t < 150 and "recon_info_p_w" in outs:
                    truth = np.asarray(st.s.p)
                    dec_err[t] = np.linalg.norm(np.asarray(outs["recon_info_p_w"]) - truth, axis=-1)
                first[:] = False
                u = np.clip((np.asarray(act["action"], np.float32) + 1.0) / 2.0, 0, 1)
                live = alive & ~finished
                if t < U_STEPS:
                    ucmd[t][live] = u[live]

                st_next, obs_next, _, term, _ = step_b(st, jnp.asarray(u))
                # During burn-in everything is frozen: the drone does not move, and
                # the policy sees the same observation again.
                hold = jnp.asarray(~live | (t < args.burn_in))
                st = _hold(hold, st, st_next)
                obs = _hold(hold, obs, obs_next)

                p = np.asarray(st.s.p)
                v = np.asarray(st.s.v)
                q = np.asarray(st.s.q)
                eul = np.asarray(quat_to_euler(st.s.q))
                gates = np.asarray(gates_passed(st))
                gate_hist[t] = gates
                ser["alt"][t][live] = (-p[:, 2])[live]
                ser["vz"][t][live] = v[:, 2][live]
                ser["speed"][t][live] = np.linalg.norm(v, axis=1)[live]
                # tilt of the body z axis from vertical
                tilt = np.degrees(np.arccos(np.clip(
                    1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
                ser["tilt"][t][live] = tilt[live]
                ser["rate_l1"][t][live] = np.abs(np.asarray(st.s.omega_b)).sum(1)[live]
                ser["motor"][t][live] = (np.asarray(st.s.motor).mean(1) / w_max_nom)[live]

                plane_now = np.asarray(st.plane)
                crossed = live & (plane_now > prev_plane)
                if crossed.any():
                    c_, y_, _ = jax.vmap(plane_pose, in_axes=(None, 0, 0))(
                        cfg.track, jnp.asarray(np.maximum(plane_now - 1, 0)), st.t_g)
                    loc = np.asarray(jnp.einsum("bij,bj->bi", gate_rotation(y_), st.s.p - c_))
                    off = np.maximum(np.abs(loc[:, 1]), np.abs(loc[:, 2]))
                    off_cross[t][crossed] = off[crossed]
                    plane_cross[t][crossed] = (plane_now - 1)[crossed]
                prev_plane = plane_now.copy()

                terminated = np.asarray(term) & live
                death_t[terminated] = t
                finished |= live & ~terminated & (gates >= args.target_gates)
                alive &= ~terminated
                if (finished | ~alive).all():
                    break

            # ---- state at death.  `_hold` froze it there.
            plane = np.asarray(st.plane)
            hit_plane = np.maximum(plane - 1, 0)   # the plane a gate strike crossed
            centre, yaw, _ = jax.vmap(plane_pose, in_axes=(None, 0, 0))(
                cfg.track, jnp.asarray(hit_plane), st.t_g)
            local = np.asarray(jnp.einsum(
                "bij,bj->bi", gate_rotation(yaw), st.s.p - centre))
            put("gates", np.asarray(gates_passed(st)))
            put("cause", np.asarray(st.term_cause))
            put("finished", finished)
            put("death_t", death_t)
            put("plane_end", plane)
            put("p_end", st.s.p)
            put("v_end", st.s.v)
            put("euler_end", quat_to_euler(st.s.q))
            put("local_end", local)
            put("rates_end", st.s.omega_b)
            put("gate_hist", gate_hist.T)
            put("dec_err", dec_err.T)
            put("off_cross", off_cross.T)
            put("plane_cross", plane_cross.T)
            for k, a in ser.items():
                put(f"ser_{k}", a.T)
            put("ucmd", np.transpose(ucmd, (1, 0, 2)))
            done += b
            print(f"  {done}/{args.episodes}", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in rec.items()}
    out["dt"] = np.float32(CONTROL_DT)
    out["protocol"] = np.array(args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    ok = out["finished"].mean()
    print(f"wrote {args.out}   protocol={args.protocol}  reached "
          f"{args.target_gates} gates: {ok:.1%}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
