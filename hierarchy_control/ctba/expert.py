"""A hand-written CTBA controller that flies the track, so the replay buffer holds
completed laps before the agent has flown one (curriculum stage c1).

Same path tracking as skydreamer.expert (a continuous reference through every
gate, nearest point + carrot, heading along the path), but it ends where the
policy ends: a collective throttle, roll/pitch angle targets and a yaw rate,
which Betaflight's ANGLE mode (ctba.plant: estimator, level loop, rate loop)
then flies.  No attitude or rate loop of its own -- that is the point of CTBA.

Privileged: it reads the true state, this episode's plant parameters (thrust
map scale, battery, climb loss, drag, level gain) and the FC's own attitude
estimate, from which it inverts the level loop.  The demonstrations it produces are in the policy's own
action space, at the policy's rate, through the same delays.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from skydreamer.dynamics import quat_to_matrix

from . import params as P
from . import plant
from .env import command_to_action

GRAVITY_VEC = jnp.array([0.0, 0.0, P.GRAVITY])


class Gains(NamedTuple):
    k_vel: float = 3.5          # velocity error -> acceleration, 1/s
    k_att: tuple = (7.0, 7.0, 2.5)   # attitude error -> wanted body rate, 1/s
    max_tilt_deg: float = 35.0
    a_thrust_min: float = 3.0   # m/s^2: never ask for (near) free fall


class Carry(NamedTuple):
    idx: jax.Array
    z_prev: jax.Array


def init(ref, p_w) -> Carry:
    return Carry(idx=jnp.argmin(jnp.linalg.norm(ref["p"] - p_w, axis=1)).astype(jnp.int32),
                 z_prev=jnp.array([0.0, 0.0, -1.0]))


def _vee(S):
    return jnp.stack([S[2, 1], S[0, 2], S[1, 0]])


def control(carry: Carry, st, ref: dict, v_nom: float = 2.0, lookahead: float = 0.8,
            alt_floor: float = 0.6, g: Gains = Gains(), search_back: int = 10,
            search_fwd: int = 140):
    """One step.  `st` is a ctba.env.EnvState.  Returns (carry, action in [-1, 1]^4)."""
    s, p = st.s, st.params
    Pp, TAN, V = ref["p"], ref["tangent"], ref["v"]
    n = Pp.shape[0]
    cand = (carry.idx + jnp.arange(-search_back, search_fwd)) % n
    i = cand[jnp.argmin(jnp.linalg.norm(Pp[cand] - s.p, axis=1))]
    j = (i + jnp.int32(lookahead / ref["ds"])) % n

    d = Pp[j] - s.p
    speed = jnp.minimum(v_nom, V[j])
    v_cmd = speed * d / jnp.maximum(jnp.linalg.norm(d), 1e-6)
    v_cmd = v_cmd.at[2].add(-jnp.maximum(alt_floor + s.p[2], 0.0) * 2.0)

    R = quat_to_matrix(s.q)
    vb = R.T @ s.v
    drag_b = jnp.stack([-p.drag_x * vb[0] - p.drag_x2 * vb[0] * jnp.abs(vb[0]),
                        -p.drag_y * vb[1] - p.drag_y2 * vb[1] * jnp.abs(vb[1]), 0.0])
    a_des = g.k_vel * (v_cmd - s.v)
    a_thrust = a_des - GRAVITY_VEC - R @ drag_b

    up = jnp.minimum(a_thrust[2], -g.a_thrust_min)
    horiz = a_thrust[:2]
    lim = jnp.abs(up) * jnp.tan(jnp.deg2rad(g.max_tilt_deg))
    horiz = horiz * jnp.minimum(1.0, lim / jnp.maximum(jnp.linalg.norm(horiz), 1e-9))
    a_thrust = jnp.concatenate([horiz, up[None]])
    T = jnp.maximum(jnp.linalg.norm(a_thrust), 1e-3)
    z_B = -a_thrust / T

    tan = TAN[j]
    h = jnp.array([tan[0], tan[1], 0.0])
    cur = jnp.array([R[0, 0], R[1, 0], 0.0])
    h = jnp.where(jnp.linalg.norm(h) > 0.3, h, cur)
    x_C = h / jnp.maximum(jnp.linalg.norm(h), 1e-6)
    y_B = jnp.cross(z_B, x_C)
    ny = jnp.linalg.norm(y_B)
    y_B = jnp.where(ny > 1e-3, y_B / jnp.maximum(ny, 1e-9), jnp.array([0.0, 1.0, 0.0]))
    R_des = jnp.stack([jnp.cross(y_B, z_B), y_B, z_B], axis=1)

    # the body rates wanted (attitude error + the reference's own rotation) ...
    dt = P.CONTROL_DT
    w_ff = jnp.clip(R.T @ jnp.cross(carry.z_prev, (z_B - carry.z_prev) / dt), -6.0, 6.0)
    e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
    rates = -jnp.asarray(g.k_att) * e_R + w_ff
    yaw_rate = jnp.clip(rates[2], -P.YAW_RATE_MAX, P.YAW_RATE_MAX)
    # ... turned into the angle targets that make Betaflight's level loop ask for them:
    # its setpoint is level_gain * (target - FC estimate), so target = estimate + rate / gain.
    # (Alone, the level loop's 0.2 s time constant is too slow to hold a racing line.)
    tgt = plant.attitude_rp(s.q_est) + rates[:2] / p.level_gain
    tgt = jnp.clip(tgt, -P.ANGLE_LIMIT, P.ANGLE_LIMIT)
    roll, pitch = tgt[0], tgt[1]

    # thrust along body z the plant must produce, before its climb loss
    a_t_need = T - p.climb_loss * jnp.clip(vb[2], -25.0, 3.0)
    v_bat = P.battery_voltage(p, s.a_t)
    thr = P.throttle_for(a_t_need, v_bat, p.thrust_scale)

    cmd = jnp.stack([thr, roll, pitch, yaw_rate])
    return Carry(idx=i.astype(jnp.int32), z_prev=z_B), command_to_action(cmd)


def reference(track, v_nom: float):
    """skydreamer's continuous reference path through the gates, as jnp arrays."""
    from skydreamer import expert_path as ep
    from skydreamer.params import NOMINAL

    r = ep.reference(track, NOMINAL, margin=0.35, v_cap=v_nom * 1.4, iters=1, omega_max=1e9)
    R = {k: jnp.asarray(r[k], jnp.float32) for k in ("p", "tangent", "v")}
    R["ds"] = float(r["ds"])
    return R


def screener(ccfg, ref, v_nom, lookahead, alt_floor, gains):
    """Jitted, vmapped: episode keys -> (gates passed, termination cause)."""
    from . import env as E

    def one(key):
        st, _ = E.reset(key, ccfg)
        c = init(ref, st.s.p)

        def body(carry, _):
            st, c, done = carry
            c2, a = control(c, st, ref, v_nom, lookahead, alt_floor, gains)
            st2, _, _, term, trunc = E.step(st, a, ccfg)
            st2 = jax.tree.map(lambda x, y: jnp.where(done, y, x), st2, st)
            return (st2, c2, done | term | trunc), dict(
                done=done | term | trunc, cause=st2.term_cause, gates=E.gates_passed(st2))

        _, o = jax.lax.scan(body, (st, c, jnp.bool_(False)), None, length=ccfg.base.max_steps)
        k = jnp.minimum(jnp.argmax(o["done"]), o["done"].shape[0] - 1)
        return o["gates"][k], o["cause"][k]

    return jax.jit(jax.vmap(one))
