"""`embodied.Env` adapter: DreamerV3 drives ctba.env.

Registered as suite `ctba` in DreamerV3's `make_env` by scripts/patch_dreamerv3.py,
so `--task ctba_c1_team44up` arrives here as `c1_team44up`.

Same two precautions as skydreamer.embodied_env (read its docstring): the env
computes on CPU, and nothing touches JAX until the first step, so DreamerV3's
trainer chooses its backend before any worker does.
"""
from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import env as E
from . import tasks


def _cpu():
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


class Ctba:
    def __init__(self, task="c1_team44up", size=64, max_steps=2000, seed=0):
        tasks.split(task)                       # fail early on a bad name
        self._task = task[len("ctba_"):] if task.startswith("ctba_") else task
        self._size, self._max_steps, self._seed = int(size), int(max_steps), int(seed)
        self._built, self._state, self._done = False, None, True

    def _build(self):
        if self._built:
            return
        device = _cpu()
        ctx = jax.default_device(device) if device else contextlib.nullcontext()
        with ctx, jax.transfer_guard("allow"):
            self._cfg = tasks.make_config(self._task, self._size, self._max_steps)
            self._rng = jax.random.key(self._seed)
        self._device_ctx = (lambda: jax.default_device(device)) if device else contextlib.nullcontext
        self._reset = jax.jit(functools.partial(E.reset, ccfg=self._cfg))
        self._step = jax.jit(functools.partial(E.step, ccfg=self._cfg))
        self._built = True

    @property
    def cfg(self):
        self._build()
        return self._cfg

    @functools.cached_property
    def obs_space(self):
        import elements as el
        n = self._size
        mask_shape = (n, n // 8, 1) if E.PACK_MASK else (n, n, 1)
        out = {"mask": el.Space(np.uint8, mask_shape)}
        out.update({k: el.Space(np.float32, (d,)) for k, d in E.OBS_DIMS.items()})
        out.update({f"log/{k}": el.Space(np.float32) for k in E.LOG_KEYS})
        out.update(reward=el.Space(np.float32), is_first=el.Space(bool),
                   is_last=el.Space(bool), is_terminal=el.Space(bool))
        return out

    @property
    def act_space(self):
        import elements as el
        # (throttle, roll, pitch, yaw rate), each in [-1, 1]; ctba.env.action_to_command scales them
        return {"reset": el.Space(bool), "action": el.Space(np.float32, (4,), -1.0, 1.0)}

    def step(self, action):
        self._build()
        with self._device_ctx(), jax.transfer_guard("allow"):
            if action["reset"] or self._done:
                self._rng, sub = jax.random.split(self._rng)
                self._state, obs = self._reset(sub)
                self._done = False
                return self._pack(obs, 0.0, is_first=True)
            self._state, obs, reward, term, trunc = self._step(
                self._state, jnp.asarray(action["action"], jnp.float32))
            term, trunc = bool(term), bool(trunc)
            self._done = term or trunc
            return self._pack(obs, float(reward), is_last=self._done, is_terminal=term)

    def _pack(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        out = {k: (np.asarray(v).astype(np.uint8) if k == "mask" else np.asarray(v).astype(np.float32))
               for k, v in obs.items()}
        out.update(reward=np.float32(reward), is_first=is_first, is_last=is_last,
                   is_terminal=is_terminal)
        return out

    def close(self):
        pass
