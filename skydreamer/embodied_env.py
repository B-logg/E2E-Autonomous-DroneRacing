"""`embodied.Env` adapter so DreamerV3 can drive the JAX racing environment.

`patches/informed_dreamer.patch` registers this class in DreamerV3's
`make_env` constructor table (a hardcoded dict in `dreamerv3/main.py` -- adding
to `embodied/envs/__init__.py` does nothing).  All you need is for the
`skydreamer` package to be importable, i.e. `pip install -e .`.

`main.py` splits `task` on the first underscore, so `skydreamer_inverted_loop`
arrives here as `inverted_loop`.

One process drives one environment, which is what `embodied` expects; run
`--run.envs 16` to get parallelism.  Every worker must get its own `seed` or
all of them replay the same trajectory -- that is what `use_seed: True` in the
env config is for.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import track as T
from .env import EnvConfig, reset, step

TRACKS = {
    "inverted_loop": T.inverted_loop,
    "ladder_inverted_loop": T.ladder_inverted_loop,
}


def _spaces():
    import elements

    return elements


class SkyDreamer:
    """Observation keys without a prefix are `o_t`; `info_*` keys are `i_t`
    and are consumed only by the (informed) decoder."""

    def __init__(self, task="inverted_loop", size=64, max_steps=2000, seed=0):
        if task.startswith("skydreamer_"):  # tolerate an unsplit task name
            task = task.split("_", 1)[1]
        if task == "big":
            raise NotImplementedError(
                "the big track (paper Figure 9) has no published gate coordinates "
                "and is not reconstructed; see docs/paper_gaps.md B1"
            )
        if task not in TRACKS:
            raise ValueError(f"unknown track {task!r}; have {sorted(TRACKS)}")

        # Construction copies the track and the RNG seed to the device, both of
        # which the process-wide `jax_transfer_guard='disallow'` DreamerV3 sets
        # would otherwise reject.
        with jax.transfer_guard("allow"):
            self.cfg = EnvConfig(
                track=TRACKS[task](), max_steps=int(max_steps), image_size=int(size)
            )
            self._rng = jax.random.key(seed)
        self._reset = jax.jit(functools.partial(reset, cfg=self.cfg))
        self._step = jax.jit(functools.partial(step, cfg=self.cfg))
        self._state = None
        self._done = True

    # -- spaces ------------------------------------------------------------

    @functools.cached_property
    def obs_space(self):
        el = _spaces()
        with jax.transfer_guard("allow"):
            _, obs = self._reset(jax.random.key(0))
        out = {}
        for k, v in obs.items():
            if k == "mask":
                out[k] = el.Space(np.uint8, tuple(v.shape))
            else:
                out[k] = el.Space(np.float32, tuple(v.shape))
        out.update(
            reward=el.Space(np.float32),
            is_first=el.Space(bool),
            is_last=el.Space(bool),
            is_terminal=el.Space(bool),
        )
        return out

    @property
    def act_space(self):
        el = _spaces()
        # Normalized motor commands, one per rotor, straight to the ESC.
        return {
            "reset": el.Space(bool),
            "action": el.Space(np.float32, (4,), 0.0, 1.0),
        }

    # -- stepping ----------------------------------------------------------

    def step(self, action):
        # The env is the host<->device boundary: actions come in as numpy and
        # observations go out as numpy.  DreamerV3 disallows transfers
        # process-wide, so permit them here and nowhere else.
        with jax.transfer_guard("allow"):
            return self._step_guarded(action)

    def _step_guarded(self, action):
        if action["reset"] or self._done:
            self._rng, sub = jax.random.split(self._rng)
            self._state, obs = self._reset(sub)
            self._done = False
            return self._pack(obs, 0.0, is_first=True)

        self._state, obs, reward, terminal, truncated = self._step(
            self._state, jnp.asarray(action["action"], jnp.float32)
        )
        terminal, truncated = bool(terminal), bool(truncated)
        self._done = terminal or truncated
        return self._pack(
            obs,
            float(reward),
            is_last=self._done,
            # Truncation is a time limit, not a failure: bootstrapping must
            # continue through it or the value function learns that surviving
            # to 2000 steps is worthless.
            is_terminal=terminal,
        )

    def _pack(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        out = {}
        for k, v in obs.items():
            v = np.asarray(v)
            out[k] = (v > 0.5).astype(np.uint8) * 255 if k == "mask" else v.astype(np.float32)
        out.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return out

    def close(self):
        pass


def make(task, **kwargs):
    return SkyDreamer(task=task, **kwargs)
