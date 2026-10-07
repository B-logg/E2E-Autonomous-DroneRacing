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

Two things this class is careful about on a GPU box, neither of which shows up
in a CPU smoke test:

1. **The environment computes on CPU, explicitly.**  `embodied`'s Driver spawns
   one process per environment (portal uses a spawn context), and each one
   re-imports this module.  Left alone, sixteen workers would each create their
   own CUDA context and take VRAM away from the trainer for a 64x64 raycast
   that belongs on CPU anyway.

2. **No JAX work happens until the environment is first used.**  DreamerV3's
   `make_agent` builds an environment to read its spaces *before* it constructs
   the Agent, and it is the Agent that calls `embodied.jax.setup(platform=...)`.
   If we initialised a JAX backend in `__init__`, the trainer's backend would be
   chosen before `setup` ever ran.  So the spaces are declared statically and
   the track, RNG and jitted functions are built lazily.
"""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import hardware, track as T
from .env import EnvConfig, reset, step
from .params import (
    BIG_TRACK_W_MAX_DISTURBANCE,
    BIG_TRACK_W_MAX_RESAMPLE_EVERY,
    NOMINAL,
)

TRACKS = {
    "inverted_loop": T.inverted_loop,
    "ladder_inverted_loop": T.ladder_inverted_loop,
    # The paper's Figure 9, measured off the dimensioned panels on page 14 of
    # the PDF.  Nothing trains on it; it is the zero-shot layout test.
    "big": T.big_track,
}

# `hw_<track>`: the same layout on our drone and gate (skydreamer/hardware.py,
# docs/hardware.md).  A task name rather than a flag because the name travels --
# through the config the spawned workers read, into config.yaml, and so into
# evaluate.py and visualize.py without any of them needing to be told.
for _name in list(TRACKS):
    TRACKS[hardware.HW_PREFIX + _name] = TRACKS[_name]


def _split_hw(task: str):
    """`('hw_big')` -> `(True, 'big')`."""
    if task.startswith(hardware.HW_PREFIX):
        return True, task[len(hardware.HW_PREFIX):]
    return False, task


# Per-track tunnel size.  Table III's 0.8 m applies unless the paper overrides
# it: section III-B trains the ladder inverted loop at 0.3 m.
TRACK_T_G = {
    "inverted_loop": None,  # Table III default, 0.8 m
    "ladder_inverted_loop": 0.3,
    "big": 0.5,             # III-D uses t_g = 0.5 for the big track
}
TRACK_T_G.update({hardware.HW_PREFIX + k: hardware.HW_PROFILE["t_g"] for k in list(TRACK_T_G)})

# III-D, big track only: "introduce disturbances of +-300 rad/s to w_max,
# randomly resampled every 10 timesteps, to account for imperfect actuator
# response modeling".
TRACK_W_MAX_DISTURBANCE = {
    "inverted_loop": 0.0,
    "ladder_inverted_loop": 0.0,
    "big": BIG_TRACK_W_MAX_DISTURBANCE,
}
TRACK_W_MAX_DISTURBANCE.update(
    {hardware.HW_PREFIX + k: v for k, v in list(TRACK_W_MAX_DISTURBANCE.items())})


def _spaces():
    import elements

    return elements


def _cpu():
    """The device the environment runs on.  Falls back to the default device if
    no CPU backend is present, which should not happen but is not worth
    crashing over."""
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


class SkyDreamer:
    """Observation keys without a prefix are `o_t`; `info_*` keys are `i_t`
    and are consumed only by the (informed) decoder."""

    def __init__(self, task="inverted_loop", size=64, max_steps=2000, seed=0):
        if task.startswith("skydreamer_"):  # tolerate an unsplit task name
            task = task.split("_", 1)[1]
        if task not in TRACKS:
            raise ValueError(f"unknown track {task!r}; have {sorted(TRACKS)}")

        self._task = task
        self._size = int(size)
        self._max_steps = int(max_steps)
        self._seed = int(seed)
        self._built = False
        self._state = None
        self._done = True

    def _build(self):
        """Construct the track, RNG and jitted functions, pinned to CPU.

        Deliberately not done in `__init__`: see the module docstring."""
        if self._built:
            return
        device = _cpu()
        ctx = jax.default_device(device) if device else contextlib.nullcontext()
        # Construction copies the track and the RNG seed to the device, both of
        # which the process-wide `jax_transfer_guard='disallow'` DreamerV3 sets
        # would otherwise reject.
        is_hw, _ = _split_hw(self._task)
        # `NOMINAL` is process-global and read at trace time, so it is set (or
        # restored) before anything is jitted.  Restoring matters: a test process
        # that built an `hw_*` env must not leave the paper's env on our thrust.
        hardware.apply_dynamics(hardware.HW_PROFILE["twr"] if is_hw else None)
        with ctx, jax.transfer_guard("allow"):
            track = TRACKS[self._task]()
            extra = {}
            if is_hw:
                hw = hardware.HW_PROFILE
                track = track._replace(outer=jnp.full(track.outer.shape, hw["gate_outer"],
                                                      track.outer.dtype))
                extra = dict(cam_offset=hw["cam_offset"], d_g=hw["d_g"], fov=hw["fov"])
            self._cfg = EnvConfig(
                track=track,
                max_steps=self._max_steps,
                image_size=self._size,
                t_g=TRACK_T_G[self._task],
                w_max_disturbance=TRACK_W_MAX_DISTURBANCE[self._task],
                w_max_resample_every=BIG_TRACK_W_MAX_RESAMPLE_EVERY,
                **extra,
            )
            self._rng = jax.random.key(self._seed)
        self._device_ctx = (lambda: jax.default_device(device)) if device else (
            lambda: contextlib.nullcontext())
        self._reset = jax.jit(functools.partial(reset, cfg=self._cfg))
        self._step = jax.jit(functools.partial(step, cfg=self._cfg))
        self._built = True

    @property
    def cfg(self) -> EnvConfig:
        """Building on access keeps callers (scripts/evaluate.py) from having to
        know about the laziness."""
        self._build()
        return self._cfg

    # -- spaces ------------------------------------------------------------

    @functools.cached_property
    def obs_space(self):
        """Declared statically rather than by running a reset, so that reading
        the spaces cannot initialise a JAX backend.  `test_obs_space_matches_a
        _real_reset` pins these against the environment's actual output."""
        el = _spaces()
        n = self._size
        vec = lambda d: el.Space(np.float32, (d,))  # noqa: E731
        from .env import PACK_MASK

        # Bit-packed: eight pixels per byte.  The agent's `packbits` config
        # names this key and expands it back to (n, n, 1) before the encoder,
        # so what the CNN sees is unchanged.
        mask_shape = (n, n // 8, 1) if PACK_MASK else (n, n, 1)
        out = {
            "mask": el.Space(np.uint8, mask_shape),
            "rates": vec(3),
            "rpm": vec(4),
            "flight_plan": vec(T.FLIGHT_PLAN_DIM),
            "info_p_w": vec(3),
            "info_p_g": vec(3),
            "info_v_w": vec(3),
            "info_v_g": vec(3),
            "info_att": vec(4),
            "info_rates": vec(3),
            "info_rpm": vec(4),
            "info_cam": vec(3),
            "info_dyn": vec(len(NOMINAL)),
            "info_meas_rates": vec(3),
            "info_meas_rpm": vec(4),
            "info_flight_plan": vec(T.FLIGHT_PLAN_DIM),
        }
        # `log/*` never reaches the agent -- embodied strips it in the driver
        # and again in the replay buffer -- but the spaces still have to
        # declare it, and run/train.py aggregates it per episode.
        out.update({f"log/{k}": el.Space(np.float32) for k in (
            "diverged", "hit_gate", "hit_ground", "rate_l1", "speed")})
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
        self._build()
        with self._device_ctx(), jax.transfer_guard("allow"):
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
            # the env already emits `mask` as uint8 (packed or 0/255)
            out[k] = v.astype(np.uint8) if k == "mask" else v.astype(np.float32)
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
