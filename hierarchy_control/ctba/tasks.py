"""Task names -> environment configs, and the curriculum's stages.

A task is `<stage>_<track>`, e.g. `c1_team44up`.  The track tables (room, floor,
launch, erosion, speed cap ...) are skydreamer.track's; the stage decides which
of the reward terms are on.  DreamerV3 sees the task as `ctba_<stage>_<track>`.

The curriculum (planned before training; README.md):
  c1  complete the track: demonstrations in the buffer, a loose speed cap
      (5 m/s), the centre-of-mass gate window with some leniency (0.60 m train).
  c2  regulate speed: the room's cap (4 m/s, track.V_CAP).
  c3  stay off the floor: + floor-sink penalty and the altitude margin.
  c4  clear gates with the whole airframe: the real gate plane scored and tested
      on the CAD footprint, train window = real window (0.45 m).
  ce  what every stage is scored on: c4's conditions (the real test).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from skydreamer import env as SE
from skydreamer import track as T

from . import airframe as A
from . import params as P

TRACKS = {
    "team": T.team_track,
    "team4up": T.team4up_track,
    "team14up": T.team14up_track,
    "team44up": T.team44up_track,
}

CAM_FOV = (114.6, 92.15)   # deg, the camera the GateNet masks were trained for

STAGES = {
    "c1": dict(v_cap=5.0, d_g=(0.60, 0.45), floor_penalty=False, alt_margin=False, swept=False),
    "c2": dict(v_cap=None, d_g=(0.60, 0.45), floor_penalty=False, alt_margin=False, swept=False),
    "c3": dict(v_cap=None, d_g=(0.60, 0.45), floor_penalty=True, alt_margin=True, swept=False),
    "c4": dict(v_cap=None, d_g=(0.45, 0.45), floor_penalty=True, alt_margin=True, swept=True),
    "ce": dict(v_cap=None, d_g=(0.45, 0.45), floor_penalty=True, alt_margin=True, swept=True),
}

# Per-step reward weights in skydreamer.track are per 90 Hz step; ours are 100 Hz.
STEP_SCALE = T.CONTROL_FREQ / P.CONTROL_FREQ


class CtbaConfig(NamedTuple):
    base: SE.EnvConfig          # track, room, floor, reward switches (skydreamer's)
    stage: str
    track_name: str


def split(task: str) -> tuple[str, str]:
    if task.startswith("ctba_"):
        task = task[len("ctba_"):]
    stage, _, track = task.partition("_")
    if stage not in STAGES or track not in TRACKS:
        raise ValueError(f"unknown task {task!r}: stages {sorted(STAGES)}, tracks {sorted(TRACKS)}")
    return stage, track


def make_config(task: str, image_size: int = 64, max_steps: int = 2000) -> CtbaConfig:
    stage, name = split(task)
    st = STAGES[stage]
    track = TRACKS[name]()
    room = T.ROOMS[name]
    reach = A.reach()
    v_cap = st["v_cap"] if st["v_cap"] is not None else T.V_CAP[name]
    launch = T.LAUNCH[name]
    launch = launch[:4] + (A.rest_altitude(),)
    base = SE.EnvConfig(
        track=track,
        max_steps=max_steps,
        image_size=image_size,
        cam_offset=A.cam_offset(),
        d_g=st["d_g"],
        fov=CAM_FOV,
        room=room,
        v_cap=v_cap,
        w_speed=T.W_SPEED * STEP_SCALE,
        floor=(reach, T.FLOOR[name]),
        launch=launch,
        air_start_alt=T.AIR_START[name],
        w_floor=(T.W_FLOOR_STRONG * STEP_SCALE) if st["floor_penalty"] else 0.0,
        erode_px=T.ERODE_PX[name],
        swept_points=A.footprint().astype(np.float32) if st["swept"] else None,
        w_alt=(T.W_ALT * STEP_SCALE) if st["alt_margin"] else 0.0,
    )
    return CtbaConfig(base=base, stage=stage, track_name=name)
