"""Our airframe's geometry for collision and gate scoring, from the team's CAD
(drone_geometry/drone_design_tilt30.step, 2026-11-18, via drone_geometry.py) and the
hanging-photo centre of mass.

CAD/user frame: x forward, y right, z UP, cm, origin at the motor-diagonal centre
in the prop plane.  Returned points are in the simulator's body frame: FRD
(z down), metres, origin at the centre of mass.
"""
from __future__ import annotations

import numpy as np

MASS = 0.95                       # kg, all-up (hardware_spec.md)
COM = (0.01, 0.05, 1.15)          # cm, user frame (z depends on an unmeasured nut height)
MOTOR_X, MOTOR_Y = 15.66 / 2, 19.75 / 2   # cm, CAD motor holes
PROP_R = 6.35                     # cm, 5 in
CAM = (12.00, 0.12, 0.93)         # cm, lens front face (CAD drone_design_tilt30.step)
CAM_TILT_DEG = 30.0               # up from horizontal, fixed mount (CAD 30.0; training 25-35, ctba.camera)

# CAD cross-sections (drone_geometry.py): x stations, bottom and top z, half width
# (shell, mounts, lid, Jetson housing, camera cover and legs; the frame plate excluded).
XS = [13.56, 12.5, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0, -9.0, -11.0, -13.0, -15.0, -16.01]
BOT = [-0.56, -1.95, -2.73, -3.04, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -3.23, -0.41, -0.24, -0.12, 0.32]
TOP = [-0.52, 0.84, 2.08, 3.41, 4.14, 4.86, 5.59, 5.61, 5.69, 5.39, 4.74, 4.02, 3.30, 2.57, 1.85, 1.13, 0.47]
HALFW = [0.06, 1.99, 2.95, 3.30, 5.32, 5.68, 3.90, 3.91, 3.71, 3.52, 3.32, 3.12, 2.92, 2.59, 2.10, 1.22, 0.03]
STAND_X = (8.04, -8.16)           # cm, front / rear floor-contact centres (CAD)
STAND_Y = 4.0
STAND_Z = -4.3                    # stand bottoms (floor contact); measured 2026-11-18 (= CAD tilt30 legs)


def _to_body(pts_cm: np.ndarray) -> np.ndarray:
    """user frame cm -> FRD metres about the CoM."""
    c = np.asarray(COM)
    d = (np.asarray(pts_cm, float) - c) / 100.0
    return np.stack([d[:, 0], d[:, 1], -d[:, 2]], 1)


def footprint(ring: int = 12) -> np.ndarray:
    """(M, 3) body points covering props, arms, shell, stands and camera."""
    pts = []
    ang = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    for sx in (1, -1):
        for sy in (1, -1):
            cx, cy = sx * MOTOR_X, sy * MOTOR_Y
            pts.append((cx, cy, 0.0))
            pts += [(cx + PROP_R * np.cos(a), cy + PROP_R * np.sin(a), 0.0) for a in ang]
            pts += [(cx * f, cy * f, 0.0) for f in (0.25, 0.5, 0.75)]
    for x, b, t, h in zip(XS, BOT, TOP, HALFW):
        mid = (b + t) / 2
        pts += [(x, 0.0, b), (x, 0.0, t), (x, h, mid), (x, -h, mid)]
    for x in STAND_X:
        for y in (STAND_Y, -STAND_Y):
            pts.append((x, y, STAND_Z))
    pts.append(CAM)
    return _to_body(np.array(pts))


def reach() -> float:
    """Horizontal distance from the CoM to the furthest airframe point, m."""
    f = footprint()
    return float(np.max(np.hypot(f[:, 0], f[:, 1])))


def cam_offset() -> float:
    """Camera ahead of the CoM along body x, m."""
    return (CAM[0] - COM[0]) / 100.0


def rest_altitude() -> float:
    """CoM height when standing on the floor, m."""
    return (COM[2] - STAND_Z) / 100.0
