"""Centre of mass from hanging photos.

Each photo: the drone hangs on a string; the string's line in the image and the
camera centre span a plane that, in 3D, contains the string and therefore the CoM.
The drone's pose in each photo comes from the 4 motor hubs (known body points)
by PnP with the iPhone's intrinsics.  Planes from several photos (two hanging
points, many view angles) are intersected by least squares.

Body frame for input and output = drone_geometry.py: origin at the centre of the
motor diagonals, prop plane; x forward, y right, z UP, cm.  That frame is
left-handed, so the PnP works internally in FLU (x forward, y LEFT, z up) and flips
y on the way in and out.  (An earlier run labelled the photos' left and right
motors the wrong way round; measurements.json is corrected.)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

W, H = 1200, 1600                      # upright, downscaled frames
F = 24 / 43.27 * np.hypot(W, H)        # 24 mm equivalent focal (EXIF) -> px
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
MX, MY = 7.83, 9.88                    # CAD (Part Studio 1-2.step) motor centre holes; was tape 7.75, 10.25
MOTOR = {"FR": (MX, -MY), "FL": (MX, MY), "RR": (-MX, -MY), "RL": (-MX, MY)}   # FLU


def project(rvec, t, X):
    Xc = Rotation.from_rotvec(rvec).apply(X) + t
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3]


def pose(obs):
    """obs: list of (body xyz, image uv).  Returns R (body->cam), t, rms px."""
    X = np.array([o[0] for o in obs], float)
    U = np.array([o[1] for o in obs], float)
    best = None
    for seed in Rotation.random(64, random_state=0).as_rotvec():
        r = least_squares(lambda p: (project(p[:3], p[3:], X) - U).ravel(),
                          np.r_[seed, 0, 0, 60.0], method="lm")
        if r.x[5] > 0 and (best is None or r.cost < best.cost):
            best = r
    rms = np.sqrt(2 * best.cost / len(X))
    return Rotation.from_rotvec(best.x[:3]).as_matrix(), best.x[3:], rms


def plane(meas):
    """Body-frame plane (n, d) with n.c = d containing the string."""
    obs = [((*MOTOR[k], meas["hub_z"]), uv) for k, uv in meas["motors"].items()]
    R, t, rms = pose(obs)
    a, b = (np.linalg.inv(K) @ np.r_[p, 1.0] for p in meas["string"])
    n_c = np.cross(a, b)
    n_c /= np.linalg.norm(n_c)
    n_b = R.T @ n_c                      # plane through the camera centre
    o_b = -R.T @ t                       # camera centre in body frame
    return n_b, n_b @ o_b, rms, o_b


def main(path):
    data = json.loads(Path(path).read_text())
    rows = []
    for name, m in data.items():
        n, d, rms, cam = plane(m)
        rows.append((name, m["hang"], n, d, rms, cam))
        print(f"{name}: hang {m['hang']}, PnP rms {rms:.1f} px, camera at "
              f"({cam[0]:+.0f}, {-cam[1]:+.0f}, {cam[2]:+.0f}) cm (x fwd, y right, z up), n = {np.round(n, 3)}")
    N = np.array([r[2] for r in rows])
    D = np.array([r[3] for r in rows])
    c, *_ = np.linalg.lstsq(N, D, rcond=None)
    res = N @ c - D
    c = c * np.array([1, -1, 1])          # FLU -> x fwd, y right, z up
    print(f"\nCoM = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) cm   "
          f"(plane residuals: {np.round(res, 2)})")
    # leave-one-out spread as an uncertainty estimate
    loo = []
    for i in range(len(rows)):
        m = np.ones(len(rows), bool); m[i] = False
        loo.append(np.linalg.lstsq(N[m], D[m], rcond=None)[0] * np.array([1, -1, 1]))
    loo = np.array(loo)
    print(f"leave-one-out range: x {loo[:,0].min():+.2f}..{loo[:,0].max():+.2f}, "
          f"y {loo[:,1].min():+.2f}..{loo[:,1].max():+.2f}, z {loo[:,2].min():+.2f}..{loo[:,2].max():+.2f}")
    sv = np.linalg.svd(N, compute_uv=False)
    print(f"conditioning (singular values of plane normals): {np.round(sv, 2)}")
    return c, loo


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("measurements.json"))
