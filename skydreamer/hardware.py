"""Our flight-test drone and gate, and the swept-footprint check they allow.

Numbers are from docs/hardware.md; `(est.)` there means an estimate here too.
Nothing in training imports this: it feeds `scripts/evaluate.py --protocol
physical` and the `--hw-*` overrides.

Body frame is NED-style, x forward, y right, z down (same as the simulator)."""

from __future__ import annotations

import numpy as np

from .params import NOMINAL

# --- gate (given) ----------------------------------------------------------
GATE_INNER = 1.5      # m, clear opening edge length
GATE_OUTER = 2.1      # m
GATE_THICKNESS = 0.05  # m

# --- drone ------------------------------------------------------------------
DIAGONAL = 0.250      # m, motor centre to motor centre (given; guards unknown)
PROP_R = 0.0635       # m, 5 in (physical radius; the dynamics' own 0.0486 is the
                      # 75% blade station and already matches a 5 in prop)
CAM_FORWARD = 0.10    # m ahead of the CoM (given)
CAM_TIP = 0.12        # m, forward extent of the camera mount (est.)
CAM_FOV = (114.6, 92.15)   # deg, horizontal x vertical (given); 720 x 480 px
TWR_CENTRAL = 4.3     # est.: 4 x 1.024 kgf / 0.8 kg = 5.1 unloaded, x ~0.85 for sag
TWR_RANGE = (3.7, 5.1)

# The simulator's nominal thrust-to-weight, k_w * 4 * w_max^2 / g = 6.07.  Captured
# before anything here touches NOMINAL, so `apply_dynamics` always scales from the
# paper's value and calling it twice does not compound.
PAPER_K_W = NOMINAL["k_w"]
NOMINAL_TWR = PAPER_K_W * 4 * NOMINAL["w_max"] ** 2 / 9.81

# --- what a `hw_*` task changes -----------------------------------------------
# Everything below is derived from the numbers above; see docs/hardware.md.
#   gate_outer  our frame is 2.1 m outside (0.30 m wide), not the paper's 2.7 m.
#   cam_offset  camera 0.10 m ahead of the centre of mass.
#   fov         114.6 x 92.15 deg: the nominal K is a 104.4 deg square, so the columns
#               are all valid and ~6 rows top and bottom are blank (see render.fov_valid).
#   twr         k_w scaled so nominal thrust-to-weight is 4.3; Table III's k_w +-30%
#               and w_max +-20% then give a median ~4.2 and a 10-90% range of ~2.9-6.1
#               (1-99%: 2.3-7.2; thrust scales with k_w w_max^2).  NOT measured -- replace with
#               the static-thrust test when there is one.
#   d_g         centre-of-mass window: the 0.75 m opening half-width less the
#               ~0.15-0.19 m airframe reach.  Train 0.65 is a little lenient, as the
#               paper's 0.8 is against its 0.75; eval 0.60 is the physical window.
#   t_g         pre/post planes at +-0.20 m, i.e. the airframe's reach, so a prop
#               ahead of or behind the centre is held to the same window.
HW_PROFILE = dict(gate_outer=GATE_OUTER, cam_offset=CAM_FORWARD, fov=CAM_FOV, twr=TWR_CENTRAL,
                  d_g=(0.65, 0.60), t_g=0.40)
HW_PREFIX = "hw_"


def apply_dynamics(twr: float | None) -> None:
    """Set the process-wide nominal thrust-to-weight (None = the paper's).

    `NOMINAL` is read when the env is traced and when parameters are sampled, so
    this must run before the first reset in the process -- `SkyDreamer._build`
    does it, once per worker."""
    NOMINAL["k_w"] = PAPER_K_W if twr is None else PAPER_K_W * twr / NOMINAL_TWR


def footprint(prop_r: float = PROP_R, diagonal: float = DIAGONAL, ring: int = 12) -> np.ndarray:
    """(M, 3) body-frame points covering the airframe: each prop disc (centre plus a
    ring at its edge), the four arms, and the camera tip."""
    a = diagonal / 2 / np.sqrt(2)
    centres = np.array([[a, a], [a, -a], [-a, a], [-a, -a]])
    ang = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    disc = np.stack([np.cos(ang), np.sin(ang)], -1) * prop_r
    pts = [np.append(c, 0.0) for c in centres]
    pts += [np.append(c + d, 0.0) for c in centres for d in disc]
    # arms: centre to each motor
    pts += [np.append(c * f, 0.0) for c in centres for f in (0.25, 0.5, 0.75)]
    pts += [[0.0, 0.0, 0.0], [CAM_TIP, 0.0, 0.0]]
    return np.array(pts, float)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Body->world, [w, x, y, z]; the numpy twin of `dynamics.quat_to_matrix`."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z); r[..., 0, 1] = 2 * (x * y - w * z); r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z); r[..., 1, 1] = 1 - 2 * (x * x + z * z); r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y); r[..., 2, 1] = 2 * (y * z + w * x); r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def gate_axes(yaw: float) -> np.ndarray:
    """World->gate rotation, as `track.gate_rotation`: x normal, y right, z down."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def check_flight(pos, quat, gates, n_steps, dt, *, opening=GATE_INNER / 2,
                 virtual_half=1.0, window_s=0.15, near=1.4, points=None):
    """Score one flight against the real gates.

    pos (T,3), quat (T,4): per-step ground truth.  `gates` is a list of
    (centre(3), yaw, visible).  A real gate is passed if every footprint point
    that crosses its plane does so inside the square `opening` (half-width);
    a virtual gate only asks the drone's centre to cross within `virtual_half`.
    The centre's crossing is found first; the points are checked within
    +-`window_s` of it, so a prop that swings through earlier or later than the
    centre is still seen.  Centres crossing more than `near` m from the gate axis
    are another gate's business.

    Returns (ok, list of (step, gate, centre_off, worst_point_off, ok))."""
    pts = footprint() if points is None else points
    pos = np.asarray(pos)[:n_steps]
    quat = np.asarray(quat)[:n_steps]
    W = int(round(window_s / dt))
    events = []
    ok = True
    for gi, (centre, yaw, visible) in enumerate(gates):
        R = gate_axes(yaw)
        loc = (pos - centre) @ R.T            # (T,3) gate frame
        cross = np.nonzero((loc[:-1, 0] < 0) & (loc[1:, 0] >= 0))[0]
        for k in cross:
            f = -loc[k, 0] / (loc[k + 1, 0] - loc[k, 0])
            yz = loc[k, 1:] + f * (loc[k + 1, 1:] - loc[k, 1:])
            off = float(np.max(np.abs(yz)))
            if off > near:
                continue
            if not visible:
                good = off <= virtual_half
                events.append((int(k), gi, off, off, bool(good)))
                ok &= good
                continue
            lo, hi = max(0, k - W), min(len(pos), k + W + 2)
            Rb = quat_to_matrix(quat[lo:hi])                   # (w,3,3)
            world = pos[lo:hi, None, :] + np.einsum("wij,mj->wmi", Rb, pts)
            pl = (world - centre) @ R.T                          # (w,M,3)
            s0, s1 = pl[:-1, :, 0], pl[1:, :, 0]
            hit = (s0 * s1 < 0) | (s1 == 0)
            if not hit.any():
                events.append((int(k), gi, off, off, True))
                continue
            w_i, m_i = np.nonzero(hit)
            ff = -s0[w_i, m_i] / (s1[w_i, m_i] - s0[w_i, m_i])
            yzp = pl[w_i, m_i, 1:] + ff[:, None] * (pl[w_i + 1, m_i, 1:] - pl[w_i, m_i, 1:])
            worst = float(np.max(np.abs(yzp)))
            good = worst <= opening
            events.append((int(k), gi, off, worst, bool(good)))
            ok &= good
    return ok, events
