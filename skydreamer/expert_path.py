"""Reference path and speed profile for the demonstration controller.

NOT FROM THE PAPER.  SkyDreamer trains end-to-end from scratch; this module
exists because that did not work here (see docs/deviations.md).  Its only job
is to describe a *flyable* lap -- a geometric path through the gates plus a
speed along it that the drone's 6 g of thrust can actually produce -- so that
`expert.py` can fly it and put successful laps into the replay buffer.

Everything here is host-side numpy, computed once before training starts.  The
controller that tracks it is JAX and lives in `expert.py`.

The path is a periodic cubic Hermite spline through the gate centres with the
tangent at each gate forced to the gate normal, so the drone necessarily
crosses every gate plane travelling the intended direction.  The speed profile
is the standard forward-backward pass: start from the curvature limit
`v <= sqrt(a_lat / kappa)`, then sweep forward and backward enforcing the
longitudinal acceleration limit, which is what makes the result trackable
rather than merely feasible pointwise.
"""

from __future__ import annotations

import numpy as np

from . import track as trk

# Fraction of the drone's measured envelope the reference is allowed to use.
# The controller needs headroom on top of the reference: tracking error,
# the 11.1 ms action delay and the 30 ms motor lag all have to be paid for out
# of whatever thrust the reference did not already spend.  0.6 leaves 40%.
ENVELOPE_MARGIN = 0.6


SMOOTH_M = 0.30   # smoothing length along the path, metres


def _smooth(x: np.ndarray, ds: float, length: float = SMOOTH_M) -> np.ndarray:
    """Periodic Gaussian blur along the path.  `length` is well under the
    controller's own bandwidth (0.3 m is 40 ms at 7 m/s against loops that
    close in 200 ms), so it removes sampling noise without removing anything
    the drone could have tracked."""
    sigma = max(length / max(ds, 1e-9) / 3.0, 1e-6)
    half = max(int(np.ceil(3 * sigma)), 1)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([x[-half:], x, x[:half]], 0)
    if x.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid")
                     for j in range(x.shape[1])], -1)


def gate_normals(track: trk.Track) -> np.ndarray:
    """Unit travel direction at each gate (= +x of the gate frame)."""
    yaw = np.asarray(track.yaw)
    return np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], -1)


def build_path(track: trk.Track, samples_per_gate: int = 400,
               tangent_scale: float = 1.4, lead: float = 0.0) -> dict:
    """Periodic Hermite spline through the gate centres, resampled uniformly in
    arc length.

    `tangent_scale` sets how hard the curve commits to its tangent before
    bending toward the next waypoint.  1.0 (tangent magnitude = chord length)
    is the Catmull-Rom-like default.

    `lead` is why this is not simply a spline through the gate centres.  A gate
    forces its tangent to the gate normal, and with the next gate pulling
    sideways the curvature spikes *at the gate* -- on the inverted loop that
    drove the minimum radius to 0.4 m no matter how the tangents were scaled,
    because two of the three gates face the same way 5 m apart and the turn has
    nowhere to go but into the gate crossing.  Adding a waypoint `lead` metres
    before and after each gate, with free (Catmull-Rom) tangents, makes the
    crossing itself straight and moves the turn into the gap between gates,
    where it has room.

    It defaults to 0 (off) because on the inverted loop it *loses*: the lead
    segments lengthen the path and leave the turn a shorter window, taking the
    minimum radius from 0.39 m down to 0.21 m.  It is kept for the big track,
    where the gates are 6 m apart and the turns have room to use it.
    """
    P = np.asarray(track.pos, dtype=np.float64)
    N = gate_normals(track).astype(np.float64)
    n = len(P)

    # Waypoints: for each gate, lead-in / gate / lead-out.  All three carry the
    # *same* forced tangent, the gate normal, which makes the crossing a
    # straight `2 * lead` segment aimed through the gate.  Leaving the lead
    # points' tangents free instead is worse than having no lead points at all:
    # a Catmull-Rom tangent there points at the *next gate*, which on this
    # track is roughly opposite to the direction the gate 1.2 m later forces,
    # so the spline cusps and the minimum radius collapses to 0.06 m.
    pts, fixed = [], []
    for i in range(n):
        if lead > 0:
            pts.append(P[i] - N[i] * lead); fixed.append(N[i])
        pts.append(P[i]); fixed.append(N[i])
        if lead > 0:
            pts.append(P[i] + N[i] * lead); fixed.append(N[i])
    pts = np.array(pts)
    m = len(pts)

    tangents = np.zeros_like(pts)
    for i in range(m):
        if fixed[i] is not None:
            tangents[i] = fixed[i]
        else:
            d = pts[(i + 1) % m] - pts[(i - 1) % m]
            tangents[i] = d / max(np.linalg.norm(d), 1e-9)

    # Dense sample of each segment, then one global arc-length resample.
    raw = []
    per = max(int(samples_per_gate * n / m), 8)
    for i in range(m):
        p0, p1 = pts[i], pts[(i + 1) % m]
        chord = np.linalg.norm(p1 - p0)
        m0 = tangents[i] * chord * tangent_scale
        m1 = tangents[(i + 1) % m] * chord * tangent_scale
        t = np.linspace(0.0, 1.0, per, endpoint=False)[:, None]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        raw.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    raw = np.concatenate(raw, 0)

    # Arc length of the raw samples, then resample at uniform spacing so that
    # every downstream quantity (curvature, the speed profile's `ds`) is on a
    # regular grid.
    d = np.linalg.norm(np.diff(np.vstack([raw, raw[:1]]), axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s_raw[-1])
    m = len(raw)
    s = np.linspace(0.0, total, m, endpoint=False)
    p = np.stack([np.interp(s, s_raw[:-1], raw[:, k], period=total)
                  for k in range(3)], -1)

    # Periodic central differences on the uniform grid.
    ds = total / m
    dp = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / (2 * ds)
    d2p = (np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)) / ds**2
    speed = np.linalg.norm(dp, axis=1, keepdims=True)
    tangent = dp / np.maximum(speed, 1e-9)
    # kappa = |r' x r''| / |r'|^3, valid for any parameterisation.
    kappa = (np.linalg.norm(np.cross(dp, d2p), axis=1)
             / np.maximum(speed[:, 0] ** 3, 1e-9))

    # Where each gate centre falls along the path, so the controller can report
    # progress and `collect.py` can check the plane order.
    gate_s = np.array([float(s[np.argmin(np.linalg.norm(p - P[i], axis=1))])
                       for i in range(n)])

    return dict(p=p, tangent=tangent, kappa=kappa, s=s, ds=ds,
                total=total, gate_s=gate_s, gate_pos=P, gate_normal=N)


def envelope(params) -> tuple[float, float]:
    """(lateral, longitudinal) acceleration the drone can actually produce, m/s^2.

    Thrust is `k_w * sum(w^2)` with `sum(w^2) <= 4 w_max^2`, and gravity eats
    1 g of it in level flight.  The advance-ratio term adds a few percent more
    at speed; ignoring it is the conservative direction.
    """
    a_max = float(params["k_w"]) * 4.0 * float(params["w_max"]) ** 2
    g = 9.81
    a_lat = float(np.sqrt(max(a_max**2 - g**2, 0.0)))
    return a_lat, a_max - g


def speed_profile(path: dict, a_lat: float, a_long: float,
                  v_max: float = 25.0, v_min: float = 1.0,
                  passes: int = 4, v_seed: np.ndarray | None = None) -> np.ndarray:
    """Fastest speed along the path that respects both acceleration limits.

    Pointwise the curvature limit is `v <= sqrt(a_lat / kappa)`.  That profile
    is not trackable on its own: it can demand a step change in speed at a
    curvature discontinuity.  The forward and backward sweeps fix that by
    enforcing `v_{i+1}^2 <= v_i^2 + 2 a_long ds` in both directions, which is
    what turns a pointwise bound into a trajectory.  The path is a closed loop,
    so the sweeps are repeated until they stop changing.
    """
    ds = path["ds"]
    v = np.minimum(np.sqrt(a_lat / np.maximum(path["kappa"], 1e-9)), v_max)
    if v_seed is not None:
        v = np.minimum(v, v_seed)
    v = np.maximum(v, v_min)
    for _ in range(passes):
        for i in range(len(v)):  # forward, wrapping
            j = (i + 1) % len(v)
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_long * ds))
        for i in range(len(v) - 1, -1, -1):  # backward, wrapping
            j = (i - 1) % len(v)
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_long * ds))
    return v


def _derivatives(path: dict, v: np.ndarray) -> dict:
    """Velocity and acceleration the reference demands, as world vectors.

    Differentiate the velocity vector, smoothed, rather than assembling
    `a = (dv/dt) T + v^2 kappa N` from the Frenet frame.  The Frenet form is
    the textbook one and it is unusable here for two separate reasons: kappa
    comes from differencing an interpolated path twice, so it jitters by 7%
    sample to sample, and the principal normal N flips sign outright at every
    inflection, where the curve is locally straight and N is undefined.  Both
    land in the acceleration -- it moved 2.2 m/s^2 between adjacent samples,
    the implied thrust direction jumped up to 90 deg, and the body rate that
    would hold it came out at 68 rad/s average against a drone that can do
    about 17.
    """
    path["v"] = v
    Vs = _smooth(v[:, None] * path["tangent"], path["ds"])
    dVds = (np.roll(Vs, -1, 0) - np.roll(Vs, 1, 0)) / (2 * path["ds"])
    path["accel"] = _smooth(dVds * v[:, None], path["ds"])
    path["a_tangential"] = np.sum(path["accel"] * path["tangent"], 1)
    return path


def reference(track: trk.Track, params, margin: float = ENVELOPE_MARGIN,
              v_cap: float | None = None, omega_max: float = 10.0,
              iters: int = 4, heading_ahead: float = 3.0, **kw) -> dict:
    """Path, speed profile, and the attitude and body rate that fly it.

    `omega_max` is a limit on the reference's own rotation, enforced the same
    way the lateral acceleration limit is: where the attitude would have to
    turn faster than the drone can manage, slow down there and re-run the
    profile.  It is not a refinement.  Two places on the inverted loop demand
    an unbounded rotation at any speed -- directly above the gate being aimed
    at, where the horizontal bearing to it sweeps through everything, and
    wherever the reference acceleration passes through zero, where the thrust
    direction is free to swing.  Both are geometric, so no amount of smoothing
    removes them; flying slower through them does, because for a fixed path the
    rotation rate scales with speed.
    """
    path = build_path(track, **kw)
    a_lat, a_long = envelope(params)
    v = speed_profile(path, a_lat * margin, a_long * margin,
                      v_max=v_cap if v_cap is not None else 25.0)
    for _ in range(iters):
        attitude_reference(_derivatives(path, v), heading_ahead=heading_ahead)
        w = np.linalg.norm(path["omega_ff"], axis=1)
        if w.max() <= omega_max * 1.02:
            break
        # Scale down where it is too fast, then let the forward-backward sweep
        # spread the slowdown over a trackable distance.
        v = np.minimum(v, v * omega_max / np.maximum(w, 1e-6))
        v = speed_profile(path, a_lat * margin, a_long * margin,
                          v_max=v_cap if v_cap is not None else 25.0, v_seed=v)
    path["a_lat_limit"] = a_lat
    path["a_long_limit"] = a_long
    path["omega_max"] = omega_max
    path["lap_time"] = float(np.sum(path["ds"] / path["v"]))
    return path


def attitude_reference(path: dict, heading_ahead: float = 3.0,
                       switch_ahead: float = 1.5) -> dict:
    """The attitude the reference implies, and the body rate that holds it.

    Without this the controller has to *discover* every rotation through
    attitude error.  That cannot work here: the split-S turns the thrust vector
    through 180 deg in about half a second, which is 6 rad/s, while the
    attitude loop is capped near 5 rad/s by the 30 ms motor lag.  The loop
    permanently lags, the thrust points somewhere other than where the path
    needs it, and the drone leaves the track -- which is exactly the 30 m
    tracking error this started with.  Feeding the rotation forward leaves the
    loop only the error to correct.

    The heading points at the gate being flown to, not along the path.  A
    quadrotor is omnidirectional in translation, so yaw is free, and the only
    thing that wants it is the camera -- the policy's sole view of the world is
    the gate mask, so a demonstration is worthless if the gate is out of frame.
    Tying yaw to the direction of travel fails outright on this track: between
    the two gates that face the same way the path doubles back, the horizontal
    travel direction reverses, and the reference swings 170 deg in 39 ms.

    Three details keep that bearing usable.  It switches to the next gate
    `switch_ahead` metres *before* reaching the current one, because the
    bearing to a gate you are about to fly through sweeps through every angle
    as you pass over it.  It is unwrapped and smoothed as an angle rather than
    as a vector, since the switch is a step and blurring a direction vector
    just concentrates it.  And the whole-lap turn is taken out before the
    periodic filter and put back after, because the heading accumulates a whole
    number of turns per lap and is not periodic.
    """
    g = np.array([0.0, 0.0, 9.81])
    a_thrust = path["accel"] - g
    T = np.maximum(np.linalg.norm(a_thrust, axis=1, keepdims=True), 1e-6)
    z_B = -a_thrust / T

    s, gate_s, gp = path["s"], path["gate_s"], path["gate_pos"]
    order = np.argsort(gate_s)
    nxt = np.searchsorted(gate_s[order],
                          (s + switch_ahead) % path["total"], side="left") % len(gate_s)
    d = gp[order[nxt]] - path["p"]

    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = psi[-1] - psi[0] + (psi[-1] - psi[-2])
    lin = np.linspace(0.0, turn, len(psi), endpoint=False)
    psi = _smooth(psi - lin, path["ds"], heading_ahead) + lin
    x_C = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], -1)

    y_B = np.cross(z_B, x_C)
    ny = np.linalg.norm(y_B, axis=1, keepdims=True)
    y_B = np.where(ny > 1e-3, y_B / np.maximum(ny, 1e-9), np.array([0.0, 1.0, 0.0]))
    x_B = np.cross(y_B, z_B)
    R = np.stack([x_B, y_B, z_B], axis=-1)          # (S, 3, 3), columns

    # omega = vee(R^T dR/dt), and dR/dt = (dR/ds) v.
    dRds = (np.roll(R, -1, 0) - np.roll(R, 1, 0)) / (2 * path["ds"])
    S = np.einsum("sji,sjk->sik", R, dRds)
    omega = np.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], -1) * path["v"][:, None]

    path["R_des"] = R
    path["omega_ff"] = omega
    path["thrust_ref"] = T[:, 0]
    path["psi_ref"] = psi
    return path
