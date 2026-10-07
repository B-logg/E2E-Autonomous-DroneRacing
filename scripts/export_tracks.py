#!/usr/bin/env python3
"""Dump the track definitions to JSON and YAML.

The tracks live in `skydreamer/track.py` as Python, because that is what the
simulator consumes and there is no sense in having it parse a data file at
import time.  But the gate coordinates are also the thing you need when you go
and build the physical track, or when you want to check the layout without
reading code, so they are exported here.

Generated, never hand-edited: a second hand-maintained copy of the same numbers
drifts from the first, and then nobody knows which one the policy actually
flew.  `tests/` checks the committed files still match `track.py`.

Usage
  python scripts/export_tracks.py            # rewrite tracks/tracks.{json,yaml}
  python scripts/export_tracks.py --check    # verify they are up to date
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONVENTIONS = {
    "frame": (
        "Aerospace NED. x north, y east, z DOWN, so altitude = -z. "
        "The origin is the track centre as drawn in the paper's figures."
    ),
    "position_m": "[x, y, z] in metres, gate centre.",
    "altitude_m": "Convenience: -z, i.e. height above the floor.",
    "yaw_deg": (
        "Heading of the intended direction of travel through the gate, "
        "measured from +x toward +y. The gate plane is perpendicular to it."
    ),
    "visible": (
        "false = a virtual gate: it is never rendered, so the camera cannot "
        "see it, but it still scores and still acts as a collision volume. "
        "Virtual gates are how the paper forces a maneuver -- a split-S is a "
        "real gate with a virtual one 2.7 m above it."
    ),
}

# Per-track provenance.  The labels match the ones in track.py: [TEXT] is
# stated in the paper, [FIG] was measured off a figure, [OURS] is our choice.
NOTES = {
    "inverted_loop": [
        "Paper Figures 4 and 6, small track. Two real gates plus one virtual.",
        "[FIG] gate x = +3.0 and -2.0, altitude 1.35 m, measured off the plots.",
        "[TEXT] the 2.7 m real-to-virtual separation, and the 1.5 m loop radius "
        "that follows from it, are stated in section V.",
        "[FIG] validated: the same calibration reproduces the stated 2.7 m gate "
        "outer size and the 6 x 4 m flight area.",
    ],
    "ladder_inverted_loop": [
        "Section III-B: the same two real gates, with a ladder added at gate 1. "
        "This is the track the paper's simulation figures come from.",
        "[TEXT] t_g = 0.3 m here, smaller than Table III's 0.8, 'to further "
        "demonstrate SkyDreamer's ability to execute tight maneuvers'.",
        "[OURS] the two virtual gates that shape the ladder. The paper gives the "
        "maneuver in prose ('a full 360 degree left turn, and flies back over "
        "it', 'remaining mostly within 1 m of the gate') but not their placement.",
    ],
    "big": [
        "Paper Figure 9, large track. Eight real gates plus four virtual.",
        "[FIG] measured off the top-down and side panels on page 14 of the PDF. "
        "The arXiv HTML drops those panels because they are vector, not raster, "
        "which is why earlier attempts had only the 3D render to go on.",
        "[FIG] validated: the measured gate blocks come out 2.10 m wide spanning "
        "altitude 0.17-2.34 m, against the MAVLab gate's stated 2.1 m outer size.",
        "[OURS] the four virtual gates, and the gate order, follow the paper's "
        "prose description of the maneuvers.",
        "Gate size is deliberately left at the training value: this track is an "
        "out-of-distribution test of the layout, and changing the gates too "
        "would also change the mask, so a failure could not be attributed.",
    ],
}


def build() -> dict:
    from skydreamer.embodied_env import TRACK_T_G, TRACKS
    from skydreamer.params import EVAL, TRAIN

    out = {
        "schema": 1,
        "generated_by": "scripts/export_tracks.py from skydreamer/track.py",
        "warning": "Generated file. Edit skydreamer/track.py, then re-run the script.",
        "conventions": CONVENTIONS,
        "gate_tolerances": {
            "comment": (
                "Table III. d_g is the effective half-size used for scoring and "
                "collision; t_g is the pre-gate to post-gate tunnel thickness. "
                "Both are randomized per episode within these columns."
            ),
            "d_g_m": {"train": TRAIN.d_g, "evaluation": EVAL.d_g},
            "t_g_m": {"train": TRAIN.t_g, "evaluation": EVAL.t_g},
        },
        "tracks": {},
    }

    for name in TRACKS:
        if name.startswith("hw_"):
            continue   # same layouts on our drone and gate; docs/hardware.md
        tr = TRACKS[name]()
        pos = np.asarray(tr.pos, float)
        yaw = np.degrees(np.asarray(tr.yaw, float))
        vis = np.asarray(tr.visible, bool)
        alt = -pos[:, 2]

        gates = []
        for i, (p, y, v) in enumerate(zip(pos, yaw, vis), start=1):
            gates.append(
                {
                    "index": i,
                    "position_m": [round(float(c), 4) for c in p],
                    "altitude_m": round(float(-p[2]), 4),
                    "yaw_deg": round(float(y) % 360.0, 2),
                    "visible": bool(v),
                }
            )

        override = TRACK_T_G.get(name)
        out["tracks"][name] = {
            "notes": NOTES[name],
            "gate_size_m": {
                "inner": round(float(tr.inner[0]), 4),
                "outer": round(float(tr.outer[0]), 4),
            },
            "tunnel_t_g_m": override if override is not None else "Table III default",
            "counts": {
                "total": int(len(gates)),
                "real": int(vis.sum()),
                "virtual": int((~vis).sum()),
            },
            "extent_m": {
                "x": [round(float(pos[:, 0].min()), 2), round(float(pos[:, 0].max()), 2)],
                "y": [round(float(pos[:, 1].min()), 2), round(float(pos[:, 1].max()), 2)],
                "altitude": [round(float(alt.min()), 2), round(float(alt.max()), 2)],
            },
            "gates": gates,
        }
    return out


# --------------------------------------------------------------------------
# YAML, written out directly.  The structure is fixed and shallow, and doing it
# by hand keeps the comments -- which carry the provenance, the whole point of
# the file -- without adding a serializer dependency to a two-package project.
# --------------------------------------------------------------------------


def _y(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v if v.replace("_", "").replace("-", "").isalnum() else json.dumps(v)
    return json.dumps(v)


def to_yaml(d: dict) -> str:
    L = [
        "# SkyDreamer race tracks -- gate coordinates.",
        "#",
        "# GENERATED by scripts/export_tracks.py from skydreamer/track.py.",
        "# Edit the Python, then re-run the script; this file is checked in tests.",
        "",
        f"schema: {d['schema']}",
        "",
        "conventions:",
    ]
    for k, v in d["conventions"].items():
        L.append(f"  {k}: {json.dumps(v)}")

    g = d["gate_tolerances"]
    L += ["", "gate_tolerances:", f"  comment: {json.dumps(g['comment'])}"]
    for key in ("d_g_m", "t_g_m"):
        L.append(f"  {key}: {{train: {g[key]['train']}, evaluation: {g[key]['evaluation']}}}")

    L += ["", "tracks:"]
    for name, t in d["tracks"].items():
        L += [f"  {name}:", "    notes:"]
        L += [f"      - {json.dumps(n)}" for n in t["notes"]]
        L.append(
            f"    gate_size_m: {{inner: {t['gate_size_m']['inner']},"
            f" outer: {t['gate_size_m']['outer']}}}"
        )
        L.append(f"    tunnel_t_g_m: {_y(t['tunnel_t_g_m'])}")
        c = t["counts"]
        L.append(f"    counts: {{total: {c['total']}, real: {c['real']}, virtual: {c['virtual']}}}")
        e = t["extent_m"]
        L.append(
            f"    extent_m: {{x: {e['x']}, y: {e['y']}, altitude: {e['altitude']}}}"
        )
        L += ["    # x, y, z are NED metres (z is down); altitude_m = -z.", "    gates:"]
        for gate in t["gates"]:
            L.append(
                f"      - {{index: {gate['index']:>2},"
                f" position_m: {gate['position_m']},"
                f" altitude_m: {gate['altitude_m']},"
                f" yaw_deg: {gate['yaw_deg']},"
                f" visible: {_y(gate['visible'])}}}"
            )
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the files are stale")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "tracks")
    args = ap.parse_args()

    data = build()
    payloads = {
        args.out / "tracks.json": json.dumps(data, indent=2) + "\n",
        args.out / "tracks.yaml": to_yaml(data),
    }

    if args.check:
        stale = [p for p, text in payloads.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale))
            print("run: python scripts/export_tracks.py")
            return 1
        print("tracks/ is up to date")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for p, text in payloads.items():
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
