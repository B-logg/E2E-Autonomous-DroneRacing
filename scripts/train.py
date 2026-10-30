#!/usr/bin/env python3
"""Drive the paper's three-phase training schedule (section III-A).

DreamerV3 has no mid-run config schedule, so each phase is a separate
invocation resuming from the previous phase's checkpoint in the same logdir.

  small tracks (17M steps, ~50 h on one A100 80GB)
    phase 1   0 ->  8M   defaults
    phase 2   8M -> 13M  batch_length 64 -> 256   (long-horizon parameter ID)
    phase 3  13M -> 17M  actent 3e-4 -> 1e-5, lr 4e-5 -> 2e-6   (fine-tune)

  big track (35M steps; III-D)
    phase boundaries move to 23M and 31M, train_ratio 64, symlog off.

  --finetune (run 6, 9M steps; docs/deviations.md)
    fine-tune a finished paper-drone agent on our hardware, switching our
    conditions on a few at a time (skydreamer/hardware.py, HW_STAGES):
    stage 1   0   -> 0.5M  hw1: camera forward, our FOV, 2.4 m frame
    stage 2   0.5 -> 1.5M  hw2: + our 2.1 m frame
    stage 3   1.5 -> 3M    hw3: + our thrust-to-weight
    stage 4   3   -> 7M    hw:  + our gate window (the full profile)
    stage 5   7   -> 9M    hw, the paper's phase-3 fine-tune settings
    Stages 1-4 use the paper's phase-2 settings.  The agent to start from is
    passed as --run.from_checkpoint (./run.sh --finetune does this).

  --finetune team (run 7, 10M steps; docs/deviations.md)
    fine-tune run 6 onto our real track and gate (track.TEAM_TRACK):
    stage 1   0   -> 1.5M  hwg_inverted_loop: our real gate on the layout it knows
    stage 2   1.5 -> 4M    hwg_team: our track, window 0.60
    stage 3   4   -> 7M    hw_team:  our track, window 0.50
    stage 4   7   -> 10M   hw_team, the paper's phase-3 fine-tune settings
    (./run.sh --finetune <ckpt> --team)

Usage:
  python scripts/train.py --logdir ~/logdir/sd/run1
  python scripts/train.py --logdir ~/logdir/sd/big1 --preset big
  python scripts/train.py --logdir ~/logdir/sd/run1 --from-phase 2   # resume
  python scripts/train.py --logdir ~/logdir/sd/ft1 --finetune --run.from_checkpoint <ckpt>
  python scripts/train.py --logdir ~/logdir/sd/ft2 --finetune team --run.from_checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "third_party" / "dreamerv3" / "dreamerv3" / "main.py"

PRESETS = {
    "small": dict(configs="skydreamer", boundaries=(8e6, 13e6, 17e6)),
    "big": dict(configs="skydreamer_big", boundaries=(23e6, 31e6, 35e6)),
}

# Phase 2 and 3 overrides, exactly as the paper words them.
PHASE_ARGS = {
    1: [],
    2: ["--batch_length", "256"],
    3: [
        "--batch_length", "256",
        "--agent.imag_loss.actent", "1e-5",
        "--agent.opt.lr", "2e-6",
    ],
}

# Only the agent is loaded from the checkpoint, not its step counter, so these
# count fine-tuning steps from 0.  Phase 2's settings are the ones run 3 kept
# improving under (73% -> 91% between 6M and 11M) without forgetting what it
# could already fly.  They also put the exploration entropy back to 3e-4, which a
# new layout needs: the agent being loaded ended on phase 3's 1e-5.
FINETUNE = {
    # run 6: run 3's paper-drone agent onto our drone, 5% -> 69% (physical, 5 laps)
    "hw": [
        (0.5e6, "hw1_inverted_loop", PHASE_ARGS[2]),
        (1.5e6, "hw2_inverted_loop", PHASE_ARGS[2]),
        (3.0e6, "hw3_inverted_loop", PHASE_ARGS[2]),
        (7.0e6, "hw_inverted_loop", PHASE_ARGS[2]),
        (9.0e6, "hw_inverted_loop", PHASE_ARGS[3]),
    ],
    # run 7: run 6 onto our real track.  Run 6 scored 15% with our gate on its own
    # layout and 0% on our layout, so the gate goes first; and run 6's biggest gain
    # (33% -> 66%) came at its switch to phase 3, so that stage gets 3M again.
    "team": [
        (1.5e6, "hwg_inverted_loop", PHASE_ARGS[2]),
        (4.0e6, "hwg_team", PHASE_ARGS[2]),
        (7.0e6, "hw_team", PHASE_ARGS[2]),
        (10.0e6, "hw_team", PHASE_ARGS[3]),
    ],
    # run 7b: run 7's 4M agent (0% on our track: it reached G2 at ~14 m/s and hit
    # the ceiling or the far wall), now under the room's speed limit
    # (track.V_CAP) and with demonstrations in the buffer.  It restarts at the
    # wider window, then follows run 7's last two stages.
    "team2": [
        (3.0e6, "hwg_team", PHASE_ARGS[2]),
        (6.0e6, "hw_team", PHASE_ARGS[2]),
        (8.0e6, "hw_team", PHASE_ARGS[3]),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--from-phase", type=int, default=1,
                    help="1-3, or a stage of the --finetune schedule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hw", action="store_true",
                    help="train on our drone and gate (docs/hardware.md): the hw_* task")
    ap.add_argument("--finetune", nargs="?", const="hw", default=None, choices=sorted(FINETUNE),
                    help="fine-tune a finished agent in stages: hw (run 6, our drone), "
                         "team (run 7, our real track) or team2 (run 7b, from run 7's 4M "
                         "under the speed limit)")
    ap.add_argument("--steps-scale", type=float, default=1.0,
                    help="multiply every phase boundary (testing the phase chain end to end)")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    if not MAIN.exists():
        sys.exit(f"{MAIN} missing -- run scripts/setup_dreamerv3.sh first")

    preset = PRESETS[args.preset]
    # `main.py` splits the task on its first underscore: skydreamer_hw_<track>.
    track = "big" if args.preset == "big" else "inverted_loop"
    if args.finetune:
        if args.preset != "small":
            sys.exit("--finetune is defined for the small preset only")
        phases = [(steps, f"skydreamer_{task}", over) for steps, task, over in FINETUNE[args.finetune]]
    else:
        task = f"skydreamer_hw_{track}" if args.hw else None
        phases = [(steps, task, PHASE_ARGS[i + 1])
                  for i, steps in enumerate(preset["boundaries"])]
    if not 1 <= args.from_phase <= len(phases):
        sys.exit(f"--from-phase {args.from_phase}: there are only {len(phases)} phases")
    logdir = pathlib.Path(args.logdir).expanduser()

    for phase, (steps, task, over) in enumerate(phases, 1):
        if phase < args.from_phase:
            continue
        cmd = [
            sys.executable, str(MAIN),
            "--configs", preset["configs"], "size12m",
            "--logdir", str(logdir),
            "--seed", str(args.seed),
            # `run.steps` is cumulative, so each phase just raises the ceiling
            # and DreamerV3 resumes from the checkpoint already in the logdir.
            "--run.steps", f"{steps * args.steps_scale:.0f}",
            *over,
            *(["--task", task] if task else []),
            *extra,
        ]
        print(f"\n=== phase {phase} -> {steps * args.steps_scale:,.0f} steps ===\n{' '.join(cmd)}\n", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
