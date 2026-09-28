#!/usr/bin/env bash
#
# Evaluate a run -- including one that is still training.
#
#   ./run_test.sh                  score + render the newest checkpoint, both tracks
#   ./run_test.sh --no-video       numbers only, much faster
#   ./run_test.sh --track big      one track instead of both
#   ./run_test.sh --episodes 50    fewer episodes (default 100, as Table IV)
#   ./run_test.sh --watch 3600     repeat every hour, building a history
#   ./run_test.sh --history        print the history so far and exit
#
# Safe to run while `./run.sh` is training in another shell.  It copies the
# checkpoint before reading it (the trainer keeps only one and deletes the old
# one the moment it writes a new one), and it never writes inside the training
# run's own files -- results go to `<run>/eval/step_<N>/`, one directory per
# checkpoint, so repeated runs accumulate a curve instead of overwriting.
#
# It does share the GPU with the trainer, so expect training to slow down for
# the few minutes it takes.  `--no-video` is the cheap version.
#
# Each snapshot copies a ~126 MB checkpoint.  The results are tiny and kept
# forever; the copied checkpoints are pruned to the newest `--keep` (default 3)
# so an unattended `--watch` does not fill the disk.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

VENV="${SKYDREAMER_VENV:-${ROOT}/.venv}"
LOGROOT="${SKYDREAMER_LOGDIR:-${ROOT}/logdir}"
DV3_DIR="${ROOT}/third_party/dreamerv3"

VIDEO=1
EPISODES=100
LAPS=5
WATCH=0
KEEP=3
TRACKS=()
HISTORY_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-video)  VIDEO=0 ;;
    --video)     VIDEO=1 ;;
    --track)     TRACKS+=("$2"); shift ;;
    --big)       TRACKS+=("big") ;;
    --episodes)  EPISODES="$2"; shift ;;
    --laps)      LAPS="$2"; shift ;;
    --watch)     WATCH="$2"; shift ;;
    --history)   HISTORY_ONLY=1 ;;
    --keep)      KEEP="$2"; shift ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[ ${#TRACKS[@]} -gt 0 ] || TRACKS=(inverted_loop big)

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

latest_run() {
  ls -td "${LOGROOT}"/*/ 2>/dev/null | while read -r d; do
    [ -f "${d}config.yaml" ] && { printf '%s' "${d%/}"; return; }
  done
}

# --------------------------------------------------------------------------
# This script deliberately does no setup.  Setting up while a training run is
# live could rebuild the venv or re-patch DreamerV3 underneath it.
[ -x "${VENV}/bin/python" ] && [ -f "${VENV}/pyvenv.cfg" ] \
  || die "no environment at ${VENV}. Run ./run.sh --setup-only first."
[ -d "${DV3_DIR}" ] || die "DreamerV3 missing. Run ./run.sh --setup-only first."
PY="${VENV}/bin/python"
export PYTHONPATH="${DV3_DIR}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Share the card with the trainer rather than taking it: a small cap, and never
# preallocate.  The trainer's own measured peak is ~11 GB.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.15}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN_DIR="$(latest_run || true)"
[ -n "${RUN_DIR}" ] || die "no run under ${LOGROOT}"
EVAL_ROOT="${RUN_DIR}/eval"
HISTORY="${EVAL_ROOT}/history.jsonl"

print_history() {
  [ -f "${HISTORY}" ] || { echo "no evaluations yet"; return; }
  log "History  ${HISTORY}"
  "${PY}" - "${HISTORY}" <<'PY'
import json, sys, pathlib
rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
if not rows:
    print("  empty"); raise SystemExit
tracks = sorted({r["track"] for r in rows})
print(f"\n  {'step':>12}  " + "  ".join(f"{t:>28}" for t in tracks))
print(f"  {'':>12}  " + "  ".join(f"{'success  gates  diverged':>28}" for _ in tracks))
for step in sorted({r["step"] for r in rows}):
    cells = []
    for t in tracks:
        m = [r for r in rows if r["step"] == step and r["track"] == t]
        if not m:
            cells.append(f"{'-':>28}"); continue
        r = m[-1]
        d = r.get("diverged_frac")
        cells.append(f"{r['success_rate']:>7.1%} {r['mean_gates']:>6.2f} "
                     f"{('-' if d is None else f'{d:.0%}'):>9}")
    print(f"  {step:>12,}  " + "  ".join(cells))
PY
}

if [ "${HISTORY_ONLY}" = "1" ]; then
  print_history
  exit 0
fi

# --------------------------------------------------------------------------
run_once() {
  log "Snapshotting the newest checkpoint of ${RUN_DIR}"
  SNAP="$("${PY}" "${ROOT}/scripts/snapshot_ckpt.py" \
      --logdir "${RUN_DIR}" --out "${EVAL_ROOT}" --print-dir | tail -1)"
  [ -d "${SNAP}" ] || die "snapshot failed"
  STEP="$(basename "${SNAP}" | sed 's/^step_//' | sed 's/^0*//')"
  STEP="${STEP:-0}"   # an all-zero name is step 0, not an empty string

  if [ "${VIDEO}" = "1" ]; then
    # Deliberately does not install it: pip-resolving into the venv a training
    # run is using could replace numpy under the trainer.  ./run.sh installs it
    # during setup, so this only fires on a venv built before that change.
    "${PY}" -c "import matplotlib, PIL" 2>/dev/null || die \
      "matplotlib missing from ${VENV}. Either pass --no-video, or install it
  when no training is running:  ${VENV}/bin/python -m pip install matplotlib pillow"
  fi

  for track in "${TRACKS[@]}"; do
    log "step ${STEP}: scoring ${track} (${EPISODES} episodes x ${LAPS} laps)"
    "${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${SNAP}" --track "${track}" \
        --episodes "${EPISODES}" --laps "${LAPS}" || {
      printf '\033[1;31m  scoring %s failed -- continuing\033[0m\n' "${track}" >&2
      continue
    }
    if [ "${VIDEO}" = "1" ]; then
      log "step ${STEP}: rendering ${track}"
      "${PY}" "${ROOT}/scripts/visualize.py" --logdir "${SNAP}" --track "${track}" \
        || printf '\033[1;31m  rendering %s failed -- continuing\033[0m\n' "${track}" >&2
    fi
  done

  "${PY}" - "${SNAP}" "${STEP}" "${HISTORY}" <<'PY'
import json, pathlib, sys
snap, step, hist = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
hist.parent.mkdir(parents=True, exist_ok=True)
with hist.open("a") as f:
    for p in sorted(snap.glob("evaluation*.json")):
        d = json.loads(p.read_text())
        diag = d.get("diagnostics", {})
        term = diag.get("termination", {})
        total = sum(term.values()) or 1
        f.write(json.dumps({
            "step": step,
            "track": d["track"],
            "success_rate": d["success_rate"],
            "mean_gates": d["mean_gates_passed"],
            "diverged_frac": term.get("diverged", 0) / total if term else None,
            "flight_time_mean": diag.get("duration_s", {}).get("mean"),
            "max_speed_mean": diag.get("max_speed_ms_all", {}).get("mean"),
            "decode_p_w_early": d.get("decode_error", {}).get("early", {}).get("p_w"),
        }) + "\n")
PY
  # Keep every result; keep only the newest few of the copied checkpoints,
  # which are 126 MB each and only needed to re-run an old evaluation.
  ls -d "${EVAL_ROOT}"/step_*/ 2>/dev/null | sort | head -n -"${KEEP}" | while read -r d; do
    [ -d "${d}ckpt" ] || continue
    rm -rf "${d}ckpt"
    printf '  pruned checkpoint copy from %s\n' "$(basename "${d}")"
  done

  print_history
  log "Done  ${SNAP}"
}

if [ "${WATCH}" != "0" ]; then
  log "Watching: re-evaluating every ${WATCH}s.  Ctrl-C to stop."
  while true; do
    run_once || printf '\033[1;31mevaluation round failed -- will retry\033[0m\n' >&2
    log "sleeping ${WATCH}s"
    sleep "${WATCH}"
  done
else
  run_once
fi
