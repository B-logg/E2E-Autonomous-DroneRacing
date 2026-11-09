#!/usr/bin/env bash
#
# Score several checkpoints on the same flights, then compare and diagnose them.
#
#   ./run_compare.sh 7f3M=logdir/small-team6-.../snapshots/step_000003026800 \
#                    7g1.5M=eval_7g/step_000001541392
#
#   --episodes N   flights per checkpoint and lap count (default 400)
#   --seed S       the flights' seed (default 20000: new flights, not the
#                  10000 that run_test.sh and every earlier score used)
#   --laps "1 3"   lap counts (default 1 and 3)
#   --out DIR      where results go (default compare/)
#   --track T      the layout flown (default team; team4up = G4 raised 0.5 m)
#
# NAME=PATH: PATH is a training snapshot (`<run>/snapshots/step_N`, config from
# `<run>/config.yaml`), an evaluation directory (`<run>/eval/step_N`, or one
# made by hand: ckpt/ and config.yaml), or a checkpoint directory itself.
#
# Every checkpoint gets the same (seed, episodes), hence the same airframes,
# starts and noise: scripts/compare_evals.py pairs them flight by flight.
# Every flight's trajectory is kept (`--dump`) for scripts/diagnose_failures.py.
# Results: <out>/<NAME>/evaluation_physical_<L>lap_s<S>{.json,.txt,_episodes.json,_traj.npz}
# and <out>/report.txt.  Bring <out> home without the ckpt/ copies:
#   ssh ... "cd ~/E2E-Autonomous-DroneRacing && tar cf - --exclude=ckpt compare" | tar xf -
#
# Run it with no training on the GPU, or expect both to slow down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
VENV="${SKYDREAMER_VENV:-${ROOT}/.venv}"
PY="${VENV}/bin/python"
DV3_DIR="${ROOT}/third_party/dreamerv3"

EPISODES=400
SEED=20000
LAPS="1 3"
OUT="${ROOT}/compare"
TRACK=team
RUNS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift ;;
    --seed)     SEED="$2"; shift ;;
    --laps)     LAPS="$2"; shift ;;
    --out)      OUT="$2"; shift ;;
    --track)    TRACK="$2"; shift ;;
    -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
    *=*)        RUNS+=("$1") ;;
    *) printf 'unknown argument: %s (expected NAME=PATH)\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ ${#RUNS[@]} -gt 0 ] || die "give at least one NAME=PATH (./run_compare.sh --help)"
[ -x "${PY}" ] || die "no environment at ${VENV}. Run ./run.sh --setup-only first."
export PYTHONPATH="${DV3_DIR}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The checkpoint directory (one holding `latest`) and the config for PATH.
find_ckpt() {
  local p="$1"
  if [ -f "${p}/latest" ]; then echo "${p}"; return; fi
  if [ -f "${p}/ckpt/latest" ]; then echo "${p}/ckpt"; return; fi
  return 1
}
find_config() {
  local d="$1"
  for _ in 1 2 3 4; do
    [ -f "${d}/config.yaml" ] && { echo "${d}/config.yaml"; return; }
    d="$(dirname "${d}")"
  done
  return 1
}

# Check every PATH before spending an hour on the first one.
NAMES=()
for r in "${RUNS[@]}"; do
  name="${r%%=*}"; path="${r#*=}"
  [ -d "${path}" ] || die "${name}: no directory ${path}"
  find_ckpt "${path}" >/dev/null || die "${name}: no checkpoint (a 'latest' file) in ${path} or ${path}/ckpt"
  find_config "${path}" >/dev/null || die "${name}: no config.yaml at or above ${path}"
  NAMES+=("${name}")
done

for r in "${RUNS[@]}"; do
  name="${r%%=*}"; path="${r#*=}"
  dest="${OUT}/${name}"
  mkdir -p "${dest}"
  if [ ! -f "${dest}/ckpt/latest" ]; then
    rm -rf "${dest}/ckpt"
    cp -r "$(find_ckpt "${path}")" "${dest}/ckpt"
  fi
  cp "$(find_config "${path}")" "${dest}/config.yaml"
  for laps in ${LAPS}; do
    tag="_${laps}lap_s${SEED}"
    if [ -f "${dest}/evaluation_physical${tag}_traj.npz" ]; then
      log "${name}: ${laps} lap(s), seed ${SEED} already scored -- skipping"
      continue
    fi
    log "${name}: ${EPISODES} flights x ${laps} lap(s), seed ${SEED}"
    "${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${dest}" --track "${TRACK}" --protocol physical \
        --episodes "${EPISODES}" --laps "${laps}" --seed "${SEED}" --tag "${tag}" --dump
  done
done

log "Comparison"
ARGS=()
for n in "${NAMES[@]}"; do ARGS+=("${n}=${OUT}/${n}"); done
{
  "${PY}" "${ROOT}/scripts/compare_evals.py" "${ARGS[@]}" --laps ${LAPS}
  for n in "${NAMES[@]}"; do
    for laps in ${LAPS}; do
      echo
      "${PY}" "${ROOT}/scripts/diagnose_failures.py" "${OUT}/${n}" --laps "${laps}" \
          --json "${OUT}/${n}/diagnosis_${laps}lap_s${SEED}.json"
    done
  done
} | tee "${OUT}/report.txt"
log "Done.  ${OUT}/report.txt"
