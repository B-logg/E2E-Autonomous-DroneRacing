#!/usr/bin/env bash
#
# CTBA racing policy: setup, demonstrations, curriculum training, evaluation.
#
#   ./run.sh --setup                 venv + DreamerV3 (via skydreamer) + ctba registration
#   ./run.sh --smoke                 ~10 min CPU/GPU: tests + a tiny 4-stage run + eval
#   ./run.sh --collect-demos         CTBA demonstrations on our track -> ./demos_ctba
#   ./run.sh                         the curriculum, 20M steps (demos seeded if ./demos_ctba exists)
#   ./run.sh --resume [--from-stage N]   continue the newest run
#   ./run.sh --eval <run dir>        1 and 3 laps, 200 flights each, real-gate airframe check
#   STOCHGAN_WEIGHTS=<G.npz> ./run.sh --finetune-stochgan <snapshot or run dir> [--ft-run <dir>]
#                                    2M steps on StochGAN masks from that agent (scripts/finetune_stochgan.py),
#                                    then 1/3-lap evaluations on clean (ce) and StochGAN (ceg) masks;
#                                    --ft-run resumes an unfinished fine-tune run
#
# Lives inside the skydreamer repository (hierarchy_control/); SKYDREAMER=<path> overrides.
# Track: CTBA_TRACK (default team44up).  Runs go to CTBA_LOGDIR (default ./logdir).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYDREAMER="${SKYDREAMER:-${HERE}/..}"
SKYDREAMER="$(cd "${SKYDREAMER}" && pwd)"
export SKYDREAMER
DV3_DIR="${SKYDREAMER}/third_party/dreamerv3"
VENV="${SKYDREAMER_VENV:-${SKYDREAMER}/.venv}"
PY="${VENV}/bin/python"
LOGROOT="${CTBA_LOGDIR:-${HERE}/logdir}"
TRACK="${CTBA_TRACK:-team44up}"
DEMOS="${CTBA_DEMOS:-${HERE}/demos_ctba}"
SNAP_EVERY="${CTBA_SNAPSHOT_EVERY:-5e5}"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

MODE="train"; EVAL_DIR=""; FROM_STAGE=""; FT_SRC=""; FT_RUN=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --setup) MODE="setup" ;;
    --smoke) MODE="smoke" ;;
    --collect-demos) MODE="demos" ;;
    --resume) MODE="resume" ;;
    --from-stage) FROM_STAGE="$2"; shift ;;
    --eval) MODE="eval"; EVAL_DIR="$2"; shift ;;
    --finetune-stochgan) MODE="stochgan"; FT_SRC="$2"; shift ;;
    --ft-run) FT_RUN="$2"; shift ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

setup() {
  log "skydreamer setup (venv, DreamerV3 + informed patch)"
  [ -x "${SKYDREAMER}/run.sh" ] || die "no skydreamer at ${SKYDREAMER} (set SKYDREAMER=)"
  # skydreamer's setup builds the venv and the patched DreamerV3; it exits after setup
  (cd "${SKYDREAMER}" && ./run.sh --setup-only)
  log "ctba package + DreamerV3 registration"
  # skydreamer's setup may have just installed uv into ~/.local/bin (not yet on PATH here);
  # its venv has no pip, so uv is the installer.
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PY}" --quiet -e "${HERE}"
  else
    "${PY}" -m pip install --quiet -e "${HERE}"
  fi
  "${PY}" "${HERE}/scripts/patch_dreamerv3.py" "${DV3_DIR}"
}

# The registration is re-applied on every start: skydreamer's setup recreates the
# DreamerV3 checkout whenever its own patch changes, which drops ours.
quick_setup() {
  [ -x "${PY}" ] && [ -f "${DV3_DIR}/dreamerv3/main.py" ] || setup
  "${PY}" -c "import ctba" 2>/dev/null || setup
  "${PY}" "${HERE}/scripts/patch_dreamerv3.py" "${DV3_DIR}" >/dev/null
}

JAX_ARGS=()
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no nvidia-smi: running on CPU (fine for --smoke only)"
  JAX_ARGS=(--jax.platform cpu)
fi
export PYTHONPATH="${DV3_DIR}:${HERE}:${SKYDREAMER}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
LOG_ARGS=(--logger.outputs jsonl
          --logger.filter 'score|length|fps|ratio|train/loss/|train/rand/|epstats/log/')
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)
ENVS=$(( CORES < 16 ? CORES : 16 ))

case "${MODE}" in
setup)
  setup; log "Setup complete. Next: ./run.sh --smoke"; exit 0 ;;

smoke)
  quick_setup
  RUN="${LOGROOT}/smoke-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "${RUN}/replay"
  log "tests"
  "${PY}" -m pytest "${HERE}/tests" -q -x
  log "demonstrations (4 episodes)"
  "${PY}" "${HERE}/scripts/collect_demos.py" --out "${RUN}/demos" --track "${TRACK}" \
      --episodes 64 --max-keep 4 --size size1m
  cp "${RUN}/demos/replay/"*.npz "${RUN}/replay/"
  log "tiny 4-stage curriculum -> ${RUN}"
  "${PY}" "${HERE}/scripts/train.py" --logdir "${RUN}" --track "${TRACK}" --size size1m \
      --steps-scale 0.0002 --run.envs 2 --batch_size 8 --batch_length 16 --report_length 16 \
      --replay_context 1 --run.train_ratio 16 --run.log_every 60 --run.report_every 1e9 \
      --run.save_every 300 ${JAX_ARGS[@]+"${JAX_ARGS[@]}"} --logger.outputs jsonl
  log "evaluation (8 flights)"
  "${PY}" "${HERE}/scripts/evaluate.py" --logdir "${RUN}" --track "${TRACK}" --episodes 8 --laps 1
  log "Smoke test passed: ${RUN}"; exit 0 ;;

demos)
  quick_setup
  log "CTBA demonstrations on ${TRACK} -> ${DEMOS}"
  "${PY}" "${HERE}/scripts/collect_demos.py" --out "${DEMOS}" --track "${TRACK}" \
      --episodes "${CTBA_DEMO_SEEDS:-4000}" --max-keep "${CTBA_DEMO_KEEP:-800}"
  exit 0 ;;

eval)
  quick_setup
  [ -d "${EVAL_DIR}" ] || die "no run at ${EVAL_DIR}"
  for L in 1 3; do
    "${PY}" "${HERE}/scripts/evaluate.py" --logdir "${EVAL_DIR}" --track "${TRACK}" \
        --episodes 200 --laps "${L}" 2>&1 | tee -a "${EVAL_DIR}/eval.log"
  done
  exit 0 ;;
esac

# ---- StochGAN fine-tune ------------------------------------------------------
if [ "${MODE}" = "stochgan" ]; then
  quick_setup
  [ -e "${FT_SRC}" ] || die "no source at ${FT_SRC} (a snapshots/step_N dir or a run dir)"
  RUN="${FT_RUN:-${LOGROOT}/ctba-stochgan-${TRACK}-$(date +%Y%m%d-%H%M%S)}"
  if [ ! -f "${RUN}/stochgan_G.npz" ]; then
    [ -n "${STOCHGAN_WEIGHTS:-}" ] && [ -f "${STOCHGAN_WEIGHTS}" ] \
      || die "set STOCHGAN_WEIGHTS=<generator .npz> (scripts/convert_stochgan.py)"
  fi
  mkdir -p "${RUN}/replay"
  log "StochGAN fine-tune ${FT_SRC} -> ${RUN}"
  "${PY}" "${SKYDREAMER}/scripts/prune_replay.py" --logdir "${RUN}" --keep-steps 10000000 \
      --interval 600 >> "${RUN}/prune.log" 2>&1 &
  PRUNER=$!
  "${PY}" "${SKYDREAMER}/scripts/milestone_snapshots.py" --logdir "${RUN}" --every "${SNAP_EVERY}" \
      --interval 300 >> "${RUN}/snapshots.log" 2>&1 &
  SNAPPER=$!
  trap 'kill ${PRUNER} ${SNAPPER} 2>/dev/null || true' EXIT
  "${PY}" "${HERE}/scripts/finetune_stochgan.py" --source "${FT_SRC}" --logdir "${RUN}" --track "${TRACK}" \
      --jax.prealloc False --run.envs "${ENVS}" ${JAX_ARGS[@]+"${JAX_ARGS[@]}"} "${LOG_ARGS[@]}" \
      ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "${RUN}/train.log"
  log "evaluation (clean masks: ce, StochGAN masks: ceg)"
  export STOCHGAN_WEIGHTS="${RUN}/stochgan_G.npz"
  for ST in ce ceg; do
    for L in 1 3; do
      "${PY}" "${HERE}/scripts/evaluate.py" --logdir "${RUN}" --track "${TRACK}" --stage "${ST}" \
          --episodes 200 --laps "${L}" 2>&1 | tee -a "${RUN}/eval.log"
    done
  done
  log "Done: ${RUN}"; exit 0
fi

# ---- train / resume -------------------------------------------------------
quick_setup
if [ "${MODE}" = "resume" ]; then
  RUN="$(ls -dt "${LOGROOT}"/ctba-* 2>/dev/null | head -1)"
  [ -n "${RUN}" ] || die "nothing to resume in ${LOGROOT}"
  [ -n "${FROM_STAGE}" ] || FROM_STAGE="$(cat "${RUN}/stage" 2>/dev/null || echo 1)"
  log "resuming ${RUN} at stage ${FROM_STAGE}"
else
  RUN="${LOGROOT}/ctba-${TRACK}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "${RUN}/replay"
  FROM_STAGE="${FROM_STAGE:-1}"
  if ls "${DEMOS}/replay/"*.npz >/dev/null 2>&1; then
    cp "${DEMOS}/replay/"*.npz "${RUN}/replay/"
    cp "${DEMOS}/replay/.demo-chunks" "${RUN}/replay/" 2>/dev/null || true
    echo "seeded $(ls "${DEMOS}/replay/"*.npz | wc -l) demonstration chunks from ${DEMOS}"
  else
    echo "WARNING: no demonstrations at ${DEMOS} (./run.sh --collect-demos); stage 1 starts without"
  fi
  log "new run ${RUN}"
fi

# Replay chunks are never deleted by DreamerV3; skydreamer's pruner keeps the disk
# bounded and leaves the demonstration chunks alone.  Snapshots every 0.5M.
"${PY}" "${SKYDREAMER}/scripts/prune_replay.py" --logdir "${RUN}" --keep-steps 10000000 \
    --interval 600 >> "${RUN}/prune.log" 2>&1 &
PRUNER=$!
"${PY}" "${SKYDREAMER}/scripts/milestone_snapshots.py" --logdir "${RUN}" --every "${SNAP_EVERY}" \
    --interval 300 >> "${RUN}/snapshots.log" 2>&1 &
SNAPPER=$!
trap 'kill ${PRUNER} ${SNAPPER} 2>/dev/null || true' EXIT

"${PY}" "${HERE}/scripts/train.py" --logdir "${RUN}" --track "${TRACK}" --from-stage "${FROM_STAGE}" \
    --jax.prealloc False --run.envs "${ENVS}" ${JAX_ARGS[@]+"${JAX_ARGS[@]}"} "${LOG_ARGS[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "${RUN}/train.log"

log "evaluation"
for L in 1 3; do
  "${PY}" "${HERE}/scripts/evaluate.py" --logdir "${RUN}" --track "${TRACK}" --episodes 200 --laps "${L}" \
      2>&1 | tee -a "${RUN}/eval.log"
done
log "Done: ${RUN}"
