#!/usr/bin/env bash
#
# One command, from a fresh clone to a trained and evaluated SkyDreamer policy.
#
#   ./run.sh --smoke      ~5 min   verify the whole pipeline works. DO THIS FIRST.
#   ./run.sh              ~50 h    the paper's run: 17M steps, 3 phases, then eval
#   ./run.sh --setup-only ~10 min  install everything, train nothing
#   ./run.sh --eval-only  ~10 min  re-evaluate the newest run
#
# Nothing is needed on the server beyond an NVIDIA GPU, a CUDA driver, git and
# curl. Python is installed by uv; no root, no conda, no system packages.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DV3_COMMIT="cdf570902b1eaba193cc8ef69426cd4edde1b0bc"   # the commit SkyDreamer cites (III-A)
DV3_DIR="${ROOT}/third_party/dreamerv3"
VENV="${SKYDREAMER_VENV:-${ROOT}/.venv}"
PY_VERSION="3.11"            # jax 0.4.33, which dreamerv3 pins, supports 3.10-3.12
LOGROOT="${SKYDREAMER_LOGDIR:-${ROOT}/logdir}"

MODE="full"
PRESET="small"
EXTRA=()
for arg in "$@"; do
  case "${arg}" in
    --smoke)      MODE="smoke" ;;
    --setup-only) MODE="setup" ;;
    --eval-only)  MODE="eval" ;;
    --big)        PRESET="big" ;;
    *)            EXTRA+=("${arg}") ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
log "Environment"
# --------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no nvidia-smi found -- will fall back to CPU (fine for --smoke, useless for a real run)"
fi

# --------------------------------------------------------------------------
log "Python environment"
# --------------------------------------------------------------------------
# uv installs its own Python, so the server's system Python is irrelevant.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; install Python ${PY_VERSION} manually and re-run"

if [ ! -x "${VENV}/bin/python" ]; then
  uv venv --python "${PY_VERSION}" "${VENV}"
fi
PY="${VENV}/bin/python"
PIP=(uv pip install --python "${PY}" --quiet)
echo "python: $(${PY} -V)"

# --------------------------------------------------------------------------
log "DreamerV3 @ ${DV3_COMMIT:0:7} + SkyDreamer patch"
# --------------------------------------------------------------------------
if [ ! -d "${DV3_DIR}/.git" ]; then
  mkdir -p "$(dirname "${DV3_DIR}")"
  git clone --quiet https://github.com/danijar/dreamerv3.git "${DV3_DIR}"
  git -C "${DV3_DIR}" checkout --quiet --detach "${DV3_COMMIT}"
  git -C "${DV3_DIR}" apply "${ROOT}/patches/informed_dreamer.patch"
  echo "patched"
else
  echo "already present (delete third_party/ to redo)"
fi

# --------------------------------------------------------------------------
log "Dependencies"
# --------------------------------------------------------------------------
if [ ! -f "${VENV}/.deps-ok" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    "${PIP[@]}" "jax[cuda12]==0.4.33"
  else
    "${PIP[@]}" "jax[cpu]==0.4.33"
  fi
  # dreamerv3's requirements.txt drags in Atari/DMLab/Minecraft extras that are
  # slow, fragile and irrelevant here. These are the ones it actually imports.
  "${PIP[@]}" "numpy<2" elements ninjax optax portal scope granular einops chex \
      jaxtyping colored_traceback tqdm "ruamel.yaml" msgpack rich cloudpickle psutil \
      pytest
  "${PIP[@]}" -e "${ROOT}"
  touch "${VENV}/.deps-ok"
fi
"${PY}" -c "import jax; print('jax', jax.__version__, jax.devices())"

export PYTHONPATH="${DV3_DIR}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

if [ "${MODE}" = "setup" ]; then
  log "Setup complete. Run './run.sh --smoke' next."
  exit 0
fi

# --------------------------------------------------------------------------
if [ "${MODE}" = "eval" ]; then
  RUN_DIR="$(ls -td "${LOGROOT}"/*/ 2>/dev/null | head -1 || true)"
  [ -n "${RUN_DIR}" ] || die "no runs under ${LOGROOT}"
  log "Evaluating ${RUN_DIR}"
  "${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${RUN_DIR}" "${EXTRA[@]+"${EXTRA[@]}"}"
  exit 0
fi

# --------------------------------------------------------------------------
if [ "${MODE}" = "smoke" ]; then
  RUN_DIR="${LOGROOT}/smoke-$(date +%Y%m%d-%H%M%S)"
  log "Smoke test -> ${RUN_DIR}"
  echo "Tiny model, 2k steps. Proves the pipeline runs; the policy will be useless."
  "${PY}" -m pytest "${ROOT}/tests" -q -x --ignore="${ROOT}/tests/test_gatenet.py"
  "${PY}" "${DV3_DIR}/dreamerv3/main.py" \
      --configs skydreamer size1m --logdir "${RUN_DIR}" \
      --run.steps 2000 --run.envs 4 --batch_size 8 --batch_length 16 \
      --report_length 16 --replay_context 1 --run.train_ratio 16 \
      --run.log_every 20 --run.report_every 1e9 --run.save_every 500
  "${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${RUN_DIR}" --episodes 8 --laps 1
  log "Smoke test passed. Now run './run.sh' for the real thing."
  exit 0
fi

# --------------------------------------------------------------------------
# The paper's run, section III-A: three phases, 17M steps total.
#   phase 1   0 ->  8M   defaults
#   phase 2   8M -> 13M  batch_length 64 -> 256
#   phase 3  13M -> 17M  entropy 3e-4 -> 1e-5, lr 4e-5 -> 2e-6
# --------------------------------------------------------------------------
RUN_DIR="${LOGROOT}/${PRESET}-$(date +%Y%m%d-%H%M%S)"
log "Training -> ${RUN_DIR}"
echo "Paper: ~50 h on 4/7 of an A100 80GB. Detach this (tmux/screen/nohup)."
mkdir -p "${RUN_DIR}"

"${PY}" "${ROOT}/scripts/train.py" \
    --logdir "${RUN_DIR}" --preset "${PRESET}" "${EXTRA[@]+"${EXTRA[@]}"}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"

log "Evaluating in simulation"
"${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${RUN_DIR}" --episodes 100 --laps 5 \
    2>&1 | tee -a "${RUN_DIR}/eval.log"

log "Done"
echo "  results   ${RUN_DIR}/evaluation.txt"
echo "  metrics   ${RUN_DIR}/metrics.jsonl"
echo "  weights   ${RUN_DIR}/ckpt"
