#!/usr/bin/env bash
#
# One command, from a fresh clone to a trained and evaluated SkyDreamer policy.
#
#   ./run.sh --smoke      ~5 min   verify the whole pipeline works. DO THIS FIRST.
#   ./run.sh              ~30-50 h the paper's run: 17M steps, 3 phases, then eval
#   ./run.sh --resume              continue the newest run after an interruption
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
    --resume)     MODE="resume" ;;
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
for tool in git curl; do
  command -v "${tool}" >/dev/null 2>&1 || die \
    "${tool} is missing. On a bare NVIDIA CUDA image:  apt-get update && apt-get install -y git curl"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no nvidia-smi found -- will fall back to CPU (fine for --smoke, useless for a real run)"
fi

# The replay buffer is the real resource constraint, and it is easy to miss.
# The paper uses replay.size = 10e6 steps (31 h of flight).  Each step stores a
# 64x64 mask plus the vectors, 4596 bytes, and DreamerV3 keeps the working set
# in RAM (embodied/core/replay.py: self.chunks) while also writing it to disk.
REPLAY_STEPS=10000000
BYTES_PER_STEP=4596
NEED_GB=$(( REPLAY_STEPS * BYTES_PER_STEP / 1000000000 ))
DISK_GB=$(df -Pk "${ROOT}" | awk 'NR==2 {printf "%d", $4/1000000}')
if command -v free >/dev/null 2>&1; then
  RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
else
  RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1000000000 ))
fi
echo "replay buffer needs ~${NEED_GB} GB RAM and ~${NEED_GB} GB disk (replay.size ${REPLAY_STEPS})"
echo "  available: ${RAM_GB} GB RAM, ${DISK_GB} GB disk free at ${ROOT}"

if [ "${MODE}" != "smoke" ] && [ "${MODE}" != "setup" ]; then
  if [ "${DISK_GB}" -lt $(( NEED_GB + 20 )) ]; then
    die "not enough disk. Need ~$(( NEED_GB + 20 )) GB (replay ${NEED_GB} + venv/CUDA wheels ~15).
     On vast.ai the disk slider is set when you rent -- raise it to 150 GB and
     re-create the instance; it costs about \$0.05/hr.
     To train on this disk anyway, shrink the buffer (this DEVIATES from the paper):
       ./run.sh --replay.size 3e6"
  fi
  if [ "${RAM_GB}" -lt $(( NEED_GB + 12 )) ]; then
    echo ""
    echo "WARNING: ${RAM_GB} GB RAM for a ~${NEED_GB} GB replay buffer plus JAX and 16 env"
    echo "  workers is likely to OOM part-way through. Prefer a 128 GB machine, or"
    echo "  shrink the buffer with  ./run.sh --replay.size 5e6  (deviates from the paper)."
    echo ""
  fi
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
if [ "${MODE}" = "resume" ]; then
  RUN_DIR="$(ls -td "${LOGROOT}"/*/ 2>/dev/null | head -1 || true)"
  [ -n "${RUN_DIR}" ] || die "no run to resume under ${LOGROOT}"
  RUN_DIR="${RUN_DIR%/}"
  log "Resuming ${RUN_DIR}"
  echo "DreamerV3 picks up from the checkpoint in this logdir; phases re-run"
  echo "cheaply because run.steps is a cumulative ceiling."
else
  RUN_DIR="${LOGROOT}/${PRESET}-$(date +%Y%m%d-%H%M%S)"
  log "Training -> ${RUN_DIR}"
  echo "Paper: ~50 h on 4/7 of an A100 80GB; a full A100 should do it in ~30 h."
  echo "Detach this (tmux/screen/nohup) -- and if the instance dies, ./run.sh --resume"
fi
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
