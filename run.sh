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

# DreamerV3's `jax.platform` defaults to cuda and embodied.jax.setup() applies
# it for real, so it has to match the machine.
JAX_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no nvidia-smi found -- running on CPU (fine for --smoke, useless for a real run)"
  JAX_ARGS=(--jax.platform cpu)
fi

# Two separate resource constraints, and they are not the same size.
#
# RAM: DreamerV3 holds the replay working set uncompressed (4596 B/step,
# embodied/core/replay.py: self.chunks), so replay.size 10e6 is ~46 GB of RAM.
#
# DISK: chunks are written with np.savez_compressed but the binary masks
# barely compress once the policy actually flies -- measured 3782 B/step on a
# real run, only 1.2x. Worse, DreamerV3 never deletes old chunk files, so the
# directory grows with the whole run, not the buffer window.
#
# The two are INDEPENDENT.  `replay.size` is the algorithm: how far back the
# world model can sample, and it lives in RAM.  The files under logdir/replay
# exist only so a restart can refill that buffer -- `Replay.save()` returns no
# manifest and `Replay.load()` just reads whatever *.npz files are present,
# newest first, up to `capacity`.  Finding fewer is not an error, it simply
# recovers less.
#
# So on a small disk, keep the paper's replay.size and shrink only the mirror:
#   SKYDREAMER_REPLAY_DISK_STEPS=4e6 ./run.sh
# That is ~19 GB of disk with the buffer still at the paper's 10e6 in RAM.
# The only thing given up is how much buffer survives a crash.
REPLAY_STEPS=10000000
TOTAL_STEPS=17000000
RAM_NEED_GB=$(( REPLAY_STEPS * 4596 / 1000000000 ))
# With the pruner running, disk is bounded by the buffer window (plus its
# 1.25 margin) rather than by TOTAL_STEPS, plus ~9 GB of venv and CUDA wheels.
DISK_STEPS=$(printf '%.0f' "${SKYDREAMER_REPLAY_DISK_STEPS:-${REPLAY_STEPS}}")
DISK_NEED_GB=$(( DISK_STEPS * 3782 * 5 / 4 / 1000000000 + 9 ))
DISK_GB=$(df -Pk "${ROOT}" | awk 'NR==2 {printf "%d", $4/1000000}')
# `free` inside a container reports the HOST's memory, not the cgroup limit --
# a vast.ai box with an 80 GB allocation happily prints 629 GB. Read the limit
# the kernel will actually enforce, and only fall back to `free` when there
# isn't one.
RAM_BYTES=""
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "${f}" ] || continue
  v=$(cat "${f}" 2>/dev/null)
  case "${v}" in
    ''|max|*[!0-9]*) continue ;;
  esac
  # cgroup v1 reports a huge sentinel when unlimited
  [ "${v}" -lt 1000000000000000 ] && RAM_BYTES="${v}" && break
done
if [ -n "${RAM_BYTES}" ]; then
  RAM_GB=$(( RAM_BYTES / 1000000000 ))
  RAM_SRC="cgroup limit"
elif command -v free >/dev/null 2>&1; then
  RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
  RAM_SRC="free (no cgroup limit found)"
else
  RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1000000000 ))
  RAM_SRC="sysctl"
fi
echo "needs ~${RAM_NEED_GB} GB RAM (buffer ${REPLAY_STEPS} steps) and ~${DISK_NEED_GB} GB disk (mirror ${DISK_STEPS} steps)"
echo "  available: ${RAM_GB} GB RAM (${RAM_SRC}), ${DISK_GB} GB disk free at ${ROOT}"

if [ "${MODE}" != "smoke" ] && [ "${MODE}" != "setup" ]; then
  if [ "${DISK_GB}" -lt "${DISK_NEED_GB}" ]; then
    die "not enough disk: ${DISK_GB} GB free, need ~${DISK_NEED_GB} GB.
     Shrink the on-disk mirror instead -- this does NOT change the paper's
     settings, it only limits how much buffer survives a restart:
       SKYDREAMER_REPLAY_DISK_STEPS=4e6 ./run.sh --resume
     Shrinking the buffer itself (--replay.size) WOULD deviate from the paper."
  fi
  if [ "${DISK_GB}" -lt $(( DISK_NEED_GB * 2 )) ]; then
    echo "NOTE: ${DISK_GB} GB disk is enough but not generous. Watch it with"
    echo "  watch -n600 df -h ${ROOT}"
  fi
  if [ "${RAM_GB}" -lt $(( RAM_NEED_GB + 12 )) ]; then
    echo ""
    echo "WARNING: ${RAM_GB} GB RAM for a ~${RAM_NEED_GB} GB replay buffer plus JAX and 16 env"
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
# Stamp the checkout with the patch it was built from.  `git pull` updates the
# patch file but cannot touch an already-patched third_party/, so without this
# a pulled fix is silently ignored and the run uses stale code.
if command -v sha256sum >/dev/null 2>&1; then
  PATCH_HASH=$(sha256sum "${ROOT}/patches/informed_dreamer.patch" | cut -d' ' -f1)
else
  PATCH_HASH=$(shasum -a 256 "${ROOT}/patches/informed_dreamer.patch" | cut -d' ' -f1)
fi
STAMP="${DV3_DIR}/.skydreamer-patch"

if [ -d "${DV3_DIR}/.git" ] && [ "$(cat "${STAMP}" 2>/dev/null)" != "${PATCH_HASH}" ]; then
  echo "patch has changed since this checkout was made -- recreating it"
  rm -rf "${DV3_DIR}"
fi

if [ ! -d "${DV3_DIR}/.git" ]; then
  mkdir -p "$(dirname "${DV3_DIR}")"
  git clone --quiet https://github.com/danijar/dreamerv3.git "${DV3_DIR}"
  git -C "${DV3_DIR}" checkout --quiet --detach "${DV3_COMMIT}"
  git -C "${DV3_DIR}" apply "${ROOT}/patches/informed_dreamer.patch"
  echo "${PATCH_HASH}" > "${STAMP}"
  echo "patched (${PATCH_HASH:0:12})"
else
  echo "already present and up to date (${PATCH_HASH:0:12})"
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

# DreamerV3's `logfn` stacks every uint8 ndim==3 observation and the `scope`
# output writes it out as one mp4 per episode.  Our 64x64 mask matches, so a
# long run quietly fills the disk with videos of segmentation masks.  Keep the
# jsonl outputs (metrics.jsonl / scores.jsonl are all we read) and drop scope.
LOG_ARGS=(--logger.outputs jsonl)

# `run.envs` is DreamerV3's, not the paper's (it never states a worker count),
# so it is free to match the machine.  Each worker is a spawned process running
# the environment on CPU; more of them than cores just thrashes.
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)
ENVS=$(( CORES < 16 ? CORES : 16 ))
ENV_ARGS=(--run.envs "${ENVS}")
echo "env workers: ${ENVS} (machine has ${CORES} cores)"

export PYTHONPATH="${DV3_DIR}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# `--run.envs 16` spawns sixteen environment processes.  The environment pins
# its own computation to CPU (skydreamer/embodied_env.py), but a spawned worker
# can still open a CUDA context, so leave the trainer some headroom instead of
# letting it preallocate almost the whole card.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"

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
      --run.log_every 20 --run.report_every 1e9 --run.save_every 500 \
      "${JAX_ARGS[@]+"${JAX_ARGS[@]}"}" "${LOG_ARGS[@]}"
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

# DreamerV3 never deletes replay chunks; without this the disk fills mid-run.
"${PY}" "${ROOT}/scripts/prune_replay.py" --logdir "${RUN_DIR}" \
    --keep-steps "${DISK_STEPS}" --interval 600 >> "${RUN_DIR}/prune.log" 2>&1 &
PRUNER=$!
trap 'kill ${PRUNER} 2>/dev/null || true' EXIT

"${PY}" "${ROOT}/scripts/train.py" \
    --logdir "${RUN_DIR}" --preset "${PRESET}" --jax.prealloc False \
    "${JAX_ARGS[@]+"${JAX_ARGS[@]}"}" "${LOG_ARGS[@]}" "${ENV_ARGS[@]}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"

log "Evaluating in simulation"
"${PY}" "${ROOT}/scripts/evaluate.py" --logdir "${RUN_DIR}" --episodes 100 --laps 5 \
    2>&1 | tee -a "${RUN_DIR}/eval.log"

log "Done"
echo "  results   ${RUN_DIR}/evaluation.txt"
echo "  metrics   ${RUN_DIR}/metrics.jsonl"
echo "  weights   ${RUN_DIR}/ckpt"
