#!/usr/bin/env bash
# Jetson benchmark:  ./run.sh            full GPU/CPU tests (about 30-60 min, mostly TensorRT engine builds)
#                    ./run.sh --quick    only the chosen GateNet (f4 240x320) + fallback 480x640
#                    add --uart /dev/ttyTHS1 --baud 500000 for the FC link, or --only-uart for just that
# Results: results/<host>-<time>/ and the same folder packed as results/<host>-<time>.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
ls onnx/policy_step.onnx >/dev/null 2>&1 || { echo "onnx/ missing: copy the whole jetson_bench folder"; exit 1; }
if ! command -v trtexec >/dev/null && [ ! -x /usr/src/tensorrt/bin/trtexec ]; then
  echo "WARNING: trtexec (TensorRT) not found; only the PyTorch/OpenCV parts will run"
fi
python3 bench.py "$@"
latest=$(ls -dt results/*/ | head -1); latest=${latest%/}
tar -czf "${latest}.tar.gz" --exclude='engines' -C results "$(basename "$latest")"
echo
echo "Packed: $(pwd)/${latest}.tar.gz   <- send this file back"
