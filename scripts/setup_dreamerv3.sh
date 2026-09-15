#!/usr/bin/env bash
# Fetch DreamerV3 at the exact commit SkyDreamer cites (section III-A), apply
# the informed-decoder + smoothness-loss patch, and register the racing env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DV3="${ROOT}/third_party/dreamerv3"
COMMIT="cdf570902b1eaba193cc8ef69426cd4edde1b0bc"

if [ -d "${DV3}/.git" ]; then
  echo "already present at ${DV3}; delete it to re-run"
else
  mkdir -p "$(dirname "${DV3}")"
  git clone https://github.com/danijar/dreamerv3.git "${DV3}"
  git -C "${DV3}" checkout --detach "${COMMIT}"
  git -C "${DV3}" apply "${ROOT}/patches/informed_dreamer.patch"
  echo "patched dreamerv3 @ ${COMMIT}"
fi

# The patch registers `skydreamer` in dreamerv3/main.py's make_env ctor table,
# which imports `skydreamer.embodied_env`. Nothing else to wire up -- the
# package just has to be importable.

cat <<EOF

done.

  pip install -e "${ROOT}"
  pip install -r "${DV3}/requirements.txt"

train phase 1 (paper III-A: 17M steps, ~50 h on one A100):

  python "${DV3}/dreamerv3/main.py" \\
      --configs skydreamer size12m \\
      --logdir ~/logdir/skydreamer/\$(date +%Y%m%d-%H%M%S)

The 3-phase schedule (batch_length 64->256 at 8M, entropy 3e-4->1e-5 and
lr 4e-5->2e-6 at 13M) is driven by scripts/train.py.
EOF
