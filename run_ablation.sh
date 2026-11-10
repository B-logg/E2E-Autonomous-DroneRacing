#!/usr/bin/env bash
# Where does the gate-crossing spread come from?  Score one checkpoint with each
# source of randomness switched off in turn, on the same flights as a plain
# run_compare.sh score, with no training.
#
#   ./run_ablation.sh 7k2.78M=logdir/small-team10-.../snapshots/step_000002779840
#
#   A0_none        every disturbance and sensor noise off (what is left is the
#                  policy, the mask and the airframe draw)
#   A1_no_accel    the slow acceleration disturbance off (Table III eps_a)
#   A2_no_moment   the slow and fast moment disturbances off (eps_M)
#   A3_no_motor    the per-step motor command noise off (eps_u)
#   A4_no_sensor   gyro and RPM telemetry noise off; disturbances as usual
#
#   --track T      default team44up (run 7k's layout)
#   --episodes N   default 400        --laps "L"  default "3"
#   --seed S       default 20000 (the seed of compare_7k_2.78M_400, so every
#                  ablation pairs flight by flight with that baseline)
#   --only "A1_no_accel A3_no_motor"   run a subset
#   --out DIR      default ablation_<NAME>
#
# All of them run at once (each evaluation caps itself at 30% of GPU memory and
# uses ~3 GB; five fit on a 24 GB card).  Results: <out>/<A*>/<NAME>/... as
# run_compare.sh writes them, logs in <out>/<A*>.log.  Bring them home with
#   ssh ... "cd ~/E2E-Autonomous-DroneRacing && tar cf - --exclude=ckpt <out>" | tar xf -
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

TRACK=team44up
EPISODES=400
LAPS="3"
SEED=20000
ONLY=""
OUT=""
RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --track)    TRACK="$2"; shift ;;
    --episodes) EPISODES="$2"; shift ;;
    --laps)     LAPS="$2"; shift ;;
    --seed)     SEED="$2"; shift ;;
    --only)     ONLY="$2"; shift ;;
    --out)      OUT="$2"; shift ;;
    -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
    *=*)        [ -z "${RUN}" ] || { echo "one NAME=PATH only" >&2; exit 2; }; RUN="$1" ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "${RUN}" ] || { echo "give NAME=PATH (./run_ablation.sh --help)" >&2; exit 2; }
NAME="${RUN%%=*}"
OUT="${OUT:-ablation_${NAME}}"

spec() {
  case "$1" in
    A0_none)      echo "--dist-scale 0,0,0,0 --gyro-noise 0,0 --rpm-noise 0" ;;
    A1_no_accel)  echo "--dist-scale 0,1,1,1" ;;
    A2_no_moment) echo "--dist-scale 1,0,0,1" ;;
    A3_no_motor)  echo "--dist-scale 1,1,1,0" ;;
    A4_no_sensor) echo "--gyro-noise 0,0 --rpm-noise 0" ;;
    *) return 1 ;;
  esac
}
ORDER="A0_none A1_no_accel A2_no_moment A3_no_motor A4_no_sensor"
[ -n "${ONLY}" ] && ORDER="${ONLY}"

mkdir -p "${OUT}"
pids=()
for a in ${ORDER}; do
  args="$(spec "${a}")" || { echo "unknown ablation ${a}" >&2; exit 2; }
  printf '==> %s: %s\n' "${a}" "${args}"
  ./run_compare.sh --out "${OUT}/${a}" --track "${TRACK}" --laps "${LAPS}" \
      --episodes "${EPISODES}" --seed "${SEED}" --eval-args "${args}" "${RUN}" \
      > "${OUT}/${a}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for a in ${ORDER}; do
  printf '\n=== %s (%s)\n' "${a}" "$(spec "${a}")"
  grep -A3 '^=== 3 laps' "${OUT}/${a}/report.txt" 2>/dev/null | tail -2 || echo "  no report: see ${OUT}/${a}.log"
done
[ "${fail}" = 0 ] || { echo "some ablation failed: see ${OUT}/*.log" >&2; exit 1; }
echo "==> Done.  ${OUT}/"
