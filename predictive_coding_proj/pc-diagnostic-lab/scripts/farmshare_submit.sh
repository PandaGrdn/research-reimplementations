#!/bin/bash
# Submit the diagnostic lab to a Slurm cluster (written for Stanford FarmShare).
#
# Each experiment becomes: 1 warmup job (dictionary cache) -> 1 array with one
# task per job, one core each -> 1 merge job. All tasks of an experiment write
# into results/<exp>/$TAG, so a resubmit only reruns what is missing.
#
#   bash scripts/farmshare_submit.sh setup            # once, on a login node: venv + dataset
#   bash scripts/farmshare_submit.sh pilot            # one 1/10-scale job, to time a core
#   bash scripts/farmshare_submit.sh run c1 c3 c2 c8  # full scale (c3 is read out of c1)
#   bash scripts/farmshare_submit.sh run c7
#   squeue -u $USER                                   # watch
#
# Knobs (environment variables):
#   SEEDS="0 1 2"  TAG=farmshare  PARTITION=normal  TIME=2-00:00:00  MEM=8G  QOS=
#   EXTRA="--overlay configs/reporting.yaml"   flags passed to every run_parallel call
#   VENV=$HOME/pc-lab-venv   PYMOD=python/3.13.0
#   DRY_RUN=1                print the sbatch commands instead of submitting
#
# There is no mid-job checkpoint: TIME must exceed one job. Run `pilot` first;
# a full job is about 10x the pilot's elapsed time. If that is over the normal
# limit, submit that experiment with QOS=long.
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB"
SEEDS="${SEEDS:-0 1 2}"
TAG="${TAG:-farmshare}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-2-00:00:00}"
MEM="${MEM:-8G}"
QOS="${QOS:-}"
EXTRA="${EXTRA:---overlay configs/reporting.yaml}"
VENV="${VENV:-$HOME/pc-lab-venv}"
PYMOD="${PYMOD:-python/3.13.0}"
DRY_RUN="${DRY_RUN:-0}"
# The venv's python by path, so the login-node steps work without activating it.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$VENV/bin/python" ]; then PYTHON="$VENV/bin/python"; else PYTHON=python; fi
fi
mkdir -p results/logs

# Runs inside every job: environment, and one BLAS thread per allocated core.
PREAMBLE="cd '$LAB'; if [ -f '$VENV/bin/activate' ]; then . '$VENV/bin/activate'; fi; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}"
RUN="$PYTHON scripts/run_parallel.py"

exp_name() {
  case "$1" in
    c1) echo c1_rollout_collapse ;; c2) echo c2_mitigation_grid ;; c3) echo c3_copy_detection ;;
    c7) echo c7_slowness_sweep ;;   c8) echo c8_amortized_contrast ;;
    *) echo "unknown claim $1 (use c1 c2 c3 c7 c8)" >&2; exit 2 ;;
  esac
}

# submit NAME TIME [sbatch args...] -- COMMAND ; prints the job id
FAKE_ID_FILE="$(mktemp)"; echo 1000 > "$FAKE_ID_FILE"; trap 'rm -f "$FAKE_ID_FILE"' EXIT
submit() {
  local name="$1" time="$2"; shift 2
  local args=()
  while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  local cmd="$PREAMBLE; $*"
  local full=(sbatch --parsable --job-name="pc-$name" --partition="$PARTITION" --time="$time"
              --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$MEM" --output="results/logs/%x-%j.out")
  if [ -n "$QOS" ]; then full+=(--qos="$QOS"); fi
  full+=(${args[@]+"${args[@]}"} --wrap "$cmd")
  if [ "$DRY_RUN" = 1 ]; then
    local fake=$(( $(cat "$FAKE_ID_FILE") + 1 )); echo "$fake" > "$FAKE_ID_FILE"
    printf 'DRY_RUN %s:' "$fake" >&2; printf ' %q' "${full[@]}" >&2; printf '\n' >&2
    echo "$fake"
  else
    "${full[@]}" | cut -d';' -f1
  fi
}

cmd_setup() {
  if command -v module >/dev/null 2>&1; then module load "$PYMOD"; fi
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  # CPU-only wheels: the jobs are CPU-bound and the CUDA wheels are several GB.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install numpy matplotlib pyyaml
  # Fetch Moving MNIST here, in case compute nodes have no internet.
  python -c "import sys; sys.path.insert(0, '.'); from src.config import resolve_config; from src.data import _ensure_moving_mnist; from pathlib import Path; c = resolve_config(); _ensure_moving_mnist((Path('.') / c['data']['root']).resolve()); print('dataset ready')"
  python tests/test_parallel.py
  echo "setup done. next: bash scripts/farmshare_submit.sh pilot"
}

cmd_pilot() {
  local dir="results/c2_mitigation_grid/${TAG}_pilot"
  local id
  id=$(submit pilot 08:00:00 -- "$RUN c2 --cells baseline --seeds 0 --n-train 100 --workers 1 --device cpu --resume $dir --no-latest")
  echo "pilot job $id (1/10 of one full job)."
  echo "when it finishes:  sacct -j $id --format=JobID,Elapsed,MaxRSS,State"
  echo "one full-scale job takes about 10x that Elapsed; MaxRSS tells you what MEM to ask for."
}

cmd_run() {
  [ $# -gt 0 ] || { echo "usage: $0 run c1 [c2 c3 c7 c8]" >&2; exit 2; }
  local c1_merge="" warm_frozen=""
  # c1 first, so c3 can depend on it whatever order was typed
  local ordered=() c
  for c in "$@"; do [ "$c" = c1 ] && ordered+=(c1); done
  for c in "$@"; do [ "$c" != c1 ] && ordered+=("$c"); done
  for c in "${ordered[@]}"; do
    local exp; exp=$(exp_name "$c")
    local dir="results/$exp/$TAG"
    if [ "$c" = c3 ] && [ "${C3_TRAIN:-0}" != 1 ]; then
      local c1dir="results/c1_rollout_collapse/$TAG" dep=()
      if [ -n "$c1_merge" ]; then dep=(--dependency="afterok:$c1_merge")
      elif [ ! -f "$c1dir/metrics.json" ] && [ "$DRY_RUN" != 1 ]; then
        echo "c3 is read out of a finished c1 run, but $c1dir/metrics.json is missing. Submit c1 with it, or set C3_TRAIN=1." >&2; exit 2
      fi
      local id; id=$(submit c3 00:30:00 ${dep[@]+"${dep[@]}"} -- "$PYTHON experiments/c3_copy_detection.py --from-c1 $c1dir --seeds $SEEDS --device cpu $EXTRA")
      echo "c3: job $id reads C3 out of $c1dir (no training)"
      continue
    fi
    local n; n=$($RUN "$c" --seeds $SEEDS --count-jobs $EXTRA)
    local warm arr merge
    # Create the run dir once, here. Many array tasks creating it at the same
    # moment on NFS fail with "File exists" (seen on FarmShare).
    [ "$DRY_RUN" = 1 ] || mkdir -p "$dir/shards"
    # c1 / c2 / c8 share one frozen-dictionary cache entry per seed; warm it once.
    # c7 unfreezes the dictionary, which is a different cache key.
    if [ "$c" != c7 ] && [ -n "$warm_frozen" ]; then
      warm="$warm_frozen"
    else
      warm=$(submit "$c-warm" 02:00:00 -- "$RUN $c --seeds $SEEDS --device cpu --warmup-only $EXTRA")
      [ "$c" != c7 ] && warm_frozen="$warm"
    fi
    arr=$(submit "$c" "$TIME" --array="0-$((n - 1))" --dependency="afterok:$warm" -- \
      "$RUN $c --seeds $SEEDS --device cpu --workers 1 --resume $dir --shard \${SLURM_ARRAY_TASK_ID}/$n $EXTRA")
    merge=$(submit "$c-merge" 00:30:00 --dependency="afterok:$arr" -- "$RUN $c --seeds $SEEDS --merge-only $dir $EXTRA")
    [ "$c" = c1 ] && c1_merge="$merge"
    echo "$c: $n tasks -> $dir   (warmup $warm, array $arr, merge $merge)"
  done
  echo "watch: squeue -u \$USER     logs: results/logs/     failed tasks: resubmit the same command, finished shards are skipped"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  pilot) shift; cmd_pilot "$@" ;;
  run)   shift; cmd_run "$@" ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
