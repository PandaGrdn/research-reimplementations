#!/bin/bash
# Submit the diagnostic lab to a Slurm cluster (written for Stanford FarmShare).
#
#   bash scripts/farmshare_submit.sh setup                    # once, on a login node
#   bash scripts/farmshare_submit.sh pilot                    # one 1/10-scale job, to time a core
#   bash scripts/farmshare_submit.sh run c1 c3 c2 c8          # Moving MNIST spine
#   DATA=kth TAG=farmshare_kth bash scripts/farmshare_submit.sh run c1 c3 c2 c8 c5 c6 c9 c11 c12 c13 c13on1
#   squeue -u $USER
#
# Experiments:
#   c1 c2 c7 c8   fanned out by scripts/run_parallel.py: warmup job -> array (one
#                 single-core task per seed or grid cell) -> merge job
#   c3            read out of this TAG's finished c1 run (no training)
#   c5 c6 c9 c11  one job each
#   c12 c13       one job each, after c11 when c11 is in the same call
#   c13on1        C13's consistency settling on this TAG's C1 models, after c1
#
# Every job requeues itself 30 minutes before its time limit and resumes from
# its last finished training epoch (src/experiment.py, train_ckpt/), so jobs
# longer than the queue's limit just take several turns. Resubmitting the same
# command reruns only what is missing.
#
# Knobs (environment variables):
#   DATA=moving_mnist|bouncing_balls|kth   adds configs/data/$DATA.yaml
#   TAG=farmshare        results go to results/<exp>/$TAG
#   SEEDS="0 1 2"        seeds for c1 c2 c3 c7 c8 c11 c12 c13 c13on1 (c5 c6 c9 use 5 by default)
#   C2_CELLS="..."       subset of C2 cells, e.g. "baseline rollout_loss all_combined"
#   OVERLAYS="configs/reporting.yaml"      config overlays, before the DATA one
#   PARTITION=normal  TIME=2-00:00:00  MEM=8G  QOS=
#   VENV=$HOME/pc-lab-venv  PYMOD=python/3.12.3
#   DRY_RUN=1            print the sbatch commands instead of submitting
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB"
SEEDS="${SEEDS:-0 1 2}"
TAG="${TAG:-farmshare}"
DATA="${DATA:-moving_mnist}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-2-00:00:00}"
MEM="${MEM:-8G}"
QOS="${QOS:-}"
OVERLAYS="${OVERLAYS:-configs/reporting.yaml}"
C2_CELLS="${C2_CELLS:-}"
VENV="${VENV:-$HOME/pc-lab-venv}"
PYMOD="${PYMOD:-python/3.12.3}"
DRY_RUN="${DRY_RUN:-0}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$VENV/bin/python" ]; then PYTHON="$VENV/bin/python"; else PYTHON=python; fi
fi
if [ "$DATA" != moving_mnist ]; then
  [ -f "configs/data/$DATA.yaml" ] || { echo "no configs/data/$DATA.yaml" >&2; exit 2; }
  OVERLAYS="$OVERLAYS configs/data/$DATA.yaml"
fi
OV="--overlay $OVERLAYS"
mkdir -p results/logs

# Runs inside every job (POSIX sh). One BLAS thread per allocated core,
# unbuffered logs, and self-requeue: Slurm sends USR1 30 min before the time
# limit (--signal=B:USR1@1800); the trap requeues the job, which then resumes.
# A signal in the first 10 minutes is ignored: on FarmShare a short test job got
# it at start, and requeueing then would loop forever.
PREAMBLE="cd '$LAB'; if [ -f '$VENV/bin/activate' ]; then . '$VENV/bin/activate'; fi; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}; T0=\$(date +%s); echo \"start \$(date) restart=\${SLURM_RESTART_COUNT:-0}\"; trap 'if [ \$(( \$(date +%s) - T0 )) -gt 600 ]; then echo \"time limit near, requeueing \$(date)\"; scontrol requeue \$SLURM_JOB_ID; sleep 900; else echo \"ignoring early USR1 \$(date)\"; fi' USR1"
RUN="$PYTHON scripts/run_parallel.py"

exp_name() {
  case "$1" in
    c1) echo c1_rollout_collapse ;; c2) echo c2_mitigation_grid ;; c3) echo c3_copy_detection ;;
    c5) echo c5_latent_smoothness ;; c6) echo c6_noise_floor ;; c7) echo c7_slowness_sweep ;;
    c8) echo c8_amortized_contrast ;; c9) echo c9_predictability ;; c11) echo c11_offline_gru ;;
    c12) echo c12_energy_fade ;; c13|c13on1) echo c13_consistency_rollout ;;
    *) echo "unknown experiment $1" >&2; exit 2 ;;
  esac
}

FAKE_ID_FILE="$(mktemp)"; echo 1000 > "$FAKE_ID_FILE"; trap 'rm -f "$FAKE_ID_FILE"' EXIT
# submit NAME TIME [sbatch args...] -- COMMAND ; prints the job id
submit() {
  local name="$1" time="$2"; shift 2
  local args=()
  while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  # the command runs in the background so the USR1 trap can fire while it works
  local cmd="$PREAMBLE; $* & wait \$!"
  local full=(sbatch --parsable --job-name="pc-$name" --partition="$PARTITION" --time="$time"
              --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$MEM" --requeue --open-mode=append
              --signal=B:USR1@1800 --output="results/logs/%x-%j.out")
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
  . "$VENV/bin/activate"
  pip install --upgrade pip
  # CPU-only torch wheels: the jobs are CPU-bound and the CUDA wheels are several GB.
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  python -c "import sys; sys.path.insert(0, '.'); from src.config import resolve_config; from src.data import _ensure_moving_mnist; from pathlib import Path; c = resolve_config(); _ensure_moving_mnist((Path('.') / c['data']['root']).resolve()); print('dataset ready')"
  python tests/test_parallel.py
  echo "setup done. next: bash scripts/farmshare_submit.sh pilot"
}

cmd_pilot() {
  local dir="results/c2_mitigation_grid/${TAG}_pilot" id
  id=$(submit pilot 08:00:00 -- "$RUN c2 --cells baseline --seeds 0 --n-train 100 --workers 1 --device cpu --resume $dir --no-latest $OV")
  echo "pilot job $id (1/10 of one full job)."
  echo "when it finishes:  sacct -j $id --format=JobID,Elapsed,MaxRSS,State"
}

cmd_run() {
  [ $# -gt 0 ] || { echo "usage: $0 run c1 [c2 c3 c5 c6 c7 c8 c9 c11 c12 c13 c13on1]" >&2; exit 2; }
  local c1_merge="" c11_job="" warm_frozen="" c ordered=() x need_all=0
  # Experiments that load the frozen dictionary outside run_parallel (c5 c6 c9
  # use 5 seeds, c11 c12 c13 c13on1 use SEEDS) must wait for one warmup that
  # caches every seed they need, or several jobs pretrain the same file at once.
  for x in "$@"; do case "$x" in c5|c6|c9|c11|c12|c13|c13on1) need_all=1 ;; esac; done
  if [ "$need_all" = 1 ]; then
    local all_seeds; all_seeds=$(printf '%s\n' $SEEDS 0 1 2 3 4 | sort -n | uniq | tr '\n' ' ')
    warm_frozen=$(submit warm 04:00:00 -- "$RUN c1 --seeds $all_seeds --device cpu --warmup-only $OV")
    echo "warmup $warm_frozen caches dictionaries for seeds $all_seeds"
  fi
  # producers first so their dependents can wait on them, whatever order was typed
  for c in c1 c11; do for x in "$@"; do [ "$x" = "$c" ] && ordered+=("$c"); done; done
  for x in "$@"; do [ "$x" != c1 ] && [ "$x" != c11 ] && ordered+=("$x"); done

  for c in "${ordered[@]}"; do
    local exp dir id dep=()
    exp=$(exp_name "$c")
    dir="results/$exp/$TAG"
    case "$c" in
      c1|c2|c7|c8)
        local cells="" n warm arr merge
        [ "$c" = c2 ] && [ -n "$C2_CELLS" ] && cells="--cells $C2_CELLS"
        n=$($RUN "$c" --seeds $SEEDS --count-jobs $cells $OV)
        [ "$DRY_RUN" = 1 ] || mkdir -p "$dir/shards"  # one creator: many tasks racing on NFS fail
        if [ "$c" != c7 ] && [ -n "$warm_frozen" ]; then
          warm="$warm_frozen"  # c1, c2, c8 share the frozen-dictionary cache entry
        else
          warm=$(submit "$c-warm" 02:00:00 -- "$RUN $c --seeds $SEEDS --device cpu --warmup-only $OV")
          [ "$c" != c7 ] && warm_frozen="$warm"
        fi
        arr=$(submit "$c" "$TIME" --array="0-$((n - 1))" --dependency="afterok:$warm" -- \
          "$RUN $c --seeds $SEEDS --device cpu --workers 1 --resume $dir --shard \${SLURM_ARRAY_TASK_ID}/$n $cells $OV")
        merge=$(submit "$c-merge" 00:30:00 --dependency="afterok:$arr" -- "$RUN $c --seeds $SEEDS --merge-only $dir $cells $OV")
        [ "$c" = c1 ] && c1_merge="$merge"
        echo "$c: $n tasks -> $dir   (warmup $warm, array $arr, merge $merge)"
        ;;
      c3)
        local c1dir="results/c1_rollout_collapse/$TAG"
        if [ -n "$c1_merge" ]; then dep=(--dependency="afterok:$c1_merge")
        elif [ ! -f "$c1dir/metrics.json" ] && [ "$DRY_RUN" != 1 ]; then
          echo "c3 is read out of $c1dir, which has no metrics.json yet. Submit c1 in the same call." >&2; exit 2
        fi
        id=$(submit c3 00:30:00 ${dep[@]+"${dep[@]}"} -- "PC_LAB_RUN_DIR=$dir $PYTHON experiments/c3_copy_detection.py --from-c1 $c1dir --seeds $SEEDS --device cpu $OV")
        echo "c3: job $id reads C3 out of $c1dir"
        ;;
      c5|c6|c9)  # inference only; these scripts default to 5 seeds
        id=$(submit "$c" "$TIME" --dependency="afterok:$warm_frozen" -- "PC_LAB_RUN_DIR=$dir $PYTHON experiments/$exp.py --device cpu $OV")
        echo "$c: job $id -> $dir"
        ;;
      c11|c12|c13)
        if [ "$c" != c11 ] && [ -n "$c11_job" ]; then dep=(--dependency="afterok:$c11_job")
        else dep=(--dependency="afterok:$warm_frozen"); fi
        id=$(submit "$c" "$TIME" --mem=24G ${dep[@]+"${dep[@]}"} -- "PC_LAB_RUN_DIR=$dir $PYTHON experiments/$exp.py --seeds $SEEDS --device cpu $OV")
        [ "$c" = c11 ] && c11_job="$id"
        echo "$c: job $id -> $dir"
        ;;
      c13on1)
        local c1dir="results/c1_rollout_collapse/$TAG" ov="results/logs/c13on1_$TAG.yaml"
        printf 'c13:\n  kinds: [independent]\n  independent_ckpt: %s/seed{seed}/temporal.pt\n' "$c1dir" > "$ov"
        if [ -n "$c1_merge" ]; then dep=(--dependency="afterok:$c1_merge")
        else dep=(--dependency="afterok:$warm_frozen"); fi
        id=$(submit c13on1 "$TIME" --mem=16G ${dep[@]+"${dep[@]}"} -- "PC_LAB_RUN_DIR=results/$exp/${TAG}_on_c1 $PYTHON experiments/$exp.py --seeds $SEEDS --device cpu $OV $ov")
        echo "c13on1: job $id -> results/$exp/${TAG}_on_c1"
        ;;
    esac
  done
  echo "watch: squeue -u \$USER   logs: results/logs/   rerun the same command to fill in anything missing"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  pilot) shift; cmd_pilot "$@" ;;
  run)   shift; cmd_run "$@" ;;
  *) sed -n '2,32p' "$0"; exit 2 ;;
esac
