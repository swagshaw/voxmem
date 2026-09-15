#!/bin/bash
# One array task per (model, config) shard. Submitted by submit_slurm.sh, which
# writes the task table this reads -- running it by hand against a stale table
# evaluates the wrong shard.
#
# Sharding by config rather than by model is deliberate: a shard is 799 items,
# and it keeps 64K -- up to a few hundred clips and tens of minutes of audio per
# item -- from holding a whole model's progress hostage. The cost is reloading
# the weights once per config instead of once per model, which is minutes
# against hours.

#SBATCH --job-name=voxmem
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -uo pipefail

: "${TASK_FILE:?submit via submit_slurm.sh}"
: "${REPO_DIR:?submit via submit_slurm.sh}"
: "${OUT_DIR:?submit via submit_slurm.sh}"

LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
LINE=$(sed -n "${LINE_NO}p" "$TASK_FILE")
if [[ -z "$LINE" ]]; then
    echo "[FAIL] no task on line $LINE_NO of $TASK_FILE"
    exit 2
fi
read -r MODEL CONFIG ADAPTER MODEL_DIR <<< "$LINE"

TAG="${MODEL##*:}"
PRED="$OUT_DIR/predictions_${TAG}_${CONFIG}.jsonl"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "job      ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}"
echo "node     $(hostname)"
echo "gpu      $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "model    $MODEL"
echo "config   $CONFIG"
echo "shard    $PRED"
echo "started  $(date)"
echo "=============================================================="

ARGS=(--config "$CONFIG" --out "$PRED" --model "$MODEL" --allow-abstention)
[[ "$ADAPTER"   != "-" ]] && ARGS+=(--adapter "$ADAPTER")
[[ "$MODEL_DIR" != "-" ]] && ARGS+=(--model-dir "$MODEL_DIR")

# Stop between items with a margin left, so the wall clock cannot SIGKILL the
# process part-way through writing one. Resubmitting continues the shard.
if [[ -n "${STOP_AFTER:-}" ]]; then
    ARGS+=(--stop-after-seconds "$STOP_AFTER")
fi
[[ -n "${STAGING_DIR:-}" ]] && ARGS+=(--staging-dir "$STAGING_DIR")

python3 "$REPO_DIR/run_voxmembench.py" "${ARGS[@]}"
STATUS=$?

echo "=============================================================="
echo "exit     $STATUS"
echo "lines    $( [[ -f "$PRED" ]] && wc -l < "$PRED" || echo 0 ) in $(basename "$PRED")"
echo "finished $(date)"
echo "=============================================================="
exit "$STATUS"
