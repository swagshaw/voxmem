#!/bin/bash
# Build the (model, config) task table and submit it as one job array.
#
#   scripts/submit_slurm.sh --models models.tsv --configs 32k \
#       --partition gpu-a100 --time 04:00:00 --gpus 1 --mem 80G
#
# models.tsv is one line per model, whitespace separated, "-" where unused:
#
#   qwen3omni   /path/qwen3omni_interleaved.py   /path/to/weights
#   phi4        /path/phi4mm_interleaved.py      /path/to/weights
#   gemini:gemini-2.5-flash   -                  -
#
# --dry-run prints the table and the sbatch line without submitting, which is
# the sane thing to do first: a wrong table submits the wrong work at scale.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"

MODELS_FILE=""
CONFIGS="32k"
OUT_DIR="$PWD/results"
PARTITION=""
TIME="04:00:00"
GPUS="1"
MEM="80G"
CPUS="8"
ACCOUNT=""
THROTTLE=""
DRY_RUN=0
STOP_MARGIN=900          # stop this many seconds before the wall clock

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)     MODELS_FILE="$2"; shift 2 ;;
        --configs)    CONFIGS="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --throttle)   THROTTLE="$2"; shift 2 ;;
        --stop-margin) STOP_MARGIN="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MODELS_FILE" && -f "$MODELS_FILE" ]] || {
    echo "--models must point at a models table" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TASK_FILE="$OUT_DIR/tasks.tsv"
: > "$TASK_FILE"

IFS=',' read -ra CONFIG_LIST <<< "$CONFIGS"
while read -r MODEL ADAPTER MODEL_DIR _rest; do
    [[ -z "${MODEL:-}" || "$MODEL" == \#* ]] && continue
    for CONFIG in "${CONFIG_LIST[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "$MODEL" "$CONFIG" \
            "${ADAPTER:--}" "${MODEL_DIR:--}" >> "$TASK_FILE"
    done
done < "$MODELS_FILE"

N=$(wc -l < "$TASK_FILE")
(( N > 0 )) || { echo "no tasks built from $MODELS_FILE" >&2; exit 2; }

# Leave a margin so the runner stops between items rather than being killed.
HOURS=${TIME%%:*}
REST=${TIME#*:}
MINUTES=${REST%%:*}
LIMIT=$(( 10#$HOURS * 3600 + 10#$MINUTES * 60 ))
STOP_AFTER=$(( LIMIT - STOP_MARGIN ))
(( STOP_AFTER > 60 )) || STOP_AFTER=$(( LIMIT / 2 ))

ARRAY="0-$(( N - 1 ))"
[[ -n "$THROTTLE" ]] && ARRAY="$ARRAY%$THROTTLE"

SBATCH_ARGS=(--array="$ARRAY" --time="$TIME" --cpus-per-task="$CPUS"
             --mem="$MEM"
             --output="$OUT_DIR/slurm-%A_%a.out"
             --error="$OUT_DIR/slurm-%A_%a.err")
[[ -n "$PARTITION" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "$ACCOUNT"   ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ "$GPUS" != "0"  ]] && SBATCH_ARGS+=(--gres=gpu:"$GPUS")

echo "tasks ($N):"
cat -n "$TASK_FILE"
echo
echo "sbatch ${SBATCH_ARGS[*]} $HERE/slurm_array.sbatch"
echo "  TASK_FILE=$TASK_FILE"
echo "  OUT_DIR=$OUT_DIR"
echo "  STOP_AFTER=${STOP_AFTER}s of a ${LIMIT}s limit"

if (( DRY_RUN )); then
    echo
    echo "dry run: nothing submitted"
    exit 0
fi

export TASK_FILE OUT_DIR REPO_DIR STOP_AFTER
sbatch "${SBATCH_ARGS[@]}" \
    --export=ALL,TASK_FILE,OUT_DIR,REPO_DIR,STOP_AFTER,STAGING_DIR \
    "$HERE/slurm_array.sbatch"
