#!/bin/bash
# Run one model over several configs, then score each.
#
#   scripts/run_matrix.sh --model gemini:gemini-2.5-flash --configs 8k,16k,32k,64k
#   scripts/run_matrix.sh --model qwen3omni --adapter /path/qwen3omni_interleaved.py \
#       --model-dir /path/to/weights --configs 32k
#
# Every step resumes, so rerunning the same command after an interruption picks
# up where it stopped rather than starting over.

set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="abstain"
CONFIGS="8k,16k,32k,64k"
OUT_DIR="results"
JUDGE=""
ADAPTER=""
MODEL_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --configs)    CONFIGS="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --judge)      JUDGE="$2"; shift 2 ;;
        --adapter)    ADAPTER="$2"; shift 2 ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        --)           shift; EXTRA+=("$@"); break ;;
        *)            EXTRA+=("$1"); shift ;;
    esac
done

# A filename-safe form of the model spec: "gemini:gemini-2.5-flash" -> the tail.
TAG="${MODEL##*:}"
TAG="${TAG//\//_}"
mkdir -p "$OUT_DIR"

RUN_ARGS=(--model "$MODEL" --allow-abstention)
[[ -n "$ADAPTER"   ]] && RUN_ARGS+=(--adapter "$ADAPTER")
[[ -n "$MODEL_DIR" ]] && RUN_ARGS+=(--model-dir "$MODEL_DIR")
(( ${#EXTRA[@]} )) && RUN_ARGS+=("${EXTRA[@]}")

STATUS=0
IFS=',' read -ra LIST <<< "$CONFIGS"
for CONFIG in "${LIST[@]}"; do
    PRED="$OUT_DIR/predictions_${TAG}_${CONFIG}.jsonl"
    echo "=============================================================="
    echo "model   $MODEL"
    echo "config  $CONFIG"
    echo "out     $PRED"
    echo "=============================================================="
    python3 "$HERE/run_voxmembench.py" --config "$CONFIG" --out "$PRED" \
        "${RUN_ARGS[@]}" || { STATUS=$?; echo "[FAIL] $CONFIG run exited $STATUS"; continue; }

    if [[ -n "$JUDGE" ]]; then
        python3 "$HERE/score_voxmembench.py" --predictions "$PRED" \
            --judge "$JUDGE" \
            --out "$OUT_DIR/metrics_${TAG}_${CONFIG}.json" \
            || { STATUS=$?; echo "[FAIL] $CONFIG scoring exited $STATUS"; }
    fi
done

echo
echo "predictions and reports are in $OUT_DIR"
exit "$STATUS"
