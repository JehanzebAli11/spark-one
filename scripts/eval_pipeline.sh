#!/usr/bin/env bash
# Predict val + test, fit temperatures on val, write the eval report.
# usage: scripts/eval_pipeline.sh <name> <out_dir> <predict command prefix...>
#   e.g. scripts/eval_pipeline.sh "gemma e2b lora" runs/gemma-e2b-lora-v1/eval \
#          python -m spark_one.hf_predict --model google/gemma-4-E2B-it --adapter runs/gemma-e2b-lora-v1/final_adapter
set -euo pipefail
NAME="$1"; OUT="$2"; shift 2
BANK="${BANK:-data/generated/bank}"
LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$EVAL_LIMIT"); fi
mkdir -p "$OUT"
"$@" --bank "$BANK" --split val --out "$OUT/val_preds.jsonl" --no-variants "${LIMIT_ARGS[@]}"
"$@" --bank "$BANK" --split test --out "$OUT/test_preds.jsonl" "${LIMIT_ARGS[@]}"
python -m spark_one.calibrate --bank "$BANK" --preds "$OUT/val_preds.jsonl" --out "$OUT/temps.json"
python -m spark_one.eval.harness --bank "$BANK" --preds "$OUT/test_preds.jsonl" --temps "$OUT/temps.json" --out "$OUT" --name "$NAME"
