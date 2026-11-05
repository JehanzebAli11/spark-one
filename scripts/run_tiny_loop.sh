#!/usr/bin/env bash
# The whole Spark-1 loop on CPU with tiny models (stand-ins for Gemma 4 31B / E2B):
# teacher (code labels) -> fake vLLM server -> teacher-score job (soft labels) ->
# student with KL + log loss, and a log-loss-only ablation -> temperature scaling -> eval.
set -euo pipefail
cd "$(dirname "$0")/.."
BANK=data/generated/bank
R=runs/tiny
SCORE_LIMIT="${SCORE_LIMIT:-30000}"
[[ -f $R/teacher/final.pt ]] || python -m spark_one.tiny.train --config configs/tiny_teacher.json
if [[ ! -f $R/teacher_scores/.done ]]; then
  python -m spark_one.teacher.fake_vllm --ckpt $R/teacher/final.pt --port 8011 --threads 3 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  until curl -sf http://127.0.0.1:8011/v1/models >/dev/null; do sleep 1; done
  python -m spark_one.teacher.score --bank $BANK --split train --primitives choice noul score \
    --limit "$SCORE_LIMIT" --out $R/teacher_scores/train.jsonl --base-url http://127.0.0.1:8011/v1 \
    --model tiny-teacher --concurrency 8
  kill $SRV; trap - EXIT
  touch $R/teacher_scores/.done
fi
[[ -f $R/student_kd/final.pt ]] || python -m spark_one.tiny.train --config configs/tiny_student.json
[[ -f $R/student_ce/final.pt ]] || python -m spark_one.tiny.train --config configs/tiny_student_ce.json
for m in teacher student_kd student_ce; do
  if [[ -f $R/$m/eval/report.json && -f $R/$m/illusions/report.json ]]; then continue; fi
  name=$(case $m in teacher) echo "tiny teacher (2.1M, code labels)";; student_kd) echo "tiny student (1.0M, KL+log loss)";; student_ce) echo "tiny student (1.0M, log loss only)";; esac)
  if [[ ! -f $R/$m/eval/report.json ]]; then
  python -m spark_one.tiny.predict --ckpt $R/$m/final.pt --bank $BANK --split val --out $R/$m/val_preds.jsonl --no-variants
  python -m spark_one.tiny.predict --ckpt $R/$m/final.pt --bank $BANK --split test --out $R/$m/test_preds.jsonl
  python -m spark_one.calibrate --bank $BANK --preds $R/$m/val_preds.jsonl --out $R/$m/temps.json
  python -m spark_one.eval.harness --bank $BANK --preds $R/$m/test_preds.jsonl --temps $R/$m/temps.json --out $R/$m/eval --name "$name" > /dev/null
  fi
  python -m spark_one.tiny.predict --ckpt $R/$m/final.pt --bank data/generated/illusions --split test --out $R/$m/illusion_preds.jsonl --no-variants
  python -m spark_one.eval.harness --bank data/generated/illusions --preds $R/$m/illusion_preds.jsonl --temps $R/$m/temps.json --out $R/$m/illusions --name "$name on illusions" > /dev/null
done
python scripts/bench_server.py --backend tiny --ckpt $R/student_kd/final.pt --out $R/serve_bench.md
python scripts/summarize_tiny.py
