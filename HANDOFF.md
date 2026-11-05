# HANDOFF — Spark-1

Living status file, updated after every milestone. Last update: 2026-11-05, cloud session
(CPU only, 4 cores, 15 GB RAM). Next owner: Claude Code on the DGX Spark (`~/lab/spark-one`).

## 1. Status

### Done and verified here (result files committed)

| what | evidence |
|---|---|
| Question-bank framework: schema, option shuffling, none-fits + companion Noul, scene-grouped 80/10/10 splits, provenance on every item | `spark_one/bank.py`, `tests/test_bank.py` |
| 13 procedural generators (8 text tasks EN + Roman Urdu, shape scenes, chat / settings / cart screenshots, bar charts) + 2 illusion generators; exact labels and Box targets | `spark_one/generators/`; sample rows in `data/sample/` |
| Full bank generated: **91,975 items** (73,757 train / 8,966 val / 9,252 test; 41,754 choice, 26,240 noul, 9,000 score, 14,981 box; 45,955 EN / 46,020 Roman Urdu; 61,920 image items; 4,536 none-fits) and **2,400-item illusion test set** (Müller-Lyer, Ebbinghaus, Ponzo, Delboeuf) | `data/sample/bank_manifest.json`, `data/sample/illusions_manifest.json` (images are not committed; rebuild is deterministic, ~3 min) |
| Tests: 28 passing (`pytest -q`), ruff clean, CI workflow | `tests/`, `.github/workflows/ci.yml` |
| Real Gemma 4 E2B (`google/gemma-4-E2B-it`, bf16 on CPU) through our HF path: chat template, image processor, prefix-cache backend, all 54 answer labels are single tokens, first-token mass on labels ≈ 0.999 (no thinking token with `enable_thinking=False`) | `runs/smoke_gemma_e2b_cpu.json` |
| Real Gemma 4 E2B LoRA trainer smoke on CPU: LoRA on 12.1M params, 2 steps, simulated crash, resume from `checkpoint-2`, final adapter saved | `runs/smoke/lora_e2b_cpu.log` |
| Eval-only: `openai/gpt-4o-mini` via OpenRouter on 300 test items + variants, $0.69 of a $1 cap. Core accuracy 0.764, ECE 0.210 (uncalibrated), option-order agreement 0.690, negation agreement 0.784; Box unsupported (OpenAI ignores assistant prefill, 0 IoU) | `runs/eval-only/gpt-4o-mini/eval/report.md`, `test_preds.cost.json` |
| Tiny CPU loop end to end: tiny teacher (2.1M) → fake vLLM → teacher-score job (30,000 soft labels, all `ok`) → KD student and log-loss-only student (1.0M) → temperatures on val → harness on 9,252 test items + 8,626 variants and on the illusion set → server latency. All three models: core accuracy 0.41–0.42 (choice ≈ chance 0.263, noul 0.62–0.64, score 0.41–0.44), calibrated ECE 0.018–0.025. Under-trained at this compute, not a bug (overfit check passes). | `runs/tiny/RESULTS.md`, `runs/tiny/*/eval/`, `runs/tiny/overfit_check.txt`, `docs/demo_tiny_student.gif` |

### Done, tested only against fakes / tiny stand-ins

| what | how it was tested |
|---|---|
| Teacher-score job (`spark_one/teacher/score.py`): vLLM chat completions, `top_logprobs`, variant pooling (" A", "**A"), coverage, retries, resume, Box steps via `continue_final_message` | fake vLLM server with canned distributions and simulated 503s (`tests/test_teacher.py`); tiny teacher served through the same fake server in the tiny loop |
| Gemma LoRA trainer (`spark_one/train/lora.py`): KL + log loss at answer tokens, checkpoint every N steps, exact resume via `LAB_RESUME_FROM` | tiny random Llama + char tokenizer (`tests/test_lora.py`); resume reproduces uninterrupted weights exactly. **Never run on Gemma weights or a GPU.** |
| HF prediction (`spark_one/hf_predict.py`) and HF server backend | tiny Llama (cache == full forward, 1e-4) and 4–6 real-E2B items on CPU |
| Server `POST /v1/systemone` | tiny backend, all 4 primitives, prefix cache hit/miss, errors (`tests/test_server.py`) |
| Jev and OpenRouter adapters + cost cap | `httpx.MockTransport` (`tests/test_adapters.py`). No real Jev call (no key). |
| spark-lab job files | schema test mirrors spark-lab's validator (`tests/test_jobs.py`); not yet enqueued on a real worker |

### Not started
- Teacher scoring with Gemma 4 31B, Gemma E2B LoRA training, Gemma evals, real Jev / OpenRouter evals at scale.

## 2. Spark setup

```bash
cd ~/lab && git clone git@github.com:JehanzebAli11/spark-one.git && cd spark-one
python3 -m venv .venv && . .venv/bin/activate
# PyTorch for ARM64 + CUDA (DGX Spark / GB10). If the cu130 index has no aarch64 wheel for your
# driver, use the NGC container instead (below).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e '.[hf,serve,dev]' transformers==5.22.1 peft==0.21.2
python3 scripts/check_spark.py          # torch.cuda True, transformers 5.22.1, teacher reachable
pytest -q                               # 28 passed
```

Container alternative (ARM64 multi-arch): `docker run --gpus all -it --ipc=host -v ~/lab:/lab -v ~/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/pytorch:<tag>-py3` then `pip install -e '/lab/spark-one[hf,serve,dev]' transformers==5.22.1 peft==0.21.2`. Pick the tag from the DGX Spark playbooks.

Models (both were **not gated** on 2026-11-05 according to the HF API; a read token still helps with rate limits):

| role | id | size |
|---|---|---|
| student | `google/gemma-4-E2B-it` | 5.12 B params, ~10 GB bf16 (`Gemma4ForConditionalGeneration`) |
| teacher | `google/gemma-4-31B-it` | 31.3 B params, ~62 GB bf16 |

```bash
huggingface-cli download google/gemma-4-E2B-it
huggingface-cli download google/gemma-4-31B-it     # or let vllm-teacher pull it
```

Teacher server: spark-lab's `vllm-teacher.service`. In `~/.config/spark-lab.env` set
`TEACHER_MODEL=google/gemma-4-31B-it`. The default `max_logprobs` of 20 is enough for
Choice/Noul/Score (the jobs score only those). To also distil Box steps (32 bins) the unit
would need `--max-logprobs 32`: that is a systemd change, ask the owner first.

Env vars: `TEACHER_BASE_URL` (default `http://localhost:8000/v1`), `TEACHER_MODEL`,
`HF_HOME`/`HF_TOKEN`, `LAB_RESUME_FROM` (set by the worker), `SPARK1_BACKEND=hf`,
`SPARK1_BASE_MODEL`, `SPARK1_ADAPTER`, `SPARK1_TEMPS` (server), `TYPESAFE_API_KEY` and
`JEV_USD_PER_MTOK_IN` (Jev eval), `OPENROUTER_API_KEY` (OpenRouter eval).

## 3. Remaining work, in order

Enqueue from `~/lab/spark-lab`: `python3 -m lab enqueue ../spark-one/jobs/<file>.json`.
Time estimates are guesses for a GB10 until step 2 measures real throughput.

| # | step | job file | expected output | estimate |
|---|---|---|---|---|
| 0 | env check | `spark-one-00-check.json` | JSON with `cuda: true`, teacher model listed | 1 min |
| 1 | build bank | `spark-one-01-build-bank.json` | `data/generated/bank/manifest.json` with `n_items: 91975`, `items_sha256` equal to `data/sample/bank_manifest.json` (same Pillow version) | 3–5 min |
| 2 | Gemma smoke on GPU | `spark-one-02-smoke-gemma.json` | `runs/smoke_gemma_e2b_gpu.json`: `label_mass_first_step` > 0.95 on every row, `tv_cached_vs_full_step0` < 0.01 | 5 min |
| 3 | teacher scores, train split | `spark-one-03-teacher-score-train.json` (needs vllm-teacher up) | `runs/teacher/train.jsonl`, ~59k rows, ≥ 95% `ok` | 4–10 h; try `--limit 20000` first |
| 4 | teacher scores val/test + teacher eval | `spark-one-04-teacher-score-test.json` | `runs/teacher/eval/report.md` | 1–2 h |
| 5 | E2B zero-shot baseline | `spark-one-05-e2b-baseline-eval.json` | `runs/gemma-e2b-base/eval/report.md` (3,000 test items + variants) | 1–3 h |
| 6 | **stop vllm-teacher**, LoRA train | `spark-one-06-e2b-lora-train.json` | `runs/gemma-e2b-lora-v1/checkpoint-*`, `final_adapter/`, falling `ce` in `train_log.jsonl` | 3–6 h (3000 steps × 16 items) |
| 7 | Spark-1 eval | `spark-one-07-e2b-lora-eval.json` | `runs/gemma-e2b-lora-v1/eval/report.md` + figures; compare with step 5 | 1–3 h |
| 8 | illusions | `spark-one-08-illusions-eval.json` | `runs/gemma-e2b-lora-v1/illusions/report.md` | 20 min |
| 9 | serving latency | `spark-one-09-serve-bench.json` | `serve_bench.md`, cold vs warm prefix | 10 min |
| 10 | eval-only comparisons | `spark-one-10-eval-openrouter.json`, `spark-one-11-eval-jev.json` | reports under `runs/eval-only/`; costs in `*.cost.json` (caps $3 / $5) | 30 min each |

After step 7: commit `runs/gemma-e2b-lora-v1/eval/{report.md,report.json,*.png}` (not checkpoints),
update README results from the report files, and push the adapter to a **private** HF repo
(owner publishes).

## 4. Risks — verify first on real hardware

1. **Prefix-cache equivalence in bf16 on GPU.** On CPU bf16 the cached and full forward agree
   on the top label but differ in log-prob on near-zero labels. Check `tv_cached_vs_full_step0`
   in step 2; if > 0.01 run the server/predict with `HFBackend(..., use_prefix_cache=False)`.
   Gemma 4 uses sliding-window layers (window 512) and a hybrid cache; deep-copying the cache
   is what `HFBackend.answer_steps` does.
2. **Teacher first-token format.** E2B puts ≈ 0.999 of the first token on our labels; the 31B
   was not tested. The score job records `coverage` per step; if many rows are `ok: false`,
   inspect the top tokens (thinking/channel tokens, markdown) before training.
3. **LoRA trainer on Gemma 4 never ran.** Watch for: `logits_to_keep` with a tensor (we fall
   back to slicing full logits, which costs memory at 262k vocab), `mm_token_type_ids`
   handling for images, LoRA target regex (`.*language_model.*\.(q|k|v|o|gate|up|down)_proj`)
   matching the right modules (trainable params printed at start), memory with grad
   accumulation 16 and batch 1.
4. **ARM64 wheels.** torch/torchvision CUDA wheels for aarch64; fall back to the NGC container.
5. **Throughput.** The estimates above are guesses; measure in step 2/3 and set `--limit`s.
6. **Box vocabulary.** Measured on CPU: zero-shot E2B puts ~0 first-token mass on the bin
   letters after `top=` (it wants to write digits; top tokens "3", "2"), so base-model Box
   numbers in step 5 are meaningless and the LoRA must learn the bin labels. If Box IoU stays
   near 0 after training, check `bin_step_accuracy` in the report and consider more Box
   steps/weight. Teacher Box distillation would need `max_logprobs >= 32`; the default plan
   trains Box on the exact two-hot target only.
8. **Prefix cache speed-up** measured on CPU (E2B bf16, image item, 290 prefix tokens): full
   forward 50 s vs 9.7 s for a question with the prefix cached (`runs/smoke/`).
7. **Image resolution.** Screenshots are 360×600 and charts 400×300; the Gemma processor
   resizes them. If chat/cart text is unreadable to the model, render at 2× (generator param).

## 5. Prompt for Claude Code on the Spark

```text
You are continuing Spark-1 in ~/lab/spark-one (DGX Spark, ARM64 + CUDA, 128 GB; spark-lab queue in
~/lab/spark-lab). Read HANDOFF.md, README.md and ~/lab/spark-lab/CLAUDE.md first and follow their rules:
training data only from code labels or the Gemma 4 31B teacher at http://localhost:8000/v1; Claude,
OpenAI, Gemini or Jev outputs may only be evaluated; long work goes through spark-lab jobs; ask before
sudo or systemd changes; never invent numbers.
Work through HANDOFF.md section 3 in order, starting at step 0. Before each job, run the matching
command once in the foreground with a small --limit to verify it, fix what breaks (add a test for the
fix), then enqueue the job file. Verify the section 4 risks as you reach them and record findings in
HANDOFF.md. After each milestone: update HANDOFF.md (status, measured timings, result file paths),
commit result reports and figures (not checkpoints or generated images), and push to main.
When the Gemma eval lands, update README results strictly from report.json files.
```

## Tiny loop (CPU proof)

`scripts/run_tiny_loop.sh`: tiny teacher (2.1M params, log loss on code labels) → served by the
fake vLLM server → teacher-score job writes soft labels → tiny student (1.0M) trained with
KL + log loss, plus a log-loss-only ablation → temperatures on val → harness on test and on the
illusion set → server latency benchmark → `runs/tiny/RESULTS.md`.

Status: done. Results in `runs/tiny/RESULTS.md` (every number copied from the harness
`report.json` files). CPU timings: teacher 2,500 steps ≈ 84 min (contended), student ≈ 26 min,
fake-vLLM teacher scoring 55 items/s, predict ≈ 14 ms/item batched.
Bug found and fixed on the way: the fake server first spread the tiny teacher's mass over the
whole answer vocabulary, so all soft labels had coverage < 0.5 and were silently dropped; a
regression test now guards it. The same failure mode (low coverage → `ok: false` → no KL
signal) is what to watch with the real 31B teacher: check the `ok` rate before training.

## Stretch list (kept here; work down it after the core)

- [x] Optical-illusion eval set with exact ground truth (Müller-Lyer, Ebbinghaus; equal cases and truth-against-illusion cases)
- [x] Demo GIF of the generators (`docs/demo_generators.gif`)
- [x] Demo GIF of the tiny student answering (`docs/demo_tiny_student.gif`)
- [x] OpenRouter eval of one cheap model on a small test sample with a hard cap (eval only): gpt-4o-mini, $0.69
- [x] Ponzo and Delboeuf illusions (illusion set now 2,400 items, 4 types)
- [ ] More illusions (checker-shadow luminance, Zöllner) and line-chart / pie-chart generators (append-only: existing item ids stay stable)
- [ ] Generator param for 2× render resolution
- [ ] Hard-negative Box items (two similar targets) and multi-image states
- [ ] Per-task temperature (vs per-primitive) and a comparison in the harness
- [ ] Choice-vs-companion-Noul consistency metric
