# HANDOFF — Spark-1

Living status file, updated after every milestone. Last update: 2026-11-05, cloud session
(CPU only, 4 cores, 15 GB RAM). Next owner: Claude Code on the DGX Spark (`~/lab/spark-one`).

## 1. Status

### Done and verified here (result files committed)

| what | evidence |
|---|---|
| Question-bank framework: schema, option shuffling, none-fits + companion Noul, scene-grouped 80/10/10 splits, provenance on every item | `spark_one/bank.py`, `tests/test_bank.py` |
| 15 procedural generators (8 text tasks EN + Roman Urdu, shape scenes, chat / settings / cart screenshots, bar, line and pie charts) + 4 illusion generators; exact labels and Box targets; reproducible across processes (tested with two `PYTHONHASHSEED`s) | `spark_one/generators/`; sample rows + images in `data/sample/` |
| Full bank v2 generated: **110,072 items** (88,260 train / 10,796 val / 11,016 test; 46,831 choice, 33,260 noul, 9,000 score, 20,981 box; 54,983 EN / 55,089 Roman Urdu; 80,017 image items; 5,179 none-fits) and **2,400-item illusion test set** (Müller-Lyer, Ebbinghaus, Ponzo, Delboeuf) | `data/sample/bank_manifest.json`, `data/sample/illusions_manifest.json` (images not committed; rebuild ~4 min). The tiny runs used bank v1 (13 generators, 91,975 items, `data/sample/bank_manifest_v1_tiny_runs.json`); v2 only appends line/pie charts and makes `date_order` reproducible (v1 `date_order` depended on PYTHONHASHSEED; its labels were correct). |
| Tests: 31 passing (`pytest -q`), ruff clean, CI workflow | `tests/`, `.github/workflows/ci.yml` |
| Real Gemma 4 E2B (`google/gemma-4-E2B-it`, bf16 on CPU) through our HF path: chat template, image processor, prefix-cache backend, all 54 answer labels are single tokens, first-token mass on labels ≈ 0.999 (no thinking token with `enable_thinking=False`) | `runs/smoke/smoke_gemma_e2b_cpu.json` |
| Real Gemma 4 E2B LoRA trainer smoke on CPU: LoRA on 12.1M params, 2 steps, simulated crash, resume from `checkpoint-2`, final adapter saved | `runs/smoke/lora_e2b_cpu.log` |
| Zero-shot Gemma 4 E2B baseline on CPU (bf16), 150 random test items (choice/noul/score) + order and negation variants, graded on 133 items (`date_order` excluded, see bank v2 note): accuracy 0.564, ECE 0.399 (badly overconfident), option-order agreement 0.541, negation agreement 0.160, mean top-1 0.986 on none-fits items. Temperatures fitted on 95 val items (choice 5.4, noul 12.0, score 7.3) cut ECE to 0.129 and log loss from 3.51 to 0.95. Small sample: treat as indicative. | `runs/gemma-e2b-base-cpu/eval/` (report + reliability figures), `temps.json` |
| Eval-only: `openai/gpt-4o-mini` via OpenRouter on 300 test items + variants, $0.69 of a $1 cap. Core accuracy 0.764, ECE 0.210 (uncalibrated), option-order agreement 0.690, negation agreement 0.784; Box unsupported (OpenAI ignores assistant prefill, 0 IoU) | `runs/eval-only/gpt-4o-mini/eval/report.md`, `test_preds.cost.json` |
| Tiny CPU loop end to end: tiny teacher (2.1M) → fake vLLM → teacher-score job (30,000 soft labels, all `ok`) → KD student and log-loss-only student (1.0M) → temperatures on val → harness on 9,252 test items + 8,626 variants and on the illusion set → server latency. All three models: core accuracy 0.41–0.42 (choice ≈ chance 0.263, noul 0.62–0.64, score 0.41–0.44), calibrated ECE 0.018–0.025. Under-trained at this compute, not a bug (overfit check passes). | `runs/tiny/RESULTS.md`, `runs/tiny/*/eval/`, `runs/tiny/overfit_check.txt`, `docs/demo_tiny_student.gif` |

### Done, tested only against fakes / tiny stand-ins

| what | how it was tested |
|---|---|
| Teacher-score job (`spark_one/teacher/score.py`): vLLM chat completions, `top_logprobs`, variant pooling (" A", "**A"), coverage, retries, resume, Box steps via `continue_final_message` | fake vLLM server with canned distributions and simulated 503s (`tests/test_teacher.py`); tiny teacher served through the same fake server in the tiny loop |
| Gemma LoRA trainer (`spark_one/train/lora.py`): KL + log loss at answer tokens, checkpoint every N steps, exact resume via `LAB_RESUME_FROM` | tiny random Llama + char tokenizer (`tests/test_lora.py`); resume reproduces uninterrupted weights exactly. On real Gemma 4 E2B only 3 CPU steps (see above). **Never run on a GPU or at scale.** |
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
# PyTorch for ARM64 + CUDA (DGX Spark / GB10). Checked 2026-11-05: the cu130 index has
# torch 2.14.1 and torchvision 0.29.1 aarch64 wheels for cp312; cu132 has torch 2.15.0.
# Match the CUDA version to `nvidia-smi`; otherwise use the NGC container (below).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e '.[hf,serve,dev]' transformers==5.22.1 peft==0.21.2
python3 scripts/check_spark.py          # torch.cuda True, transformers 5.22.1, teacher reachable
pytest -q                               # 31 passed
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
| 1 | build bank | `spark-one-01-build-bank.json` | `data/generated/bank/manifest.json` with `n_items: 110072`, `items_sha256` equal to `data/sample/bank_manifest.json` (same Pillow version) | 3–5 min |
| 2 | Gemma smoke on GPU | `spark-one-02-smoke-gemma.json` | `runs/smoke_gemma_e2b_gpu.json`: `label_mass_first_step` > 0.95 on every row, `tv_cached_vs_full_step0` < 0.01 | 5 min |
| 3 | teacher scores, train split | `spark-one-03-teacher-score-train.json` (needs vllm-teacher up) | `runs/teacher/train.jsonl`, ~70k rows, ≥ 95% `ok` | 4–10 h; try `--limit 20000` first |
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

Long-run probe (stopped): the same 1.0M student with 12k steps was stopped at step 3,000
(`configs/tiny_student_long.json`, `runs/tiny/student_long/train_log.jsonl`); a probe on 2,000
val items at step 3,000 still had choice accuracy 0.270, so CPU time went to the real E2B
baseline instead. Not worth resuming on the Spark: train the Gemma student.

## Stretch list (kept here; work down it after the core)

- [x] Optical-illusion eval set with exact ground truth (Müller-Lyer, Ebbinghaus; equal cases and truth-against-illusion cases)
- [x] Demo GIF of the generators (`docs/demo_generators.gif`)
- [x] Demo GIF of the tiny student answering (`docs/demo_tiny_student.gif`)
- [x] OpenRouter eval of one cheap model on a small test sample with a hard cap (eval only): gpt-4o-mini, $0.69
- [x] Ponzo and Delboeuf illusions (illusion set now 2,400 items, 4 types)
- [x] Line-chart and pie-chart generators (bank v2)
- [ ] More illusions (checker-shadow luminance, Zöllner)
- [ ] Generator param for 2× render resolution
- [ ] Hard-negative Box items (two similar targets) and multi-image states
- [ ] Per-task temperature (vs per-primitive) and a comparison in the harness
- [ ] Choice-vs-companion-Noul consistency metric
