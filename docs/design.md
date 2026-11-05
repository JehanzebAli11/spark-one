# Spark-1 design notes

## Answer steps

Every primitive becomes one or more categorical *steps*, and a model is only ever asked for
the distribution of the next token over a fixed label set (`spark_one/steps.py`):

| primitive | steps | labels |
|---|---|---|
| choice | 1 | `A`..`J` (option letters in presented order) |
| noul | 1 | `Yes`, `No` |
| score | 1 | `0`..`n-1` |
| box | 4: top, left, bottom, right | 32 bins `A`..`Z`,`a`..`f` per axis |

All 54 labels are single tokens in the Gemma 4 tokenizer (checked on the real tokenizer).
Training and teacher scoring read logits only at the label positions and renormalise over
the step's labels, so the model never produces free text.

Box steps are teacher-forced during training (the true earlier bins are fed in) and decoded
greedily at inference. Each coordinate's training target is an exact *two-hot* distribution:
mass split between the two nearest bin centres so that the expected coordinate equals the true
value. Measured on the 1,538 test Box items: an oracle that puts all mass on the correct bin
reaches mean IoU 0.763 (bin quantisation hurts the small UI boxes), while the two-hot oracle
with expected-value decoding reaches 1.000.

## "None fits"

Choice answers are always over the given options, so a Choice cannot say "none of these".
Following the System One docs, the bank pairs Choice items with a companion Noul ("does any
option fit?"). 12% of eligible Choice items have their correct option removed (`none_fits`,
answer `None`, Choice target = uniform distribution) and always get a companion Noul labelled
false; another 12% of normal items get a companion labelled true.

## Splits and provenance

Splits are assigned by a salted hash of the *scene* (group), so all questions about one image
or one text state land in the same split. Every item records generator, version, seed, scene
seed, index, group, the scene parameters used to draw it, `label_source: "code"`, and any
transforms (`none_fits`, `companion_noul`).

## Loss

`loss = (1 - alpha) * CE(ground truth) + alpha * KL(teacher || student)` per step, on the
label-subset logits (`spark_one/train/losses.py`). Steps without teacher probabilities use the
ground-truth term alone. The ground-truth term uses the soft target where one exists
(none-fits uniform, Box two-hot).

## Teacher scores

`spark_one/teacher/score.py` calls the vLLM chat endpoint with `max_tokens=1`,
`logprobs=true`, `top_logprobs=20`, pools token variants (`" A"`, `"A"`, `"**A"`), fills labels
missing from the top-k with a bounded estimate, and records `coverage` (raw probability mass on
labels). Rows with coverage < 0.5 are kept with `ok: false` and ignored by training. Box steps
use `continue_final_message` to prefill earlier bins.

## Calibration

One temperature per primitive, fitted on validation predictions by minimising log loss
(grid + golden-section search), applied to stored log-probabilities. The harness reports raw
and calibrated numbers side by side.

## Confidence

Choice: `(n * p_top - 1) / (n - 1)`. Score: `max(0, 1 - E|level - mode| / MAD_uniform)`.
Noul: none (the probability is the answer). Box: Spark-1's own convention, the mean Choice
confidence of the four bin steps. Confidence describes the spread of the answer, not the
probability of being right.

## Prefix caching

Prompts are split into a cacheable prefix (system turn, image, state) and a per-question
suffix (question, options, answer steps). The tiny model and the HF backend compute the
prefix KV cache once per (state, image) and reuse it for every question; the HF path tokenises
prefix and suffix separately in training too, so train and serve see identical token
sequences.

## Consistency metrics

Option-order consistency: each Choice item is re-asked with two rotations of its options;
the harness maps probabilities back and reports argmax agreement and total-variation
distance. Self-consistency: Nouls with a negated twin ("more" vs "fewer", "left" vs "right",
"on" vs "off") should satisfy `p(q) ≈ 1 - p(not q)`.
