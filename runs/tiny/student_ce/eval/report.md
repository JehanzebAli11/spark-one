# Eval: tiny student (1.0M, log loss only)

Temperatures (fitted on val): choice 4.290, noul 0.931, score 0.949, box 2.020

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 3727 | 0.263 | 0.028 | 0.014 | 0.728 | 1.320 | 0.285 |
| noul | 2619 | 0.638 | 0.025 | 0.029 | 0.402 | 0.577 | 0.776 |
| score | 926 | 0.436 | 0.061 | 0.070 | 0.583 | 1.049 | 0.637 |
| overall | 7272 | 0.420 | 0.030 | 0.025 | 0.592 | 1.018 | 0.593 |

## Box

n 1538, mean IoU 0.056, IoU>=0.5 0.001, IoU>=0.75 0.000, bin-step accuracy 0.408, bin-step ECE 0.181

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 3528 | 0.414 | 0.033 | 0.595 | 0.060 |
| ur-roman | 3744 | 0.426 | 0.019 | 0.590 | 0.052 |
| text | 2805 | 0.444 | 0.027 | 0.573 | - |
| image | 4467 | 0.405 | 0.028 | 0.605 | 0.056 |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.038, mean TV distance 0.010 (n 7188)
- Self-consistency (noul vs negated twin): logical agreement 0.330, mean |p - (1 - p_neg)| 0.080 (n 1438)
- None-fits choice items: mean top-1 0.348 (n 442; lower is better)
- Latency per item (batched CPU): mean 14.43 ms, p50 8.12, p95 53.88

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 252 | 0.266 | 0.027 |
| arithmetic.q_anyfits | 64 | 1.000 | 0.084 |
| bar_chart.box | 301 | 0.087 | 0.175 |
| bar_chart.compare | 299 | 0.495 | 0.041 |
| bar_chart.level | 301 | 0.216 | 0.071 |
| bar_chart.max | 201 | 0.234 | 0.012 |
| bar_chart.max_anyfits | 55 | 0.600 | 0.138 |
| cart.priciest | 221 | 0.204 | 0.060 |
| cart.priciest_anyfits | 55 | 0.673 | 0.129 |
| cart.total | 261 | 0.230 | 0.023 |
| cart.total_anyfits | 81 | 0.988 | 0.040 |
| chat.box | 294 | 0.017 | 0.165 |
| chat.count | 256 | 0.242 | 0.014 |
| chat.count_anyfits | 72 | 0.958 | 0.094 |
| chat.last | 276 | 0.341 | 0.035 |
| chat.last_anyfits | 34 | 0.794 | 0.224 |
| chat.mention | 295 | 0.749 | 0.068 |
| compare_decimals.q | 304 | 0.467 | 0.070 |
| compare_people.more | 281 | 0.367 | 0.028 |
| compare_people.noul | 225 | 0.493 | 0.093 |
| count_letter.q | 266 | 0.244 | 0.009 |
| count_letter.q_anyfits | 79 | 0.924 | 0.138 |
| count_list.q | 254 | 0.209 | 0.046 |
| count_list.q_anyfits | 76 | 0.500 | 0.293 |
| date_order.q | 263 | 0.289 | 0.050 |
| date_order.q_anyfits | 71 | 0.732 | 0.138 |
| day_after.q | 271 | 0.269 | 0.016 |
| day_after.q_anyfits | 81 | 0.889 | 0.062 |
| settings.box_button | 318 | 0.104 | 0.334 |
| settings.box_toggle | 318 | 0.042 | 0.313 |
| settings.count_on | 289 | 0.256 | 0.004 |
| settings.count_on_anyfits | 56 | 0.964 | 0.038 |
| settings.toggle | 318 | 0.535 | 0.044 |
| shapes.box | 307 | 0.027 | 0.064 |
| shapes.colour | 93 | 0.237 | 0.018 |
| shapes.colour_anyfits | 26 | 0.615 | 0.177 |
| shapes.count | 267 | 0.311 | 0.058 |
| shapes.count_anyfits | 77 | 0.948 | 0.080 |
| shapes.largest | 145 | 0.234 | 0.042 |
| shapes.largest_anyfits | 41 | 0.732 | 0.125 |
| shapes.leftof | 292 | 0.510 | 0.034 |
| shapes.quadrant | 131 | 0.160 | 0.092 |
| shapes.quadrant_anyfits | 18 | 0.667 | 0.272 |
| shapes.total | 307 | 0.270 | 0.071 |
| stock_score.q | 318 | 0.805 | 0.067 |
