# Eval: tiny student (1.0M, KL+log loss)

Temperatures (fitted on val): choice 21.400, noul 0.880, score 0.962, box 1.899

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 3727 | 0.263 | 0.029 | 0.011 | 0.728 | 1.320 | 0.283 |
| noul | 2619 | 0.638 | 0.033 | 0.037 | 0.402 | 0.574 | 0.771 |
| score | 926 | 0.407 | 0.050 | 0.053 | 0.598 | 1.061 | 0.581 |
| overall | 7272 | 0.416 | 0.032 | 0.025 | 0.594 | 1.018 | 0.588 |

## Box

n 1538, mean IoU 0.092, IoU>=0.5 0.003, IoU>=0.75 0.000, bin-step accuracy 0.472, bin-step ECE 0.172

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 3528 | 0.418 | 0.028 | 0.595 | 0.099 |
| ur-roman | 3744 | 0.415 | 0.024 | 0.593 | 0.086 |
| text | 2805 | 0.438 | 0.020 | 0.580 | - |
| image | 4467 | 0.403 | 0.031 | 0.603 | 0.092 |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.037, mean TV distance 0.002 (n 7188)
- Self-consistency (noul vs negated twin): logical agreement 0.320, mean |p - (1 - p_neg)| 0.087 (n 1438)
- None-fits choice items: mean top-1 0.345 (n 442; lower is better)
- Latency per item (batched CPU): mean 8.56 ms, p50 5.11, p95 30.90

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 252 | 0.250 | 0.001 |
| arithmetic.q_anyfits | 64 | 0.906 | 0.147 |
| bar_chart.box | 301 | 0.110 | 0.164 |
| bar_chart.compare | 299 | 0.492 | 0.037 |
| bar_chart.level | 301 | 0.223 | 0.057 |
| bar_chart.max | 201 | 0.189 | 0.049 |
| bar_chart.max_anyfits | 55 | 0.709 | 0.104 |
| cart.priciest | 221 | 0.267 | 0.024 |
| cart.priciest_anyfits | 55 | 0.618 | 0.107 |
| cart.total | 261 | 0.245 | 0.006 |
| cart.total_anyfits | 81 | 0.975 | 0.058 |
| chat.box | 294 | 0.035 | 0.193 |
| chat.count | 256 | 0.211 | 0.041 |
| chat.count_anyfits | 72 | 0.931 | 0.075 |
| chat.last | 276 | 0.370 | 0.019 |
| chat.last_anyfits | 34 | 0.824 | 0.155 |
| chat.mention | 295 | 0.749 | 0.058 |
| compare_decimals.q | 304 | 0.520 | 0.028 |
| compare_people.more | 281 | 0.359 | 0.025 |
| compare_people.noul | 225 | 0.467 | 0.115 |
| count_letter.q | 266 | 0.252 | 0.001 |
| count_letter.q_anyfits | 79 | 0.924 | 0.039 |
| count_list.q | 254 | 0.272 | 0.021 |
| count_list.q_anyfits | 76 | 0.539 | 0.205 |
| date_order.q | 263 | 0.308 | 0.026 |
| date_order.q_anyfits | 71 | 0.662 | 0.110 |
| day_after.q | 271 | 0.240 | 0.011 |
| day_after.q_anyfits | 81 | 0.914 | 0.136 |
| settings.box_button | 318 | 0.221 | 0.447 |
| settings.box_toggle | 318 | 0.066 | 0.292 |
| settings.count_on | 289 | 0.235 | 0.016 |
| settings.count_on_anyfits | 56 | 1.000 | 0.024 |
| settings.toggle | 318 | 0.494 | 0.047 |
| shapes.box | 307 | 0.024 | 0.056 |
| shapes.colour | 93 | 0.183 | 0.068 |
| shapes.colour_anyfits | 26 | 0.885 | 0.153 |
| shapes.count | 267 | 0.225 | 0.026 |
| shapes.count_anyfits | 77 | 0.948 | 0.019 |
| shapes.largest | 145 | 0.207 | 0.068 |
| shapes.largest_anyfits | 41 | 0.683 | 0.151 |
| shapes.leftof | 292 | 0.503 | 0.077 |
| shapes.quadrant | 131 | 0.313 | 0.062 |
| shapes.quadrant_anyfits | 18 | 0.889 | 0.139 |
| shapes.total | 307 | 0.270 | 0.063 |
| stock_score.q | 318 | 0.714 | 0.041 |
