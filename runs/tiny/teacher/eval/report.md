# Eval: tiny teacher (2.1M, code labels)

Temperatures (fitted on val): choice 21.400, noul 1.010, score 0.993, box 2.537

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 3727 | 0.263 | 0.034 | 0.016 | 0.728 | 1.320 | 0.302 |
| noul | 2619 | 0.623 | 0.025 | 0.024 | 0.424 | 0.604 | 0.733 |
| score | 926 | 0.428 | 0.030 | 0.030 | 0.596 | 1.059 | 0.572 |
| overall | 7272 | 0.414 | 0.029 | 0.018 | 0.602 | 1.029 | 0.584 |

## Box

n 1538, mean IoU 0.026, IoU>=0.5 0.000, IoU>=0.75 0.000, bin-step accuracy 0.404, bin-step ECE 0.141

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 3528 | 0.415 | 0.023 | 0.603 | 0.024 |
| ur-roman | 3744 | 0.413 | 0.022 | 0.600 | 0.029 |
| text | 2805 | 0.429 | 0.015 | 0.584 | - |
| image | 4467 | 0.404 | 0.025 | 0.613 | 0.026 |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.053, mean TV distance 0.002 (n 7188)
- Self-consistency (noul vs negated twin): logical agreement 0.261, mean |p - (1 - p_neg)| 0.071 (n 1438)
- None-fits choice items: mean top-1 0.345 (n 442; lower is better)
- Latency per item (batched CPU): mean 13.99 ms, p50 8.56, p95 48.09

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 252 | 0.274 | 0.023 |
| arithmetic.q_anyfits | 64 | 0.922 | 0.165 |
| bar_chart.box | 301 | 0.041 | 0.117 |
| bar_chart.compare | 299 | 0.488 | 0.046 |
| bar_chart.level | 301 | 0.282 | 0.014 |
| bar_chart.max | 201 | 0.259 | 0.020 |
| bar_chart.max_anyfits | 55 | 0.564 | 0.197 |
| cart.priciest | 221 | 0.276 | 0.039 |
| cart.priciest_anyfits | 55 | 0.473 | 0.225 |
| cart.total | 261 | 0.211 | 0.040 |
| cart.total_anyfits | 81 | 0.988 | 0.094 |
| chat.box | 294 | 0.027 | 0.116 |
| chat.count | 256 | 0.219 | 0.033 |
| chat.count_anyfits | 72 | 0.875 | 0.088 |
| chat.last | 276 | 0.380 | 0.030 |
| chat.last_anyfits | 34 | 0.588 | 0.209 |
| chat.mention | 295 | 0.749 | 0.035 |
| compare_decimals.q | 304 | 0.530 | 0.005 |
| compare_people.more | 281 | 0.335 | 0.001 |
| compare_people.noul | 225 | 0.529 | 0.055 |
| count_letter.q | 266 | 0.207 | 0.044 |
| count_letter.q_anyfits | 79 | 0.949 | 0.157 |
| count_list.q | 254 | 0.260 | 0.009 |
| count_list.q_anyfits | 76 | 0.513 | 0.226 |
| date_order.q | 263 | 0.319 | 0.015 |
| date_order.q_anyfits | 71 | 0.634 | 0.161 |
| day_after.q | 271 | 0.199 | 0.052 |
| day_after.q_anyfits | 81 | 0.679 | 0.071 |
| settings.box_button | 318 | 0.048 | 0.304 |
| settings.box_toggle | 318 | 0.007 | 0.253 |
| settings.count_on | 289 | 0.273 | 0.023 |
| settings.count_on_anyfits | 56 | 1.000 | 0.061 |
| settings.toggle | 318 | 0.506 | 0.076 |
| shapes.box | 307 | 0.009 | 0.037 |
| shapes.colour | 93 | 0.290 | 0.040 |
| shapes.colour_anyfits | 26 | 0.923 | 0.214 |
| shapes.count | 267 | 0.187 | 0.063 |
| shapes.count_anyfits | 77 | 0.857 | 0.083 |
| shapes.largest | 145 | 0.297 | 0.022 |
| shapes.largest_anyfits | 41 | 0.659 | 0.162 |
| shapes.leftof | 292 | 0.497 | 0.027 |
| shapes.quadrant | 131 | 0.237 | 0.014 |
| shapes.quadrant_anyfits | 18 | 0.722 | 0.196 |
| shapes.total | 307 | 0.270 | 0.048 |
| stock_score.q | 318 | 0.717 | 0.028 |
