# Eval: Gemma 4 E2B zero-shot (CPU, bf16)

Temperatures (fitted on val): noul 12.001, choice 5.392, score 7.336

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 69 | 0.507 | 0.444 | 0.209 | 0.611 | 1.143 | 0.706 |
| noul | 45 | 0.644 | 0.352 | 0.132 | 0.452 | 0.637 | 0.682 |
| score | 19 | 0.579 | 0.361 | 0.208 | 0.501 | 0.963 | 0.889 |
| overall | 133 | 0.564 | 0.399 | 0.129 | 0.541 | 0.946 | 0.697 |

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 58 | 0.638 | 0.121 | 0.477 | - |
| ur-roman | 75 | 0.507 | 0.222 | 0.591 | - |
| text | 44 | 0.659 | 0.178 | 0.482 | - |
| image | 89 | 0.517 | 0.138 | 0.571 | - |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.541, mean TV distance 0.304 (n 148)
- Self-consistency (noul vs negated twin): logical agreement 0.160, mean |p - (1 - p_neg)| 0.410 (n 25)
- None-fits choice items: mean top-1 0.699 (n 7; lower is better)
- Latency per item (batched CPU): mean 22510.68 ms, p50 28232.66, p95 44702.93

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 3 | 0.667 | 0.439 |
| arithmetic.q_anyfits | 1 | 0.000 | 0.735 |
| bar_chart.compare | 3 | 0.667 | 0.287 |
| bar_chart.level | 7 | 0.429 | 0.422 |
| bar_chart.max | 2 | 1.000 | 0.185 |
| bar_chart.max_anyfits | 1 | 1.000 | 0.439 |
| cart.priciest | 5 | 0.600 | 0.305 |
| cart.total | 3 | 0.000 | 0.864 |
| cart.total_anyfits | 4 | 0.750 | 0.413 |
| chat.count | 2 | 1.000 | 0.473 |
| chat.last | 4 | 0.000 | 0.581 |
| chat.last_anyfits | 2 | 1.000 | 0.326 |
| chat.mention | 4 | 1.000 | 0.140 |
| compare_decimals.q | 6 | 1.000 | 0.193 |
| compare_people.more | 7 | 0.857 | 0.132 |
| compare_people.noul | 6 | 0.333 | 0.503 |
| count_letter.q | 4 | 0.750 | 0.511 |
| count_letter.q_anyfits | 1 | 1.000 | 0.358 |
| count_list.q | 2 | 0.500 | 0.570 |
| count_list.q_anyfits | 2 | 1.000 | 0.457 |
| day_after.q | 4 | 0.000 | 0.468 |
| day_after.q_anyfits | 3 | 0.333 | 0.338 |
| settings.count_on | 11 | 0.273 | 0.290 |
| settings.count_on_anyfits | 1 | 0.000 | 0.613 |
| settings.toggle | 4 | 0.500 | 0.373 |
| shapes.colour | 5 | 0.800 | 0.239 |
| shapes.count | 7 | 0.429 | 0.326 |
| shapes.largest | 5 | 0.400 | 0.355 |
| shapes.largest_anyfits | 1 | 0.000 | 0.695 |
| shapes.leftof | 6 | 0.500 | 0.180 |
| shapes.quadrant | 5 | 0.800 | 0.103 |
| shapes.total | 7 | 0.429 | 0.272 |
| stock_score.q | 5 | 1.000 | 0.360 |
