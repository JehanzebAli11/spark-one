# Eval: Gemma 4 E2B zero-shot (CPU, bf16)

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 69 | 0.507 | 0.444 | 0.444 | 0.885 | 3.736 | 0.706 |
| noul | 45 | 0.644 | 0.352 | 0.352 | 0.699 | 3.760 | 0.682 |
| score | 19 | 0.579 | 0.361 | 0.361 | 0.684 | 2.102 | 0.889 |
| overall | 133 | 0.564 | 0.399 | 0.399 | 0.793 | 3.510 | 0.742 |

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 58 | 0.638 | 0.345 | 0.684 | - |
| ur-roman | 75 | 0.507 | 0.440 | 0.878 | - |
| text | 44 | 0.659 | 0.343 | 0.668 | - |
| image | 89 | 0.517 | 0.443 | 0.855 | - |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.541, mean TV distance 0.450 (n 148)
- Self-consistency (noul vs negated twin): logical agreement 0.160, mean |p - (1 - p_neg)| 0.816 (n 25)
- None-fits choice items: mean top-1 0.986 (n 7; lower is better)
- Latency per item (batched CPU): mean 22510.68 ms, p50 28232.66, p95 44702.93

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 3 | 0.667 | 0.318 |
| arithmetic.q_anyfits | 1 | 0.000 | 1.000 |
| bar_chart.compare | 3 | 0.667 | 0.296 |
| bar_chart.level | 7 | 0.429 | 0.551 |
| bar_chart.max | 2 | 1.000 | 0.000 |
| bar_chart.max_anyfits | 1 | 1.000 | 0.050 |
| cart.priciest | 5 | 0.600 | 0.346 |
| cart.total | 3 | 0.000 | 1.000 |
| cart.total_anyfits | 4 | 0.750 | 0.238 |
| chat.count | 2 | 1.000 | 0.021 |
| chat.last | 4 | 0.000 | 0.856 |
| chat.last_anyfits | 2 | 1.000 | 0.000 |
| chat.mention | 4 | 1.000 | 0.000 |
| compare_decimals.q | 6 | 1.000 | 0.000 |
| compare_people.more | 7 | 0.857 | 0.143 |
| compare_people.noul | 6 | 0.333 | 0.667 |
| count_letter.q | 4 | 0.750 | 0.379 |
| count_letter.q_anyfits | 1 | 1.000 | 0.001 |
| count_list.q | 2 | 0.500 | 0.497 |
| count_list.q_anyfits | 2 | 1.000 | 0.117 |
| day_after.q | 4 | 0.000 | 0.853 |
| day_after.q_anyfits | 3 | 0.333 | 0.666 |
| settings.count_on | 11 | 0.273 | 0.723 |
| settings.count_on_anyfits | 1 | 0.000 | 0.996 |
| settings.toggle | 4 | 0.500 | 0.500 |
| shapes.colour | 5 | 0.800 | 0.200 |
| shapes.count | 7 | 0.429 | 0.561 |
| shapes.largest | 5 | 0.400 | 0.448 |
| shapes.largest_anyfits | 1 | 0.000 | 1.000 |
| shapes.leftof | 6 | 0.500 | 0.500 |
| shapes.quadrant | 5 | 0.800 | 0.200 |
| shapes.total | 7 | 0.429 | 0.457 |
| stock_score.q | 5 | 1.000 | 0.000 |
