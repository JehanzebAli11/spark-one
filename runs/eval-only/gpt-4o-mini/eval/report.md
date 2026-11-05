# Eval: openai/gpt-4o-mini via OpenRouter (eval only, uncalibrated)

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 118 | 0.695 | 0.269 | 0.269 | 0.553 | 2.338 | 0.932 |
| noul | 86 | 0.849 | 0.136 | 0.136 | 0.282 | 1.271 | 1.000 |
| score | 33 | 0.788 | 0.206 | 0.206 | 0.378 | 1.425 | 0.938 |
| overall | 237 | 0.764 | 0.210 | 0.210 | 0.430 | 1.823 | 0.966 |

## Box

n 53, mean IoU 0.000, IoU>=0.5 0.000, IoU>=0.75 0.000, bin-step accuracy 0.000, bin-step ECE 0.031

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 104 | 0.712 | 0.272 | 0.536 | 0.000 |
| ur-roman | 133 | 0.805 | 0.164 | 0.347 | 0.000 |
| text | 101 | 0.842 | 0.155 | 0.292 | - |
| image | 136 | 0.706 | 0.261 | 0.532 | 0.000 |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement 0.690, mean TV distance 0.291 (n 226)
- Self-consistency (noul vs negated twin): logical agreement 0.784, mean |p - (1 - p_neg)| 0.204 (n 51)
- None-fits choice items: mean top-1 0.867 (n 10; lower is better)
- Latency per item (batched CPU): mean 1162.08 ms, p50 711.10, p95 3474.27

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| arithmetic.q | 7 | 0.286 | 0.677 |
| bar_chart.box | 7 | 0.000 | 0.031 |
| bar_chart.compare | 8 | 1.000 | 0.001 |
| bar_chart.level | 10 | 0.700 | 0.341 |
| bar_chart.max | 5 | 1.000 | 0.000 |
| bar_chart.max_anyfits | 2 | 1.000 | 0.000 |
| cart.priciest | 3 | 1.000 | 0.000 |
| cart.priciest_anyfits | 3 | 0.667 | 0.333 |
| cart.total | 9 | 0.222 | 0.781 |
| cart.total_anyfits | 3 | 1.000 | 0.119 |
| chat.box | 14 | 0.000 | 0.031 |
| chat.count | 8 | 0.375 | 0.486 |
| chat.count_anyfits | 3 | 1.000 | 0.000 |
| chat.last | 10 | 0.600 | 0.300 |
| chat.last_anyfits | 1 | 1.000 | 0.000 |
| chat.mention | 9 | 1.000 | 0.000 |
| compare_decimals.q | 10 | 1.000 | 0.000 |
| compare_people.more | 7 | 1.000 | 0.000 |
| compare_people.noul | 11 | 1.000 | 0.000 |
| count_letter.q | 9 | 0.778 | 0.217 |
| count_letter.q_anyfits | 5 | 0.400 | 0.581 |
| count_list.q | 17 | 0.765 | 0.203 |
| date_order.q | 10 | 1.000 | 0.002 |
| date_order.q_anyfits | 1 | 1.000 | 0.000 |
| day_after.q | 10 | 0.800 | 0.267 |
| day_after.q_anyfits | 1 | 1.000 | 0.321 |
| settings.box_button | 11 | 0.000 | 0.031 |
| settings.box_toggle | 12 | 0.000 | 0.031 |
| settings.count_on | 8 | 0.750 | 0.297 |
| settings.count_on_anyfits | 1 | 0.000 | 1.000 |
| settings.toggle | 10 | 1.000 | 0.000 |
| shapes.box | 9 | 0.000 | 0.031 |
| shapes.colour_anyfits | 1 | 1.000 | 0.003 |
| shapes.count | 8 | 0.625 | 0.369 |
| shapes.count_anyfits | 4 | 0.750 | 0.274 |
| shapes.largest | 3 | 0.333 | 0.580 |
| shapes.largest_anyfits | 1 | 0.000 | 0.881 |
| shapes.leftof | 12 | 0.500 | 0.437 |
| shapes.quadrant | 4 | 1.000 | 0.016 |
| shapes.total | 10 | 0.600 | 0.338 |
| stock_score.q | 13 | 1.000 | 0.000 |
