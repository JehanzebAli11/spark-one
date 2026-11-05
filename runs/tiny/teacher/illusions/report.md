# Eval: tiny teacher (2.1M, code labels) on illusions

Temperatures (fitted on val): choice 21.400, noul 1.010, score 0.993, box 2.537

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 2400 | 0.335 | 0.017 | 0.000 | 0.667 | 1.099 | 0.314 |
| overall | 2400 | 0.335 | 0.017 | 0.000 | 0.667 | 1.099 | 0.314 |

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 1200 | 0.339 | 0.005 | 0.667 | - |
| ur-roman | 1200 | 0.330 | 0.005 | 0.667 | - |
| text | 0 | - | - | - | - |
| image | 2400 | 0.335 | 0.000 | 0.667 | - |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement -, mean TV distance - (n 0)
- Self-consistency (noul vs negated twin): logical agreement -, mean |p - (1 - p_neg)| - (n 0)
- Latency per item (batched CPU): mean 9.69 ms, p50 9.73, p95 13.28

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| delboeuf.larger | 600 | 0.308 | 0.026 |
| ebbinghaus.larger | 600 | 0.358 | 0.024 |
| muller_lyer.longer | 600 | 0.357 | 0.023 |
| ponzo.longer | 600 | 0.315 | 0.019 |
