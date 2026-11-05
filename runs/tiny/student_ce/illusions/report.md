# Eval: tiny student (1.0M, log loss only) on illusions

Temperatures (fitted on val): choice 4.290, noul 0.931, score 0.949, box 2.020

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 2400 | 0.324 | 0.017 | 0.011 | 0.667 | 1.099 | 0.323 |
| overall | 2400 | 0.324 | 0.017 | 0.011 | 0.667 | 1.099 | 0.323 |

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 1200 | 0.319 | 0.015 | 0.667 | - |
| ur-roman | 1200 | 0.328 | 0.007 | 0.667 | - |
| text | 0 | - | - | - | - |
| image | 2400 | 0.324 | 0.011 | 0.667 | - |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement -, mean TV distance - (n 0)
- Self-consistency (noul vs negated twin): logical agreement -, mean |p - (1 - p_neg)| - (n 0)
- Latency per item (batched CPU): mean 6.58 ms, p50 6.50, p95 8.56

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| delboeuf.larger | 600 | 0.297 | 0.039 |
| ebbinghaus.larger | 600 | 0.328 | 0.007 |
| muller_lyer.longer | 600 | 0.355 | 0.020 |
| ponzo.longer | 600 | 0.315 | 0.020 |
