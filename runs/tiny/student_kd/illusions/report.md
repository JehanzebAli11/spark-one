# Eval: tiny student (1.0M, KL+log loss) on illusions

Temperatures (fitted on val): choice 21.400, noul 0.880, score 0.962, box 1.899

## Core primitives (choice, noul, score)

| primitive | n | accuracy | ECE raw | ECE calibrated | Brier cal | log loss cal | acc@50% coverage |
|---|---|---|---|---|---|---|---|
| choice | 2400 | 0.333 | 0.009 | 0.001 | 0.667 | 1.099 | 0.340 |
| overall | 2400 | 0.333 | 0.009 | 0.001 | 0.667 | 1.099 | 0.340 |

## Slices (calibrated)

| slice | n core | accuracy | ECE | Brier | box mean IoU |
|---|---|---|---|---|---|
| en | 1200 | 0.324 | 0.010 | 0.667 | - |
| ur-roman | 1200 | 0.342 | 0.008 | 0.667 | - |
| text | 0 | - | - | - | - |
| image | 2400 | 0.333 | 0.001 | 0.667 | - |

## Consistency

- Option-order consistency (choice, 2 rotations per item): argmax agreement -, mean TV distance - (n 0)
- Self-consistency (noul vs negated twin): logical agreement -, mean |p - (1 - p_neg)| - (n 0)
- Latency per item (batched CPU): mean 6.68 ms, p50 6.77, p95 8.62

## By task (calibrated)

| task | n | accuracy / mean IoU | ECE |
|---|---|---|---|
| delboeuf.larger | 600 | 0.308 | 0.025 |
| ebbinghaus.larger | 600 | 0.330 | 0.004 |
| muller_lyer.longer | 600 | 0.337 | 0.003 |
| ponzo.longer | 600 | 0.357 | 0.023 |
