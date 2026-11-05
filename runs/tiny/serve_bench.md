# Server latency: spark-1-tiny

40 requests, each 3-4 questions about one image + state.

| mode | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| cold (no prefix cache) | 106.3 | 110.0 | 147.0 |
| warm (prefix cached) | 84.4 | 85.9 | 122.5 |
