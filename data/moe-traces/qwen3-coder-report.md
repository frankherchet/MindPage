check LRU: fast replay == WorkloadSimulator (67547, 30373, 29144)
check Frequency: fast replay == WorkloadSimulator (60113, 37807, 36578)
check PriorityCost: fast replay == WorkloadSimulator (60113, 37807, 36578)
model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
tasks: 37, decode tokens: 9435, experts: 48 x 128, top-8, 9.4 MB each
distinct experts touched in decode per task: median 78% (min 54%, max 85%)
no cache: 3624 MB per decode token

| policy | GPU share | hit rate cold | hit rate after prefill (task min-max) | MB/token cold | MB/token after prefill |
|---|---|---|---|---|---|
| LRU | 5% | 0.0% | 0.0% (0%-0%) | 3624 | 3624 |
| LRU | 10% | 47.8% | 48.0% (37%-61%) | 1890 | 1883 |
| LRU | 15% | 61.3% | 61.6% (53%-72%) | 1402 | 1393 |
| LRU | 20% | 70.2% | 70.6% (62%-83%) | 1081 | 1066 |
| LRU | 25% | 77.1% | 77.6% (70%-89%) | 831 | 810 |
| LRU | 30% | 82.1% | 83.1% (76%-93%) | 648 | 614 |
| LRU | 40% | 89.2% | 90.4% (84%-97%) | 393 | 347 |
| LRU | 50% | 92.6% | 94.4% (90%-98%) | 267 | 203 |
| Frequency | 5% | 29.1% | 12.6% (6%-19%) | 2570 | 3168 |
| Frequency | 10% | 45.6% | 20.2% (11%-29%) | 1970 | 2893 |
| Frequency | 15% | 57.4% | 27.5% (16%-37%) | 1545 | 2627 |
| Frequency | 20% | 66.2% | 34.7% (21%-45%) | 1226 | 2367 |
| Frequency | 25% | 73.0% | 41.8% (28%-51%) | 979 | 2110 |
| Frequency | 30% | 78.4% | 48.8% (35%-59%) | 784 | 1855 |
| Frequency | 40% | 86.2% | 62.6% (51%-74%) | 500 | 1355 |
| Frequency | 50% | 91.0% | 75.6% (65%-85%) | 325 | 886 |
| PriorityCost | 5% | 29.1% | 12.6% (6%-19%) | 2570 | 3168 |
| PriorityCost | 10% | 45.6% | 20.2% (11%-29%) | 1970 | 2893 |
| PriorityCost | 15% | 57.4% | 27.5% (16%-37%) | 1545 | 2627 |
| PriorityCost | 20% | 66.2% | 34.7% (21%-45%) | 1226 | 2367 |
| PriorityCost | 25% | 73.0% | 41.8% (28%-51%) | 979 | 2110 |
| PriorityCost | 30% | 78.4% | 48.8% (35%-59%) | 784 | 1855 |
| PriorityCost | 40% | 86.2% | 62.6% (51%-74%) | 500 | 1355 |
| PriorityCost | 50% | 91.0% | 75.6% (65%-85%) | 325 | 886 |

reuse distance in decode (accesses between two uses; 384 accesses per token): first use 4.8%, p10 382 (1.0 tok), p25 383 (1.0 tok), p50 767 (2.0 tok), p75 2686 (7.0 tok), p90 6529 (17.0 tok), p99 29185 (76.0 tok)
