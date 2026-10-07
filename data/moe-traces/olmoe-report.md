model: allenai/OLMoE-1B-7B-0125-Instruct
tasks: 37, decode tokens: 9435, experts: 16 x 64, top-8, 12.6 MB each
distinct experts touched in decode per task: median 93% (min 82%, max 96%)
no cache: 1611 MB per decode token

| policy | GPU share | hit rate cold | hit rate after prefill (task min-max) | MB/token cold | MB/token after prefill |
|---|---|---|---|---|---|
| LRU | 5% | 0.0% | 0.0% (0%-0%) | 1611 | 1611 |
| LRU | 10% | 0.0% | 0.0% (0%-0%) | 1611 | 1611 |
| LRU | 15% | 47.5% | 47.6% (44%-51%) | 846 | 844 |
| LRU | 20% | 56.2% | 56.4% (50%-61%) | 706 | 703 |
| LRU | 25% | 62.7% | 62.9% (56%-69%) | 600 | 597 |
| LRU | 30% | 68.1% | 68.3% (62%-75%) | 514 | 510 |
| LRU | 40% | 77.0% | 77.4% (71%-84%) | 370 | 364 |
| LRU | 50% | 84.2% | 84.7% (79%-92%) | 255 | 247 |
| Frequency | 5% | 28.0% | 27.8% (21%-32%) | 1160 | 1163 |
| Frequency | 10% | 41.5% | 41.3% (30%-49%) | 942 | 945 |
| Frequency | 15% | 50.8% | 50.0% (36%-60%) | 792 | 805 |
| Frequency | 20% | 57.9% | 56.9% (42%-68%) | 678 | 695 |
| Frequency | 25% | 63.8% | 62.5% (47%-74%) | 582 | 603 |
| Frequency | 30% | 69.0% | 67.5% (52%-79%) | 499 | 523 |
| Frequency | 40% | 77.8% | 75.6% (60%-87%) | 358 | 393 |
| Frequency | 50% | 84.6% | 82.1% (68%-92%) | 248 | 288 |
| PriorityCost | 5% | 28.0% | 27.8% (21%-32%) | 1160 | 1163 |
| PriorityCost | 10% | 41.5% | 41.3% (30%-49%) | 942 | 945 |
| PriorityCost | 15% | 50.8% | 50.0% (36%-60%) | 792 | 805 |
| PriorityCost | 20% | 57.9% | 56.9% (42%-68%) | 678 | 695 |
| PriorityCost | 25% | 63.8% | 62.5% (47%-74%) | 582 | 603 |
| PriorityCost | 30% | 69.0% | 67.5% (52%-79%) | 499 | 523 |
| PriorityCost | 40% | 77.8% | 75.6% (60%-87%) | 358 | 393 |
| PriorityCost | 50% | 84.6% | 82.1% (68%-92%) | 248 | 288 |

reuse distance in decode (accesses between two uses; 128 accesses per token): first use 2.9%, p10 126 (1.0 tok), p25 127 (1.0 tok), p50 254 (2.0 tok), p75 766 (6.0 tok), p90 1792 (14.0 tok), p99 7424 (58.0 tok)
