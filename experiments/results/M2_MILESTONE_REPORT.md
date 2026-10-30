# M2 Milestone Report — Block Semantic Operators

## Scope

M2 covers BlockMap and BlockJoin execution, strict result contracts, recovery by block splitting, KV-budget admission/splitting, sequential operator composition, and initial isolated block-size measurements.
Semantic matching accuracy is not evaluated in this milestone.

## Unit tests

- Contract and memory-budget tests reported passing: 12/12.
- Includes JSON/result validation, incomplete-block recovery, pre-scheduling budget splitting, and rejecting a singleton that exceeds capacity.

## Sequential pipeline smoke test

- Request ID: `082b7cfacc8c4a65a64a99288f9cf2c3`
- HTTP API response recorded: yes
- Output rows: 1
- API latency: 40.583 seconds
- BlockMap → BlockJoin verification: PASS
- Response artifact: `experiments/results/m2_pipeline_smoke_final.json`

## Isolated sweep results

| Case | Status | Input rows | Planned initial batches | Output rows | API latency (s) |
|---|---|---:|---:|---:|---:|
| blockmap_b1 | OK | 4 | 4 | 4 | 125.524 |
| blockmap_b2 | OK | 4 | 2 | 4 | 81.02 |
| blockmap_b4 | OK | 4 | 1 | 4 | 75.199 |
| blockjoin_1x2 | OK | 2 | 2 | 2 | 124.908 |
| blockjoin_2x2 | OK | 2 | 1 | 2 | 14.915 |

## Initial observations

- BlockMap latency changed from 125.524s at B=1 to 75.199s at B=4, a 40.1% reduction in this single sweep.
- BlockJoin latency was 124.908s at 1×2 and 14.915s at 2×2.
- These are single-run initial measurements, not controlled performance claims. CPU warm-up and prefix caching may affect results.
- Raw sweep responses and `summary.csv` are retained in `experiments/results/m2_isolated_sweep_20261020_215913`.

## Operator counters from captured server log

```text
[0;36m(APIServer pid=62820)[0;0m [BlockMap] stats={'llm_calls': 1, 'validation_failures': 0, 'recovery_calls': 0, 'block_splits': 0, 'rows_processed': 1, 'blocks_processed': 1}
[0;36m(APIServer pid=62820)[0;0m [BlockJoin] candidate_pairs=1 matched_pairs=1 left_block_size=1 right_block_size=1 stats={'llm_calls': 1, 'validation_failures': 0, 'recovery_calls': 0, 'block_splits': 0, 'rows_processed': 1, 'blocks_processed': 1}
```

## M2 completion assessment

- All five sweep requests successful: yes
- Sequential BlockMap → BlockJoin smoke: pass
- Unit tests: 12 passing, as reported from the repository test run.
- Accuracy is intentionally outside this systems milestone.
- Later milestone work includes pipeline concurrency, block reformation, threshold/flush behavior, and deeper trace-driven evaluation.
