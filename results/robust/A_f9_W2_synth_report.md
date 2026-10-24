# GreenMind emulator run 20261024_231557

- endpoint `http://localhost:8000/v1` model `mango`; profile `val`: 100 graded items = 151 graded HTTP requests (+ 0 probe requests); tool mode `live`; arrivals `poisson`; seed 0; sampling preset `neutral`
- server scheduling policy: **scheduling-policy=default, max-num-batched-tokens=1024, max-num-seqs=128, speculative-config={"model":, GM_THINK=1, GM_PRIORITY=0, GM_POWER=400, GM_THINK_FRAC=0.5, GM_MAX_BATCHED=1024, GM_ARGS=--long-prefill-token-threshold 512 --scheduler-cls slo_scheduler.SLOScheduler, GM_SCHED_DOOMED=1, GM_SPEC_K=3, GM_MAX_SEQS=128, GM_SCHED_DOOMED_GUARD=0.045, GM_MODEL=$MODELS/w8a8-int8**
- energy window `graded (no probes)`: 115.6 s; max client schedule lag 0.002 s

## Energy

- 36493 J (nvml_total_energy; counter 36493 J, 10 Hz integral 36428 J), avg power 315.5 W, max sampled 436.8 W, power limit 400 W
- NOTE: NVML counts the whole GPU, including any other process sharing it.

## Score (graded items; an item passes only if all its HTTP turns pass)

| metric | value |
|---|---|
| SLO pass (items) | 94/100 (94.0%) |
| SLO pass (HTTP requests) | 145/151 (96.0%) |
| eta (passed items / J) | 0.00258 (388.2 J/item) |
| J / item | 365 |
| sT (peak pass fraction, n=67 items) | 0.970 |
| p95 TTFT | 2.02 s (p50 0.19); ratio to SLO p95 0.79 |
| p95 TPOT | 25.1 ms (p50 14.4) |
| sL (ttft 1.000, tpot 1.000) | 1.000 (strict, errors=inf: 1.000) |
| mean output tokens / request | 1018; finish mix {'length': 3, 'stop': 96, 'tool_calls': 52} |
| sE / S | n/a: pass --baseline-eta (or --save-baseline on a baseline run) |

## Comparison with the real request log of the same profile (graded requests, same SLO rule)

| metric | emulator | real log |
|---|---|---|
| graded HTTP requests | 151 | 169 |
| request-level SLO pass | 145 (96%) | 122 (72%) |
| p95 TTFT (s) | 2.02 | 2.20 |
| p95 TPOT (ms) | 25.1 | 48.8 |
| mean / median / p90 output tokens | 1018 / 247 / 2206 | 359 / 127 / 481 |
| truncated (finish=length) | 3 | 3 |
| mean tool-episode turns (tool requests per episode) | 3.55 | 4.45 |
| probe truncation (non-tool probes) | 0 / 0 | 90 / 132 |

## Per class (graded HTTP requests)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 29 | 97% | 0.27 | 0.38 | 17.1 | 22.5 | 123 | 1667 | truncated:1 |
| rag | 20 | 19 | 95% | 0.40 | 0.97 | 19.2 | 24.6 | 1425 | 561 | ttft:1 |
| tool | 71 | 70 | 99% | 0.32 | 0.69 | 12.8 | 20.8 | 1451 | 391 | ttft:1 |
| reason | 15 | 14 | 93% | 0.32 | 1.45 | 11.3 | 21.0 | 236 | 2149 | truncated:1, ttft:1 |
| thai | 10 | 9 | 90% | 0.30 | 0.41 | 27.6 | 36.1 | 139 | 3160 | truncated:1 |
| long | 5 | 4 | 80% | 0.75 | 1.36 | 18.6 | 22.3 | 15277 | 165 | ttft:1 |

## Per phase (graded, by item arrival order 20/67/13 %)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| ramp | 27 | 26 | 96% | 0.19 | 0.47 | 12.8 | 19.7 | 904 | 1378 | truncated:1 |
| peak | 93 | 91 | 98% | 0.33 | 0.74 | 16.0 | 29.0 | 1460 | 983 | truncated:1, ttft:1 |
| quiet | 31 | 28 | 90% | 0.34 | 1.43 | 12.3 | 22.5 | 1817 | 808 | ttft:3, truncated:1 |
