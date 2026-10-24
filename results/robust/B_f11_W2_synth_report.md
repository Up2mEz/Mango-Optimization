# GreenMind emulator run 20261024_232237

- endpoint `http://localhost:8000/v1` model `mango`; profile `val`: 100 graded items = 154 graded HTTP requests (+ 0 probe requests); tool mode `live`; arrivals `poisson`; seed 0; sampling preset `neutral`
- server scheduling policy: **scheduling-policy=default, max-num-batched-tokens=2048, max-num-seqs=128, speculative-config={"model":, GM_THINK=1, GM_PRIORITY=0, GM_POWER=400, GM_THINK_FRAC=0.5, GM_MAX_BATCHED=2048, GM_ARGS=--long-prefill-token-threshold 512 --scheduler-cls slo_scheduler.SLOScheduler, GM_SCHED_DOOMED=1, GM_SPEC_K=3, GM_SPEC_NODROP=1, GM_MAX_SEQS=128, GM_SCHED_DOOMED_GUARD=0.045, GM_MODEL=$MODELS/w8a8-int8**
- energy window `graded (no probes)`: 113.2 s; max client schedule lag 0.002 s

## Energy

- 35443 J (nvml_total_energy; counter 35443 J, 10 Hz integral 35417 J), avg power 313.0 W, max sampled 422.2 W, power limit 400 W
- NOTE: NVML counts the whole GPU, including any other process sharing it.

## Score (graded items; an item passes only if all its HTTP turns pass)

| metric | value |
|---|---|
| SLO pass (items) | 96/100 (96.0%) |
| SLO pass (HTTP requests) | 150/154 (97.4%) |
| eta (passed items / J) | 0.00271 (369.2 J/item) |
| J / item | 354 |
| sT (peak pass fraction, n=67 items) | 0.985 |
| p95 TTFT | 0.97 s (p50 0.17); ratio to SLO p95 0.72 |
| p95 TPOT | 29.8 ms (p50 14.6) |
| sL (ttft 1.000, tpot 1.000) | 1.000 (strict, errors=inf: 1.000) |
| mean output tokens / request | 1017; finish mix {'length': 3, 'stop': 96, 'tool_calls': 55} |
| sE / S | n/a: pass --baseline-eta (or --save-baseline on a baseline run) |

## Comparison with the real request log of the same profile (graded requests, same SLO rule)

| metric | emulator | real log |
|---|---|---|
| graded HTTP requests | 154 | 169 |
| request-level SLO pass | 150 (97%) | 122 (72%) |
| p95 TTFT (s) | 0.97 | 2.20 |
| p95 TPOT (ms) | 29.8 | 48.8 |
| mean / median / p90 output tokens | 1017 / 232 / 1856 | 359 / 127 / 481 |
| truncated (finish=length) | 3 | 3 |
| mean tool-episode turns (tool requests per episode) | 3.70 | 4.45 |
| probe truncation (non-tool probes) | 0 / 0 | 90 / 132 |

## Per class (graded HTTP requests)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 27 | 90% | 0.28 | 0.45 | 18.4 | 24.8 | 123 | 2385 | truncated:3 |
| rag | 20 | 19 | 95% | 0.30 | 0.79 | 18.2 | 24.9 | 1425 | 590 | ttft:1 |
| tool | 74 | 74 | 100% | 0.29 | 0.57 | 12.6 | 22.0 | 1554 | 330 | - |
| reason | 15 | 15 | 100% | 0.29 | 0.62 | 12.4 | 21.4 | 236 | 2052 | - |
| thai | 10 | 10 | 100% | 0.27 | 0.46 | 31.2 | 39.5 | 139 | 1717 | - |
| long | 5 | 5 | 100% | 0.75 | 0.97 | 15.3 | 17.1 | 15277 | 182 | - |

## Per phase (graded, by item arrival order 20/67/13 %)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| ramp | 27 | 26 | 96% | 0.24 | 0.48 | 12.3 | 26.8 | 904 | 1629 | truncated:1 |
| peak | 88 | 87 | 99% | 0.32 | 0.70 | 16.0 | 31.9 | 1438 | 998 | truncated:1 |
| quiet | 39 | 37 | 95% | 0.27 | 0.82 | 11.3 | 20.0 | 1990 | 636 | truncated:1, ttft:1 |
