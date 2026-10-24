# GreenMind emulator run 20261024_232925

- endpoint `http://localhost:8000/v1` model `mango`; profile `val`: 100 graded items = 153 graded HTTP requests (+ 0 probe requests); tool mode `live`; arrivals `poisson`; seed 0; sampling preset `neutral`
- server scheduling policy: **scheduling-policy=default, max-num-batched-tokens=2560, max-num-seqs=128, speculative-config={"model":, GM_THINK=1, GM_PRIORITY=0, GM_POWER=400, GM_THINK_FRAC=0.5, GM_MAX_BATCHED=2560, GM_SPEC_DYN=[[1,32,3],[33,64,2],[65,128,1]], GM_ARGS=--long-prefill-token-threshold 600 --scheduler-cls slo_scheduler.SLOScheduler, GM_SCHED_DOOMED=1, GM_SPEC_K=3, GM_SPEC_NODROP=1, GM_MAX_SEQS=128, GM_SCHED_DOOMED_GUARD=0.045, GM_MODEL=$MODELS/w8a8-int8**
- energy window `graded (no probes)`: 122.8 s; max client schedule lag 0.002 s

## Energy

- 37475 J (nvml_total_energy; counter 37475 J, 10 Hz integral 37472 J), avg power 305.1 W, max sampled 427.8 W, power limit 400 W
- NOTE: NVML counts the whole GPU, including any other process sharing it.

## Score (graded items; an item passes only if all its HTTP turns pass)

| metric | value |
|---|---|
| SLO pass (items) | 95/100 (95.0%) |
| SLO pass (HTTP requests) | 148/153 (96.7%) |
| eta (passed items / J) | 0.00253 (394.5 J/item) |
| J / item | 375 |
| sT (peak pass fraction, n=67 items) | 0.970 |
| p95 TTFT | 0.64 s (p50 0.16); ratio to SLO p95 0.54 |
| p95 TPOT | 26.8 ms (p50 13.7) |
| sL (ttft 1.000, tpot 1.000) | 1.000 (strict, errors=inf: 1.000) |
| mean output tokens / request | 992; finish mix {'tool_calls': 54, 'stop': 94, 'length': 5} |
| sE / S | n/a: pass --baseline-eta (or --save-baseline on a baseline run) |

## Comparison with the real request log of the same profile (graded requests, same SLO rule)

| metric | emulator | real log |
|---|---|---|
| graded HTTP requests | 153 | 169 |
| request-level SLO pass | 148 (97%) | 122 (72%) |
| p95 TTFT (s) | 0.64 | 2.20 |
| p95 TPOT (ms) | 26.8 | 48.8 |
| mean / median / p90 output tokens | 992 / 292 / 2488 | 359 / 127 / 481 |
| truncated (finish=length) | 5 | 3 |
| mean tool-episode turns (tool requests per episode) | 3.65 | 4.45 |
| probe truncation (non-tool probes) | 0 / 0 | 90 / 132 |

## Per class (graded HTTP requests)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| chat | 30 | 28 | 93% | 0.21 | 0.41 | 17.9 | 22.4 | 123 | 1657 | truncated:2 |
| rag | 20 | 20 | 100% | 0.34 | 0.50 | 17.2 | 23.6 | 1425 | 624 | - |
| tool | 73 | 73 | 100% | 0.22 | 0.47 | 10.8 | 17.9 | 1752 | 551 | - |
| reason | 15 | 13 | 87% | 0.25 | 0.52 | 11.1 | 22.5 | 236 | 1936 | truncated:2 |
| thai | 10 | 9 | 90% | 0.19 | 0.51 | 29.5 | 32.1 | 139 | 1876 | truncated:1 |
| long | 5 | 5 | 100% | 0.52 | 0.57 | 17.5 | 25.1 | 15277 | 308 | - |

## Per phase (graded, by item arrival order 20/67/13 %)

| group | n | pass | rate | p50 TTFT/SLO | p95 TTFT/SLO | p50 TPOT ms | p95 TPOT ms | mean prompt | mean out | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| ramp | 26 | 24 | 92% | 0.15 | 0.39 | 9.6 | 25.0 | 818 | 1397 | truncated:2 |
| peak | 95 | 93 | 98% | 0.24 | 0.45 | 16.6 | 27.7 | 1628 | 876 | truncated:2 |
| quiet | 32 | 31 | 97% | 0.25 | 0.64 | 11.5 | 18.3 | 2046 | 1006 | truncated:1 |
