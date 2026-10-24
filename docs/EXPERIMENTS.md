# Experiments: what we tried, what worked, what failed

Format: hypothesis, change, result, conclusion. Failures are kept on purpose. Numbers are real grader scores (Final = 300
requests) unless marked "replay". Tables with all numbers: [../results/RESULTS.md](../results/RESULTS.md). Request logs:
`results/reqlog_*.jsonl`. Environment variable names refer to `scripts/serve.sh`.

Score recap: `S = 100 * A * sE^0.6 * sT^0.2 * sL^0.2`. Baseline (untuned BF16 `vllm serve`): about 613 J per on-time request (0.00163 on-time requests per joule).

## Timeline

| # | S | change | verdict |
|---|---|---|---|
| Final 1 | 41.8 | AWQ + MTP k=3, FCFS, batched 8192 | starting point; only 51 % of requests on time |
| Final 2 | 28.6 | thinking budget + batched 2048, FCFS | worse |
| Final 3 | 40.4 | newest-first priority, batched 8192 | no gain |
| Final 4 | 52.7 | batched 1024 + long-prefill threshold 512 | first real gain |
| Final 5 | 80.0 | custom SLO scheduler | large gain |
| Final 6 | 74.0 | scheduler without doomed-decode skipping | worse (TTFT tail) |
| Final 7 | 85.6 | doomed skipping guarded at 45 ms TPOT | gain |
| Final 8 | 76.8 | + 2 API server processes | worse |
| Final 9 | 98.6 | INT8 W8A8 weights | large gain |
| Final 10 | 99.5 | batched 2048 | small gain |
| Final 11 | 99.5 | + MTP no-block-drop | no measurable change; submitted |

## E1. Diagnose before tuning: metadata request log (Validation V1-V3)

- Hypothesis: we do not know what the grader sends nor where SLO misses come from.
- Change: middleware request log, metadata only (no prompt text): `GM_REQLOG=.run/reqlog.jsonl`. FCFS, 8192.
- Result (`reqlog_v3_fcfs8192.jsonl`): the grader sends `enable_thinking: true` on every request but `max_tokens` varies (8192 on
  about half, 32-1024 on the rest). With thinking forced on and a small `max_tokens` the model thinks until the budget is gone:
  93 of 341 requests ended with `finish_reason=length` (always a miss, usually wrong). 45 TTFT misses were short prompts stuck behind
  an 8K-token prefill chunk. Queue time was about 0 at Validation load.
- Conclusion: two fixable causes (wasted decode on unfinishable thinking; head-of-line blocking by big prefill chunks).

## E2. Thinking budget (kept)

- Change: `thinking_token_budget = max_tokens - clamp(frac * max_tokens, 24, 1024)` set per request by the middleware (vLLM
  `--reasoning-config`). `max_tokens` and the thinking flag are untouched (organizers allow a cap on thinking). Fraction 0.4 in
  Final 2, 0.5 afterwards.
- Result: truncations fall from 93 to 13 in replay; accuracy retention stayed above 0.98 in every later Final (R 1.04-1.05 for INT8).
- Conclusion: kept in every config from Final 2 on. Final 2 itself scored worse (28.6); it combined the thinking budget with batched 2048 and no long-prefill
  threshold, FCFS (see E4), and we did not run the budget alone on the grader.

## E3. Newest-first priority alone (FAILED)

- Hypothesis: under overload FCFS cascades; serving the newest request first lets fresh requests make their TTFT.
- Change: middleware stamps `priority = -arrival_ms`, `--scheduling-policy priority` (Validation V2, Final 3).
- Result: V2 47.4 vs 50.6 (noise level, 58/100 both); Final 3 40.4 vs Final 1 41.8.
- Why: queue time was about 0 on Validation, and on the Final the misses come from inside the running batch (prefill chunks, decode
  pace). There is nothing to reorder.

## E4. Batch size and long-prefill threshold

- Hypothesis: a huge prefill chunk blocks short requests; a smaller token budget and vLLM's `--long-prefill-token-threshold`
  bound the stall.
- Results: Final 2 (batched 2048, no threshold, FCFS) 28.6, the worst Final. Final 4 (1024 + threshold 512, priority) 52.7. Replay on AWQ
  (sweep1): 1024: 316/481, 2048: 191/481, **4096: 183/481**, so larger batches lose on AWQ. After the switch to INT8 the order flips:
  1024: 281/300, 2048: 294/300, **threshold 256: 274/300**, 864: 287, 1152: 290, batched 3072/1152: 292 (sweeps 8 and 10).
  Final 10 (INT8, 2048, 512): 99.5.
- Conclusion: a small threshold (512, which in this hybrid model means one 288-token block per request per step) is essential;
  the best token budget depends on prefill speed (1024 for AWQ, 2048 for INT8).

## E5. SLO scheduler (kept)

- Hypothesis: a request counts only if TTFT and TPOT are both inside the SLO. Under overload the best you can do is keep a subset
  fully on time and let hopeless requests be late, without dropping anything.
- Change: `scripts/slo_scheduler.py` (`--scheduler-cls`): (1) deadline-ordered waiting queue; (2) late hold: requests that can no
  longer make their TTFT are not admitted while feasible ones wait; (3) doomed-decode skipping: a request whose first token already
  missed its deadline skips decode steps while the batch is busy (never dropped, bounded skipping); (4) admission cap that
  keeps estimated TPOT under 45 ms.
- Result: Final 5 **80.0** (vs 52.7). p95 TPOT was 81 ms and p95 TTFT 6.98 s (265/300 on time); Final 6 and 7 below tune the doomed-decode skipping.

### E5a. Without doomed-decode skipping (FAILED): Final 6, 74.0
Hypothesis: skipping is a risk, remove it. Result: p95 TTFT tail grew to 18.45 s, 256/300 on time. Keep skipping.

### E5b. TPOT guard 45 ms (kept): Final 7, 85.6
Doomed skipping only while the measured average TPOT is above its guard: p95 TTFT 7.76 s, TPOT 45 ms, 262/300.

### E5c. Other scheduler variants (replay, AWQ; no gain or worse)
- max wait 5 s: 182/481 (collapse); no late hold: 280/481; hide limit 3 s: 406/481; TPOT target 40 ms: 418/481 vs 416/481 default
  (noise): not adopted. Moore-Hodgson feasibility and token-budget admission were tried in a CPU simulator and lost to the defaults
  (simulator not part of this package).

### E5d. Two API server processes (FAILED): Final 8, 76.8
- Hypothesis: the single Python API process is a CPU bottleneck at peak; `--api-server-count 2` would cut TTFT overhead.
- Result: 251/300, p95 TTFT 16.99 s, S 76.8 vs 85.6. Reverted to one API server. We did not isolate the cause. The replay (411/481,
  equal to one server) did not predict the loss, so our replay cannot see front-end effects. Lesson: changes outside the
  engine need a real Final to validate.

## E6. INT8 W8A8 weights (kept): Final 9, 98.6
- Hypothesis: at peak the GPU is limited by prefill compute; INT8 tensor-core GEMMs are faster than W4A16 dequant kernels for large
  prefill batches, at the price of a bigger weight read in decode, which MTP k=3 amortizes.
- Change: `scripts/quant_w8a8_int8.py`: RTN per-channel INT8 weights, dynamic per-token INT8 activations, MTP head bf16.
- Result: replay 411/481 -> 471/481, p95 TPOT 49.5 -> 27.8 ms, J/pass 196 -> 143. Grader: 85.6 -> **98.6**, R 1.039 (accuracy
  retention is above 1 against the BF16 reference within grader noise), eta 3.45x.

## E7. MTP prefix-cache no-drop (neutral): Final 11
- Hypothesis: vLLM drops the last matching 288-token block on every prefix-cache hit when MTP is on, so most follow-up turns of tool
  episodes are recomputed. `disable_eagle_block_drop: true` keeps the block.
- Result: replay 293 -> 295/300; grader Final 10 -> Final 11: 295/300 both, S 99.5 both, J/req 166 -> 168, R 1.037 -> 1.054.
  Not distinguishable from noise; kept because target-model outputs are unchanged and it removes redundant prefill.

## E7b. Deadline-aware long-prefill chunk sizing (failed attempt, not adopted)
- Hypothesis: shrink a long prompt's prefill chunk when running requests are close to their deadlines, so short requests do not wait
  behind a 512-token chunk (`GM_SCHED_CHUNK=1`, implemented in `slo_scheduler.py`, default off).
- Result (replay v2 on the Final #11 log): chunk 293/300, chunk with safety 1.6 (`chunk_s16`) 292/300, versus 295/300 for the baseline.
  No gain, so it was not run on the grader and not adopted. Raw output: `results/sweep12_chunk/summary.tsv`. The toggle stays
  documented as available, off, no gain measured.

## E7c. Power cap on INT8 (replay; Finals stayed at 400 W)
- Replay v2 on INT8: 400 W 291/300 and 68.0 kJ; 300 W 286/300 and 62.7 kJ; 250 W 284/300 and 55.3 kJ (`results/sweep11_power`).
- Finals ran at a fixed 400 W because sE was already saturated (eta 3.6x > 3x), so energy savings could not raise S.
  250 W is the production recommendation (-19 % energy, -2.4 % items passed, replay estimate; no capped Final was run).

## E10. Batched 2560, long-prefill threshold 600, load-adaptive MTP K (kept): Final 14, S 100.0
- Observation: Finals 9-13 lose their last S points on sT (0.93-0.98): the few missed requests are the burst of four prompts of about
  20K tokens whose TTFT misses its deadline. vLLM floors long-prefill chunks to 288-token blocks, so threshold 512 was effectively 288
  tokens per request per step.
- Hypothesis: threshold 600 gives 576-token chunks, so the burst finishes prefill about twice as fast; adaptive MTP K
  (`num_speculative_tokens_per_batch_size [[1,32,3],[33,64,2],[65,128,1]]`) frees compute at high batch to pay for it; batched tokens
  2560 gives the step budget.
- Replay (closed-loop v2 on the Final #11 log, items of 300, `results/sweep13/summary.tsv`): base 2048/512 292, dynK 294, 2560/512 293,
  2560/512 + dynK 294, 2048/600 + dynK 293, **2560/600 + dynK 298** (sT 0.991). Only the combination helps clearly.
- Grader: Final 14 = 300/300, S 100.0 (R 1.045, sT 1.00, 169 J/req, p95 TTFT 1.30 s, p95 TPOT 25 ms, accuracy 91.8 %). Control: two
  repeat runs of the Final 11 config (Finals 12 and 13) scored 99.7 (297/300) and 99.5 (295/300), so run-to-run spread is about
  2 items and the 300/300 includes some luck; the replay direction agrees. Submitted configuration.
- Finals 12 and 13 were repeats of the Final 11 config (no change); the Final 12 request log also contains the requests of a
  Validation run that preceded it.

## E8. Things that did not work or were not used

| idea | result | why |
|---|---|---|
| Governor / dynamic power limit | dropped before any run | the rules only allow a power limit held constant for the whole run; a dynamic limit is on the wrong side of that line |
| Dual engine: single-stream Rust engine for idle moments + vLLM for bulk | dropped | the load never leaves the engine idle (about 120 concurrent requests even in the quiet phase), so the fast path would get 1-3 % of requests; no tool calling or thinking parser; two GPU processes violate the one-serving-process rule |
| Power caps on AWQ (300/250/200 W) | worse | on-time requests fall faster than joules (316 -> 273 -> 226 -> 112 of 481; J/pass 254 -> 257 -> 272 -> 491) |
| Power caps on INT8 (replay) | replay S_est up (91.8 -> 98.6 at 250 W) but p95 TTFT ratio 1.00 and TPOT 41 ms | thin margin; Finals used 400 W because sE was saturated; 250 W is the production recommendation (-19 % energy, -2.4 % items, replay) |
| fp32 GDN state + fused CUDA decode kernel | 202/481 vs 316/481 | state traffic doubles; TPOT 74 ms |
| bf16 GDN state + CUDA kernel | 408/481 vs 411/481 | no gain, accuracy risk |
| Thai-tuned MTP head | 318/481 vs 316/481 (priority), 401 vs 411 (SLO scheduler) | Thai is 10 % of the load; no gain; not used |
| Load-adaptive speculative k on AWQ (open-loop) | 402/481 vs 411/481 | no gain there; adopted later on INT8 together with batched 2560 / threshold 600 (E10) |
| Batched 4096 | 183/481 | head-of-line blocking |
| Raising the long-prefill threshold to 864/1152 (batched 2048, K=3) | 287, 290 vs 293 of 300 | short prompts wait behind long chunks; 600 with batched 2560 and adaptive K did work (E10) |
| SGLang | not used | lost to vLLM in earlier offline benchmarks on this model |
| Response caching, disabling thinking, changing sampling | forbidden | rules |

## E9. Evaluation harness (tooling)

`replay_v2.py` rebuilds items from a request log and replays their load closed-loop; it is how Finals 9-14 and sweeps 8-13 were
chosen (see [METHOD.md](METHOD.md)). It reproduces the grader's passed-item count to within one item on Finals 5-11 (not re-checked on 12-14).
