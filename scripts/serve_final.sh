#!/usr/bin/env bash
# GreenMind submission profile = Final #14 (S = 100.0, 300/300 SLO-met, R 1.045, 169 J/request; Finals #12/#13 = Final #11 config).
#   scripts/serve_final.sh start|stop|status|logs
# One vLLM 0.32 process on :8000 (model name "mango"), fixed 400 W power limit, held for the whole run:
#   weights   : our W8A8 INT8 (RTN, per-channel weights, dynamic per-token activations) made during the event from the
#               official CMKL/MANGO1.5-Qwen3.5-9B BF16 weights (scripts/quant_w8a8_int8.py), MTP head kept bf16.
#               Model dir: $GM_MODEL (default <repo>/models/w8a8-int8, created by scripts/setup.sh).
#   decoding  : MTP self-speculative, load-adaptive K via num_speculative_tokens_per_batch_size [[1,32,3],[33,64,2],[65,128,1]]
#               (K=3 up to 32 running seqs, 2 up to 64, 1 above; GM_SPEC_DYN), GDN state fp16 (vLLM default Triton spec-decode path)
#   prefix    : GM_SPEC_NODROP=1 -> "disable_eagle_block_drop": true in --speculative-config (Finals #11-#14).
#               With MTP, vLLM drops the last matching 288-token block of every prefix-cache hit; this keeps it.
#               Target-model outputs are unchanged, it only removes redundant prefill (follow-up tool turns).
#   batching  : max-num-batched-tokens 2560, long-prefill-token-threshold 600 (chunks floor to 288-token blocks: 576 effective;
#               512 was effectively 288), max-num-seqs 128
#   scheduler : scripts/slo_scheduler.py (deadline order, late hold, TPOT-capped admission,
#               doomed-decode skipping guarded at 45 ms average TPOT)
#   middleware: scripts/slo_priority.py (thinking_token_budget = max_tokens - clamp(0.5*max_tokens, 24, 1024);
#               metadata-only request log at $GM_REQLOG, default .run/reqlog.jsonl)
#
# Earlier Finals: #10 = batched 2048 / threshold 512, no-drop off, GM_SPEC_DYN empty;  #11-#13 = same with GM_SPEC_NODROP=1;
#   #14 = this profile as shipped (#11-#13 batched 2048 / threshold 512 / no adaptive K: see results/RESULTS.md).
#
# Optional toggle GM_SCHED_CHUNK=1 (deadline-aware long-prefill chunks, slo_scheduler.py): available, off, no gain measured
# (replay 293/300 and 292/300 vs 295/300; see results/RESULTS.md 2.8). Not used by any Final.
# Power: Finals ran at a fixed 400 W (sE was already saturated). 250 W is the production recommendation (replay: -19 % energy,
# -2.4 % items); see results/RESULTS.md 2.3.
exec env GM_MODEL="${GM_MODEL:-$(cd "$(dirname "$0")/.." && pwd)/models/w8a8-int8}" \
  GM_POWER=400 GM_THINK=1 GM_THINK_FRAC=0.5 GM_PRIORITY=0 \
  GM_MAX_SEQS=128 GM_SPEC_K=3 GM_MAX_BATCHED=2560 \
  GM_SPEC_NODROP="${GM_SPEC_NODROP:-1}" \
  GM_SCHED_DOOMED=1 GM_SCHED_DOOMED_GUARD=0.045 \
  GM_SPEC_DYN="${GM_SPEC_DYN-[[1,32,3],[33,64,2],[65,128,1]]}" GM_SPEC_NODROP="${GM_SPEC_NODROP-1}" \
  GM_ARGS="--long-prefill-token-threshold 600 --scheduler-cls slo_scheduler.SLOScheduler" \
  "$(dirname "$0")/serve.sh" "$@"
