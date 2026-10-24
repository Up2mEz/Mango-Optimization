# GreenMind submission: SLO-aware INT8 serving of MANGO1.5-Qwen3.5-9B on one A100

Team: ThunderPermanent (Eakkachat Sathaphon, Paradee Pimoleakaksorn, Pongsura Pongpisan, Thanakrit Samoena)

We serve `CMKL/MANGO1.5-Qwen3.5-9B` with **one vLLM 0.32.0 process** on one A100 40 GB at a **fixed 400 W power limit**.
The submitted configuration (Final #14) reached **S = 100.0** on the grader: **300/300 requests on time**, **3.64x** the baseline's
on-time requests per joule, **169 J/request**, p95 TTFT **1.30 s**, p95 TPOT **25 ms**, accuracy **91.8 %** versus 87.8 % for the
BF16 baseline (R 1.045). The same stack without the last tuning step (Finals #11-#13) scored 99.5-99.7. This repository contains only the code that serves and reproduces that
result, the request logs and sweep outputs we used to pick the configuration, and our write-ups, including what did not work.

## Architecture

```
 grader (open loop, streaming, thinking on, tool calling)
        |  OpenAI-compatible /v1/chat/completions
        v
 SLO middleware  (scripts/slo_priority.py, vLLM --middleware)
   - thinking budget: thinking_token_budget = max_tokens - clamp(0.5*max_tokens, 24, 1024)  (max_tokens, thinking flag untouched)
   - metadata-only request log (no prompt text): arrival, token counts, TTFT/TPOT, finish reason
        |
        v
 vLLM 0.32.0  (single process, :8000, model name "mango")
   - custom SLO scheduler (scripts/slo_scheduler.py): deadline order, late hold, doomed-decode skipping (guarded at 45 ms TPOT),
     TPOT-capped admission
   - INT8 W8A8 weights (CUTLASS int8 GEMM), built from the official BF16 weights by scripts/quant_w8a8_int8.py
   - MTP self-speculative decoding (the model's own head, bf16), load-adaptive K: 3 up to 32 running sequences, 2 up to 64, 1 above; GDN state fp16
   - chunked prefill: max-num-batched-tokens 2560, long-prefill-token-threshold 600 (576 effective), max-num-seqs 128
   - prefix caching on, MTP eagle block drop disabled (GM_SPEC_NODROP=1)
        |
        v
 NVIDIA A100-SXM4 40 GB, power limit 400 W set once before start and held for the whole run
```

## Repository layout

```
README.md  NOTICE.md  PACKAGE_CHECKLIST.md
scripts/   serve.sh  serve_final.sh  setup.sh  slo_priority.py  slo_scheduler.py  test_slo_scheduler.py
           quant_w8a8_int8.py  replay_v2.py  replay_load.py  sweep_replay.sh  item_lib.py
results/   RESULTS.md  reqlog_*.jsonl (real grader runs, metadata only)  sweep*/ (summary.tsv + per-arm json)
docs/      EXPERIMENTS.md  METHOD.md  MODEL_CARD.md  HF_UPLOAD.md
slides/    README.md (the pitch PDF is submitted separately through the organizers' form)
```

## Quickstart

Requirements: Ubuntu 24.04, NVIDIA driver 580 (CUDA 12.x wheels), one A100 40 GB, root/sudo (for `nvidia-smi -pl`), about 40 GB free disk.

```bash
git clone https://github.com/Up2mEz/Mango-Optimization.git
cd Mango-Optimization
# 1a. build the INT8 checkpoint locally from the official BF16 weights (CPU only, ~3 min, ~5 GB RAM) ...
scripts/setup.sh --convert
# 1b. ... or download our pre-built INT8 weights instead (see "Weights" below)
GM_HF_INT8=Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8 scripts/setup.sh --download-int8
# 2. serve the submitted configuration (Final #14)
scripts/serve_final.sh start        # stop | status | logs
# 3. smoke test
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mango","messages":[{"role":"user","content":"Say hi"}],"max_tokens":256}'
```

Paths are configurable: `GM_MODEL` (model dir, default `models/w8a8-int8`), `GM_VENV` (default `.venv-vllm`),
`GM_MODELS` (setup.sh model root), `GM_REQLOG` (request log path, default `.run/reqlog.jsonl`, empty disables it).

## Exact commands for the graded configurations

`serve_final.sh` expands to (environment of `scripts/serve.sh`):

```bash
GM_MODEL=models/w8a8-int8 GM_POWER=400 GM_THINK=1 GM_THINK_FRAC=0.5 GM_PRIORITY=0 GM_MAX_SEQS=128 GM_SPEC_K=3 \
GM_MAX_BATCHED=2560 GM_SPEC_NODROP=1 GM_SCHED_DOOMED=1 GM_SCHED_DOOMED_GUARD=0.045 \
GM_SPEC_DYN='[[1,32,3],[33,64,2],[65,128,1]]' \
GM_ARGS="--long-prefill-token-threshold 600 --scheduler-cls slo_scheduler.SLOScheduler" \
scripts/serve.sh start
# vllm serve models/w8a8-int8 --served-model-name mango --max-model-len 40960 --max-num-seqs 128 --max-num-batched-tokens 2560
#   --gpu-memory-utilization 0.92 --enable-prefix-caching --enable-chunked-prefill --reasoning-parser qwen3
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder
#   --speculative-config '{"model": "models/w8a8-int8", "num_speculative_tokens": 3, "method": "mtp", "disable_eagle_block_drop": true,
#      "num_speculative_tokens_per_batch_size": [[1,32,3],[33,64,2],[65,128,1]]}'
#   --middleware slo_priority.SLOPriority --reasoning-config '{"reasoning_parser": "qwen3", "reasoning_start_str": "<think>", "reasoning_end_str": "\n</think>\n\n"}'
#   --long-prefill-token-threshold 600 --scheduler-cls slo_scheduler.SLOScheduler --mamba-ssm-cache-dtype float16
```

Finals #11-#13 are the same with `GM_SPEC_DYN=` (empty), `GM_MAX_BATCHED=2048` and threshold 512 (the earlier profile; `serve_final.sh` hard-codes the batched-token and `GM_ARGS` values, so edit those two lines), Final #10 additionally with `GM_SPEC_NODROP=0`. Configurations of Finals #1-#9 are listed in
[results/RESULTS.md](results/RESULTS.md); each is an environment override of `serve.sh` (AWQ model dir for #1-#8).

Reproduce the offline evaluation (no grader needed, needs the GPU and a running replay-capable server; stop other GPU work first):

```bash
# score a real grader log with our item scorer (CPU only)
python scripts/replay_v2.py --score-real results/reqlog_f11_int8_b2048_nodrop.jsonl
# replay the Final-load shape against candidate configs, one line per config in OUT/summary.tsv
REPLAY=replay_v2.py scripts/sweep_replay.sh results/mysweep results/reqlog_f10_int8_b2048.jsonl \
  "b2048_t512|GM_MODEL=models/w8a8-int8 GM_THINK=1 GM_THINK_FRAC=0.5 GM_MAX_BATCHED=2048 GM_SCHED_DOOMED=1 GM_SCHED_DOOMED_GUARD=0.045 GM_ARGS='--long-prefill-token-threshold 512 --scheduler-cls slo_scheduler.SLOScheduler'"
# scheduler unit/differential test (CPU only, ~1 min)
CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts .venv-vllm/bin/python scripts/test_slo_scheduler.py
```

## Rules-compliance checklist

| rule (PARTICIPANT_BRIEFING) | how we comply |
|---|---|
| Serve CMKL/MANGO1.5-Qwen3.5-9B or a derived model | derived: INT8 W8A8 of the official BF16 weights |
| Derived weights made by us during the event, from official weights | `scripts/quant_w8a8_int8.py` run on the official BF16 weights during the event; RTN needs no calibration data |
| Open source only, no hosted APIs, no external calls | vLLM, PyTorch, compressed-tensors, safetensors; the server makes no external calls |
| Spec-decode head disclosed, its energy counts | the model's own MTP head (bf16), k = 3; no separate draft model. All 14 Finals used the stock head (the CMKL checkpoint's for AWQ, the one copied from the official BF16 checkpoint for INT8); the Thai-tuned head was only a replay experiment |
| OpenAI-compatible streaming chat with usage, tool calling with tool_choice auto | stock vLLM endpoint, `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| Grader fixes max_tokens, temperature, thinking mode, sampling | we never change them; the middleware only adds vLLM's `thinking_token_budget` (a cap on thinking, confirmed allowed by the organizers) |
| Fixed power limit held for the whole run | `nvidia-smi -pl 400` before server start, never changed. No governor, no dynamic power (we dropped one for this reason) |
| No settings that apply only during the measured window; no LD_PRELOAD, no NVML tampering | none used |
| One serving process, no other GPU workloads during a graded run | one `vllm serve`; replay/sweep tools are never run during a graded run |
| No response caching of graded requests; normal KV and prefix caching is fine | only vLLM's automatic prefix caching (KV and GDN state); no response cache |
| Source available to organizers | this repository |

## Weights

- Base model: [CMKL/MANGO1.5-Qwen3.5-9B](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B) (official BF16, unchanged).
- Our optimized weights (INT8 W8A8, derived from the base model): [Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8](https://huggingface.co/Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8) (model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).
  They can be rebuilt deterministically (RTN, no calibration data) from the base model with `scripts/setup.sh --convert`.
- No other modified weights are used by the submitted configuration.

## Documents

- [results/RESULTS.md](results/RESULTS.md): all 14 Final submissions with configuration and score, plus the key sweeps.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): hypothesis, change and result for every step, including failed attempts.
- [docs/METHOD.md](docs/METHOD.md): how we measured (metadata request log, item reconstruction, closed-loop replay).
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/HF_UPLOAD.md](docs/HF_UPLOAD.md): Hugging Face model card and upload steps.
- [slides/](slides/): instructions for the pitch PDF, which is submitted separately.

## Credits and licences

See [NOTICE.md](NOTICE.md). vLLM (Apache-2.0) is used unmodified; our scheduler and middleware plug in through vLLM's
`--scheduler-cls` and `--middleware` extension points. Base model by CMKL (licence as published on its model card).
