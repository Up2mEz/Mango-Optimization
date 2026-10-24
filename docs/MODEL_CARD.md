---
base_model: CMKL/MANGO1.5-Qwen3.5-9B
base_model_relation: quantized
language:
  - th
  - en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE
tags:
  - thai
  - bilingual
  - qwen3.5
  - int8
  - w8a8
  - compressed-tensors
  - vllm
  - mtp
  - speculative-decoding
  - quantized
pipeline_tag: text-generation
library_name: vllm
---

# Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8: MANGO1.5-Qwen3.5-9B, INT8 W8A8

INT8 (weights and activations) quantization of [`CMKL/MANGO1.5-Qwen3.5-9B`](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B),
made by team ThunderPermanent (Eakkachat Sathaphon, Paradee Pimoleakaksorn, Pongsura Pongpisan, Thanakrit Samoena) during the GreenMind hackathon (October 2026) from the official BF16 weights. Source code, serving
configuration and results: https://github.com/Up2mEz/Mango-Optimization.

## Method

- Scheme: `compressed-tensors` format `int-quantized`, W8A8.
- Weights: symmetric **per-output-channel INT8, round-to-nearest (RTN)**, scale = absmax / 127 (float32, shape `[out, 1]`).
  No calibration data and no fine-tuning.
- Activations: **dynamic per-token symmetric INT8**, quantized on the fly by the kernel (declared in `config.json` only).
- Quantized (200 linear layers): all 32 MLPs (gate/up/down), q/k/v/o of the 8 full-attention layers, `in_proj_qkv`, `in_proj_z` and
  `out_proj` of the 24 Gated-DeltaNet layers.
- Kept in bf16 (ignore list: `lm_head`, `re:.*visual.*`, `re:.*mtp.*`, `re:.*linear_attn\.in_proj_[ab]$`): `lm_head`, embeddings, norms,
  GDN `in_proj_a`/`in_proj_b`, `conv1d`, `A_log`, `dt_bias`, the vision tower, and the **MTP head** (copied verbatim, so speculative
  decoding works as in the base model).
- Produced by `scripts/quant_w8a8_int8.py` in the GitHub repository (about 3 minutes on CPU, shard by shard). Rebuild:
  `scripts/setup.sh --convert`.

## How to serve (vLLM 0.32.0, one A100 40 GB)

```bash
vllm serve Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8 --served-model-name mango --max-model-len 40960 --max-num-seqs 128 \
  --max-num-batched-tokens 2560 --long-prefill-token-threshold 600 --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"model": "Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8", "num_speculative_tokens": 3, "method": "mtp",
  "num_speculative_tokens_per_batch_size": [[1,32,3],[33,64,2],[65,128,1]]}' \
  --mamba-ssm-cache-dtype float16
```

The full graded profile (SLO scheduler, thinking budget middleware, fixed 400 W) is `scripts/serve_final.sh` in the repository.

## Results (GreenMind grader, Final = 300 requests)

| Final | S | R (accuracy vs BF16 baseline) | SLO-met | J/request | p95 TTFT | p95 TPOT |
|---|---|---|---|---|---|---|
| #9 | 98.6 | 1.039 | 287/300 | 170 | 3.42 s | 26 ms |
| #10 | 99.5 | 1.037 | 295/300 | 166 | 2.32 s | 27 ms |
| #11 | 99.5 | 1.054 | 295/300 | 168 | 2.42 s | 27 ms |
| #12 | 99.7 | 1.045 | 297/300 | 166 | 1.62 s | 24 ms |
| #13 | 99.5 | 1.037 | 295/300 | 176 | 2.43 s | 27 ms |
| #14 (submitted) | 100.0 | 1.045 | 300/300 | 169 | 1.30 s | 25 ms |

Grader macro accuracy: 91.2 / 91.0 / 92.5 / 91.8 / 91.0 / 91.8 % versus 87.8 % for the BF16 baseline (grader-measured, six task classes, with the
thinking-budget middleware). These are task-level results of the full serving stack, not a standalone benchmark of the weights.

## Limitations

RTN without calibration: expect small quality differences against BF16 on tasks we did not measure. Not evaluated outside
the GreenMind grader classes (chat/instructions, RAG, tool calling, reasoning/math/code, Thai, long context).

## Licence

Same licence as the base model `CMKL/MANGO1.5-Qwen3.5-9B` (see its model card). Credits: CMKL (MANGO1.5), Qwen team (Qwen3.5),
vLLM and compressed-tensors projects.
