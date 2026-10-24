#!/usr/bin/env bash
# GreenMind grader endpoint: ONE vLLM process (rules: single serving process, fixed power limit).
#
#   scripts/serve.sh start|stop|status|logs
#
# Env knobs (defaults = current best known config):
#   GM_MODEL        model dir             (default $ROOT/models/w8a8-int8 = our INT8 W8A8; official BF16 or CMKL AWQ also work)
#   GM_SPEC_MODEL   MTP head source        (default = GM_MODEL; set to the mtpft6 merged ckpt for Thai MTP)
#   GM_SPEC_K       num_speculative_tokens (default 3; 0 disables speculation)
#   GM_SPEC_DYN     load-adaptive K, e.g. "[[1,32,3],[33,64,2],[65,128,1]]" (batch-size ranges -> K)
#   GM_SPEC_NODROP  1 = "disable_eagle_block_drop": true (MTP prefix-cache hits keep their last 288-token block;
#                   vLLM 0.32 eagle block drop, see docs/EXPERIMENTS.md)
#   GM_PORT         public port            (default 8000)   GM_NAME served model name (default mango)
#   GM_MAX_LEN      max-model-len          (default 40960 = 32K prompt + 8K output)
#   GM_MAX_SEQS     max-num-seqs           (default 128)
#   GM_MAX_BATCHED  max-num-batched-tokens (default 2048)
#   GM_MEM          gpu-memory-utilization (default 0.92)
#   GM_POWER        fixed power limit W, applied before start and held (default: leave as is)
#   GM_PRIORITY     1 = SLO-aware newest-first waiting queue (scripts/slo_priority.py)
#   GM_THINK        1 = thinking budget = max_tokens - reserve (see slo_priority.py; GM_THINK_FRAC/MIN/MAX)
#   GM_REQLOG       metadata-only request log (default .run/reqlog.jsonl; "" disables)
#   GM_CLOCK        locked SM clock MHz (nvidia-smi -lgc), unset = reset to default clocks
#   GM_EXTRA        extra vLLM args        (default "--mamba-ssm-cache-dtype float16")
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN="$ROOT/.run"; mkdir -p "$RUN"
PIDF="$RUN/gm_vllm.pid"; LOG="$RUN/gm_vllm.log"
VENV="${GM_VENV:-$ROOT/.venv-vllm}"
MODEL="${GM_MODEL:-$ROOT/models/w8a8-int8}"
SPEC_MODEL="${GM_SPEC_MODEL:-$MODEL}"
SPEC_K="${GM_SPEC_K:-3}"
PORT="${GM_PORT:-8000}"
NAME="${GM_NAME:-mango}"
EXTRA="${GM_EXTRA---mamba-ssm-cache-dtype float16}"

start() {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "already running (pid $(cat "$PIDF"))"; return 0; fi
  if [ -n "${GM_CLOCK:-}" ]; then  # locked SM clock (MHz), held for the whole run; reset with: sudo nvidia-smi -rgc
    sudo nvidia-smi -lgc "$GM_CLOCK,$GM_CLOCK" || { echo "clock lock failed"; return 1; }
  else
    sudo nvidia-smi -rgc >/dev/null 2>&1
  fi
  if [ -n "${GM_POWER:-}" ]; then
    sudo nvidia-smi -pm 1 >/dev/null; sudo nvidia-smi -pl "$GM_POWER" || { echo "power limit failed"; return 1; }
  fi
  local spec=()
  local dyn=""
  [ -n "${GM_SPEC_DYN:-}" ] && dyn=", \"num_speculative_tokens_per_batch_size\": $GM_SPEC_DYN"
  [ "${GM_SPEC_NODROP:-0}" = 1 ] && dyn="$dyn, \"disable_eagle_block_drop\": true"
  [ "$SPEC_K" != 0 ] && spec=(--speculative-config "{\"model\": \"$SPEC_MODEL\", \"num_speculative_tokens\": $SPEC_K, \"method\": \"mtp\"$dyn}")
  local prio=(--middleware slo_priority.SLOPriority)
  [ "${GM_PRIORITY:-0}" = 1 ] && prio+=(--scheduling-policy priority)
  [ "${GM_THINK:-0}" = 1 ] && prio+=(--reasoning-config '{"reasoning_parser": "qwen3", "reasoning_start_str": "<think>", "reasoning_end_str": "\n</think>\n\n"}')
  # shellcheck disable=SC2086
  GM_PRIORITY=${GM_PRIORITY:-0} GM_THINK=${GM_THINK:-0} GM_REQLOG=${GM_REQLOG-$RUN/reqlog.jsonl} PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${GM_MAX_LEN:-40960}" \
    --max-num-seqs "${GM_MAX_SEQS:-128}" \
    --max-num-batched-tokens "${GM_MAX_BATCHED:-2048}" \
    --gpu-memory-utilization "${GM_MEM:-0.92}" \
    --enable-prefix-caching --enable-chunked-prefill \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    "${spec[@]}" "${prio[@]}" $EXTRA ${GM_ARGS:-} > "$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "vLLM pid $(cat "$PIDF") on :$PORT (log $LOG); waiting for /health"
  for _ in $(seq 1 120); do
    curl -sf "localhost:$PORT/health" >/dev/null && { echo "ready"; return 0; }
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "vLLM exited"; tail -30 "$LOG"; return 1; }
    sleep 5
  done
  echo "not ready after 10 min"; return 1
}

stop() {
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 1; done
  pkill -9 -f "[v]llm serve" 2>/dev/null; rm -f "$PIDF"; echo stopped
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) curl -s "localhost:$PORT/health" -o /dev/null -w "health %{http_code}\n"; nvidia-smi --query-gpu=power.draw,power.limit,memory.used --format=csv,noheader ;;
  logs) tail -f "$LOG" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
