#!/usr/bin/env bash
# Sweep serve.sh configs against a replayed real grader load (replay_load.py); one line per config in $OUT/summary.tsv.
#   scripts/sweep_replay.sh OUTDIR LOG "name|ENV=.. ENV=.." ["name|..."]...
# REPLAY=replay_v2.py (default replay_load.py) selects the replayer; v2 = closed-loop per-item (items passed, sT, p95 over items, S_est).
# Each arm holds $LOCK for stop -> start -> replay. `flock -o` keeps the lock fd out of the server process,
# so a server left running between arms never holds the lock (it used to inherit it and deadlock the next arm).
# The last arm's server is left running. Never run this during a real grader submission.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$1; LOG=$2; shift 2
mkdir -p "$OUT"
PY=${GM_PYTHON:-${GM_VENV:-$ROOT/.venv-vllm}/bin/python}
LOCK=${GM_LOCK:-$ROOT/.run/gpu.lock}; mkdir -p "$ROOT/.run"
REPLAY=${REPLAY:-replay_load.py}
for spec in "$@"; do
  name=${spec%%|*}; envs=${spec#*|}
  echo "=== $name: $envs"
  ARM_ENVS=$envs flock -o "$LOCK" bash -c "
    cd '$ROOT'
    scripts/serve.sh stop >/dev/null
    env GM_REQLOG= $envs scripts/serve.sh start | tail -1
    if ! curl -sf localhost:8000/health >/dev/null; then
      echo 'server failed'; tail -20 .run/gm_vllm.log; scripts/serve.sh stop >/dev/null; exit 1
    fi
    $PY scripts/$REPLAY '$LOG' --out '$OUT/$name.json' >/dev/null
  " || { printf '%s\tFAILED\t%s\n' "$name" "$envs" >> "$OUT/summary.tsv"; continue; }
  ARM_NAME=$name ARM_ENVS=$envs $PY -c "
import json, os
s = json.load(open('$OUT/$name.json'))['summary']
if 'items_pass' in s:  # replay_v2
    row = [s['items_pass'], s['items'], 'sT=%.3f' % s['sT'], round(s['p95_ttft_ratio_item'], 2), round(s['p95_tpot_ms_item'], 1),
           s['joules'], s['J_per_pass_item'], 'S=%s' % s.get('S_est')]
else:
    row = [s['graded_pass'], s['graded_n'], s['graded_fails'], round(s['p95_ttft_ratio'], 2), round(s['p95_tpot_ms'], 1),
           s['joules'], s['J_per_graded_pass']]
print('\t'.join(map(str, [os.environ['ARM_NAME']] + row + [os.environ['ARM_ENVS']])))
" | tee -a "$OUT/summary.tsv"
done
