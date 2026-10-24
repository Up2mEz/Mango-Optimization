# Method: how we measured

The grader allows 2 Final and 5 Validation submissions per hour and does not show per-request data. We therefore built a
measurement loop around what we could observe on our own server.

## 1. Metadata request log

`scripts/slo_priority.py` (a vLLM middleware) writes one JSON line per request to `GM_REQLOG`. Keys: `t0` (arrival, epoch s),
`max_tokens`, `temperature`, `stream`, `n_msgs`, `roles` (first letters of the message roles), `n_tools`, `tool_choice`,
`tmpl_kwargs`, `think_budget`, and the server-side outcome: `ttft`, `tpot_ms`, `e2e`, `prompt_tokens`, `out_tokens`, `chunks`, `bytes`,
`finish`, `status`. **No message text is stored.** The files `results/reqlog_*.jsonl` are such logs of the real graded runs
(Validation V3 and Finals 2-11).

## 2. Item reconstruction (`scripts/item_lib.py`)

The grader counts items (plain requests or multi-turn tool episodes, up to 15 turns), but the server only sees requests. Requests with
`max_tokens != 8192` are the grader's probe phase; the 8192 ones are graded items. Tool episodes are re-chained by matching the role
sequence of turn k+1 (turn k's roles plus assistant and tool messages), the number of tools, a growing prompt, `finish=tool_calls` of
the previous turn and a client gap of 0 to 0.3 s. An item passes only if all its turns pass: streamed, no error, not truncated
(`finish != length`), TTFT <= 0.5 s + 0.25 s per 1000 prompt tokens, average TPOT <= 50 ms.

Check of the scorer against the real grader (`python scripts/replay_v2.py --score-real results/reqlog_f*.jsonl`): items passed,
scorer vs grader: Final 5 266 vs 265, Final 6 255 vs 256, Final 7 262 vs 262, Final 8 252 vs 251, Final 9 287 vs 287, Final 10 295 vs 295,
Final 11 296 vs 295. The scorer reproduces the grader to within one item. The matching p95 TTFT and TPOT values are reproduced for
Finals 5-8 (grader values in `replay_v2.py`).

## 3. Closed-loop replay (`scripts/replay_v2.py`)

Given a real log, it sends each item's first turn at its real offset, and turn k+1 after turn k finishes plus the real client gap, with
the real prompt-token growth. Prompt text is synthetic, but turn k+1 starts with turn k's text exactly, so prefix caching behaves
as in the real run. Outputs are forced to the real output length (`min_tokens = max_tokens`, `ignore_eos`, thinking off), so prefill
and decode work and arrival bursts match. Probe-phase requests are replayed open-loop. GPU energy comes from NVML (`pynvml`).
The summary has items passed, sT, p95 TTFT ratio and TPOT per item, joules, J per passed item and `S_est` with the grader's
formula (baseline eta 0.00163 items/J in the replay).

Limits: accuracy is not measured (synthetic text), so accuracy always comes from the grader; the replay is conservative in
absolute S (91-93 for configs that scored 98.6-99.5), and it does not see front-end or network effects (it missed the Final 8 loss).
It is used to compare arms against each other. `replay_load.py` is the first, open-loop version (sweeps 1-7, per-request counts).

## 4. Sweeps (`scripts/sweep_replay.sh`)

For each arm: stop server, start `serve.sh` with the arm's environment, replay, append one line to `OUT/summary.tsv` and keep the JSON.
Raw outputs are in `results/sweep*/`. A GPU lock file serializes arms; the script must never run during a graded run (one serving
process rule).

## 5. Noise

Validation (100 requests) differs by about 3 S points between identical configs (V1 vs V3), so only differences larger than that, or
confirmed on 300-request replays/Finals, are treated as real. Finals 10 and 11 are indistinguishable.
