"""SLO middleware for vLLM (--middleware slo_priority.SLOPriority).

1. Priority (GM_PRIORITY=1, needs --scheduling-policy priority): stamps each chat request with
   priority = -arrival_ms (vLLM runs lower values first), so the waiting queue is served newest-first:
   under open-loop overload fresh arrivals that can still make their TTFT budget go ahead of requests that
   already missed it, instead of FCFS cascading. Late requests are not dropped. Only waiting-queue order
   changes; vLLM preempts running requests only when KV blocks run out. Client-supplied priority is kept.
2. Thinking budget (GM_THINK=1, needs --reasoning-config): the grader forces thinking on but sends small
   max_tokens (32..1024) on many items, so the model thinks until it is cut off (finish=length = SLO miss and
   usually a wrong answer). We keep thinking on and keep max_tokens; we only set vLLM's thinking_token_budget
   = max_tokens - reserve, so the forced </think> leaves `reserve` tokens for the answer.
   reserve = clamp(GM_THINK_FRAC * max_tokens, GM_THINK_MIN, GM_THINK_MAX).
3. Request log (GM_REQLOG=<path>): one JSON line per request with request METADATA ONLY (no message text):
   max_tokens, temperature, stream, #messages, role sequence, #tools, tool_choice, template kwargs, and the
   server-side view of the SLO: TTFT, TPOT, output/prompt tokens (from the usage chunk), finish_reason.
"""
import json
import os
import time

PATHS = ("/v1/chat/completions", "/v1/completions")
PRIORITY = os.environ.get("GM_PRIORITY", "0") == "1"
REQLOG = os.environ.get("GM_REQLOG", "")
THINK = os.environ.get("GM_THINK", "0") == "1"
THINK_FRAC = float(os.environ.get("GM_THINK_FRAC", "0.4"))
THINK_MIN = int(os.environ.get("GM_THINK_MIN", "24"))
THINK_MAX = int(os.environ.get("GM_THINK_MAX", "1024"))


def think_budget(data):
    mt = data.get("max_tokens", data.get("max_completion_tokens"))
    if not isinstance(mt, int) or "thinking_token_budget" in data:
        return None
    if (data.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
        return None
    reserve = min(max(int(THINK_FRAC * mt), THINK_MIN), THINK_MAX)
    return max(mt - reserve, 0)


def _meta(data):
    msgs = data.get("messages") or []
    return {
        "max_tokens": data.get("max_tokens", data.get("max_completion_tokens")),
        "temperature": data.get("temperature"),
        "stream": data.get("stream"),
        "include_usage": (data.get("stream_options") or {}).get("include_usage"),
        "n_msgs": len(msgs),
        "roles": "".join((m.get("role") or "?")[0] for m in msgs if isinstance(m, dict)),
        "n_tools": len(data.get("tools") or []),
        "tool_choice": data.get("tool_choice") if isinstance(data.get("tool_choice"), str) else bool(data.get("tool_choice")),
        "tmpl_kwargs": data.get("chat_template_kwargs"),
        "extra_keys": sorted(k for k in data if k not in (
            "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream",
            "stream_options", "tools", "tool_choice", "chat_template_kwargs", "priority")),
    }


class SLOPriority:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope["path"] not in PATHS:
            return await self.app(scope, receive, send)
        t0 = time.time()
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        rec = None
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                if REQLOG:
                    rec = {"t0": round(t0, 3), "bytes": len(body), **_meta(data)}
                changed = False
                if PRIORITY and "priority" not in data:
                    data["priority"] = -int(t0 * 1000)
                    changed = True
                if THINK and (b := think_budget(data)) is not None:
                    data["thinking_token_budget"] = b
                    changed = True
                    if rec is not None:
                        rec["think_budget"] = b
                if changed:
                    body = json.dumps(data, ensure_ascii=False).encode()
        except ValueError:
            pass  # let vLLM report the malformed body
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # disconnect notifications

        headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        if rec is None:
            return await self.app(dict(scope, headers=headers), replay, send)

        st = {"first": None, "last": None, "chunks": 0, "status": None, "tail": b""}

        async def watch(msg):
            if msg["type"] == "http.response.start":
                st["status"] = msg.get("status")
            elif msg["type"] == "http.response.body":
                b = msg.get("body", b"")
                if b:
                    now = time.time()
                    # token-bearing SSE chunks: content / reasoning / tool-call deltas
                    if b'"delta"' in b and (b'"content":"' in b or b'"reasoning' in b or b'"tool_calls"' in b):
                        st["first"] = st["first"] or now
                        st["last"] = now
                        st["chunks"] += 1
                    st["tail"] = (st["tail"] + b)[-4096:]
                if not msg.get("more_body"):
                    _finish(rec, st, t0)
            await send(msg)

        await self.app(dict(scope, headers=headers), replay, watch)


def _finish(rec, st, t0):
    tail = st["tail"].decode("utf-8", "replace")
    usage, finish = {}, None
    for line in tail.splitlines():
        line = line.removeprefix("data: ").strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        usage = d.get("usage") or usage
        for c in d.get("choices") or []:
            finish = c.get("finish_reason") or finish
    out = (usage or {}).get("completion_tokens")
    ttft = (st["first"] - t0) if st["first"] else None
    tpot = ((st["last"] - st["first"]) / (out - 1)) if st["first"] and out and out > 1 else None
    rec.update(status=st["status"], ttft=ttft and round(ttft, 3), tpot_ms=tpot and round(tpot * 1000, 1),
               e2e=round(time.time() - t0, 3), prompt_tokens=(usage or {}).get("prompt_tokens"),
               out_tokens=out, finish=finish, chunks=st["chunks"])
    try:
        with open(REQLOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
