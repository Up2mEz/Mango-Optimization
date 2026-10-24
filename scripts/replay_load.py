"""Replay the LOAD SHAPE of a real grader run against a server to compare scheduler configs.

Input: a metadata request log written by slo_priority.py (results/reqlog_*.jsonl): arrival offset,
prompt_tokens, out_tokens, max_tokens per request (no text). Each request is re-sent at its real offset with a
synthetic prompt of the same token length and forced output length (min_tokens = max_tokens = real out_tokens,
ignore_eos, thinking off), so prefill/decode work and arrival bursts match the real run. Measures TTFT/TPOT SLO
per request plus NVML energy. Accuracy is NOT measured (synthetic text).

  python scripts/replay_load.py results/reqlog_f3_prio_think05_b8192.jsonl [--only-graded]
"""
import argparse
import asyncio
import json
import random
import time

import httpx
import pynvml

WORDS = "alpha beta gamma delta river mountain data energy green model".split()


def prompt_of(n, seed):
    rnd = random.Random(seed)
    # ~1 token per word for this vocabulary (2000 words -> 2018 tokens incl. template)
    return " ".join(rnd.choice(WORDS) for _ in range(max(n - 18, 1))) + "\nContinue the list."


async def one(client, url, model, r, t_start, res):
    await asyncio.sleep(max(0.0, t_start + r["off"] - time.time()))
    out = max(int(r["out_tokens"] or 1), 1)
    body = {"model": model, "messages": [{"role": "user", "content": prompt_of(r["prompt_tokens"] or 16, r["i"])}],
            "max_tokens": out, "min_tokens": out, "ignore_eos": True, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = last = None; usage = {}; err = None
    try:
        async with client.stream("POST", url, json=body, timeout=900) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: {"):
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                if any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d.get("choices", [])):
                    now = time.time(); first = first or now; last = now
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    n = usage.get("completion_tokens") or 0
    p = usage.get("prompt_tokens") or r["prompt_tokens"] or 0
    ttft = first - t0 if first else None
    tpot = (last - first) / (n - 1) if first and n > 1 else 0.0
    ok = err is None and ttft is not None and ttft <= 0.5 + 0.25 * p / 1000 and (n <= 1 or tpot <= 0.050)
    res.append({**r, "ttft": ttft, "tpot_ms": tpot * 1000, "p": p, "n": n, "ok": ok, "err": err})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="mango")
    ap.add_argument("--only-graded", action="store_true", help="skip probe requests (max_tokens != 8192)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = sorted((json.loads(l) for l in open(a.log)), key=lambda r: r["t0"])
    R = [r for r in R if r.get("stream") and r.get("out_tokens")]
    t00 = R[0]["t0"]
    reqs = [{"i": i, "off": r["t0"] - t00, "prompt_tokens": r["prompt_tokens"], "out_tokens": r["out_tokens"],
             "max_tokens": r["max_tokens"], "real_ok": None} for i, r in enumerate(R)]
    if a.only_graded:
        reqs = [r for r in reqs if r["max_tokens"] == 8192]
        base = reqs[0]["off"]
        for r in reqs:
            r["off"] -= base
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    res = []
    t_start = time.time() + 1
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000)) as client:
        await asyncio.gather(*(one(client, a.url, a.model, r, t_start, res) for r in reqs))
    wall = time.time() - t_start
    joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000
    g = [r for r in res if r["max_tokens"] == 8192]
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None
    def fails(rs):
        f = {"ttft": 0, "tpot": 0, "err": 0}
        for r in rs:
            if r["ok"]: continue
            if r["err"] or r["ttft"] is None: f["err"] += 1
            elif r["ttft"] > 0.5 + 0.25 * r["p"] / 1000: f["ttft"] += 1
            else: f["tpot"] += 1
        return f
    summ = {"log": a.log, "n": len(res), "pass": sum(r["ok"] for r in res), "graded_n": len(g),
            "graded_pass": sum(r["ok"] for r in g), "graded_fails": fails(g),
            "p95_ttft_ratio": q([r["ttft"] / (0.5 + 0.25 * r["p"] / 1000) for r in g if r["ttft"]], .95),
            "p95_tpot_ms": q([r["tpot_ms"] for r in g if r["n"] > 1], .95), "wall_s": round(wall, 1),
            "joules": round(joules), "J_per_graded_pass": round(joules / max(1, sum(r["ok"] for r in g)), 1)}
    print(json.dumps(summ, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summ, "requests": res}, f)


if __name__ == "__main__":
    asyncio.run(main())
