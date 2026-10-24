"""Closed-loop, per-item replay of a real grader run (successor of replay_load.py).

Episodes are rebuilt from a metadata reqlog with item_lib. Turn 1 of each item is sent at its real offset; turn k+1 is sent
when turn k has finished + the real client gap (t0[k+1]-(t0[k]+e2e[k])) with the real prompt_tokens growth. Synthetic text,
but turn k+1's prompt starts with turn k's prompt text exactly (prefix caching). Forced outputs (min=max=out_tokens, ignore_eos,
thinking off). Probe-phase requests (max_tokens != 8192) are replayed open-loop unless --no-probes.
Scoring (score_items) follows the grader rule: an item passes only if all its turns pass; p95 over items (TTFT ratio: worst turn; TPOT: mean over turns).

  replay_v2.py LOG --dry-run                    # print schedule, no requests
  replay_v2.py --score-real LOG [LOG...]        # offline: score real logs with the same scorer (no requests)
  replay_v2.py LOG --out res.json               # replay against the server (needs GPU server up)
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import item_lib  # noqa: E402

ETA_BASE = 0.00163  # items per joule, baseline (briefing)
WORDS = "alpha beta gamma delta river mountain data energy green model".split()


def slo_ttft(p):
    return 0.5 + 0.25 * p / 1000


def build_schedule(log, probes=True):
    """-> (episodes, probes). episode: {id, kind, ts, turns:[{prompt_tokens,out_tokens,gap,ts,...}]}; gap = client gap before turn."""
    R = item_lib.load(log)
    # drop non-streaming pings (health checks, some ~1000 s before the run) and rebase offsets to the first real request
    R = [r for r in R if r.get('stream') or r['max_tokens'] == 8192]
    base = R[0]['ts']
    for r in R:
        r['ts'] -= base; r['end'] -= base
    its = item_lib.items(R)
    eps = []
    for n, it in enumerate(its):
        turns = []
        for k, r in enumerate(it['turns']):
            gap = 0.0 if k == 0 else max(0.0, r['ts'] - it['turns'][k - 1]['end'])
            turns.append({'prompt_tokens': r['prompt_tokens'], 'out_tokens': max(int(r['out_tokens'] or 1), 1), 'gap': gap,
                          'ts': r['ts'], 'real_ok': r['ok'], 'real_ttft': r['ttft'], 'real_tpot_ms': r['tpot_ms']})
        eps.append({'id': n, 'kind': it['kind'], 'ts': it['ts'], 'turns': turns})
    pr = []
    if probes:
        pr = [{'id': 10000 + i, 'ts': r['ts'], 'prompt_tokens': r['prompt_tokens'], 'out_tokens': max(int(r['out_tokens'] or 1), 1)}
              for i, r in enumerate(R) if r['probe'] and r.get('stream') and r.get('out_tokens')]
    return eps, pr


def peak_window(starts, width=60.0):
    """Densest `width` s window of arrival times -> (lo, hi)."""
    s = sorted(starts); best = (0, s[0] if s else 0.0); j = 0
    for i in range(len(s)):
        while s[i] - s[j] >= width:
            j += 1
        if i - j + 1 > best[0]:
            best = (i - j + 1, s[j])
    return best[1], best[1] + width


def p95(xs):
    return sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else None


def score_items(items, joules=None):
    """items: [{ts, turns:[{ts, prompt_tokens, out_tokens, ttft, tpot_ms, ok}]}] (ts = first-turn offset, turn ts = send offset).
    ttft None / ok False = failed. Grader rule: item passes iff all turns ok."""
    for it in items:
        it['pass'] = all(t['ok'] for t in it['turns'])
        it['ttft_ratio'] = max((t['ttft'] / slo_ttft(t['prompt_tokens']) if t['ttft'] is not None else 1e9) for t in it['turns'])
        tp = [t['tpot_ms'] for t in it['turns'] if t['out_tokens'] > 1 and t['tpot_ms'] is not None]
        it['tpot'] = sum(tp) / len(tp) if tp else 0.0  # item TPOT = mean over its turns (matches grader p95; worst-turn does not)
        it['tpot_max'] = max(tp, default=0.0)
    lo, hi = peak_window([it['ts'] for it in items])
    pk = [it for it in items if lo <= it['ts'] < hi]
    reqs = [t for it in items for t in it['turns']]
    pkreq = [t for t in reqs if lo <= t['ts'] < hi]
    n_pass = sum(it['pass'] for it in items)
    rt = [t['ttft'] / slo_ttft(t['prompt_tokens']) for t in reqs if t['ttft'] is not None]
    rp = [t['tpot_ms'] for t in reqs if t['out_tokens'] > 1 and t['tpot_ms'] is not None]
    s = {'items': len(items), 'items_pass': n_pass, 'requests': len(reqs), 'req_pass': sum(t['ok'] for t in reqs),
         'peak_window_s': [round(lo, 1), round(hi, 1)], 'peak_items': len(pk), 'peak_items_pass': sum(it['pass'] for it in pk),
         'sT': sum(it['pass'] for it in pk) / max(1, len(pk)),
         'sT_req': sum(t['ok'] for t in pkreq) / max(1, len(pkreq)),
         'p95_ttft_ratio_item': p95([it['ttft_ratio'] for it in items]), 'p95_tpot_ms_item': p95([it['tpot'] for it in items]),
         'p95_tpot_ms_item_worstturn': p95([it['tpot_max'] for it in items]),
         'p95_ttft_ratio_req': p95(rt), 'p95_tpot_ms_req': p95(rp),
         'p95_ttft_s_item': p95([max((t['ttft'] if t['ttft'] is not None else 1e9) for t in it['turns']) for it in items]),
         'p95_ttft_s_req': p95([t['ttft'] for t in reqs if t['ttft'] is not None])}
    sl_t = min(1.0, 1 / max(s['p95_ttft_ratio_item'], 1e-9)); sl_p = min(1.0, 50 / max(s['p95_tpot_ms_item'], 1e-9))
    s['sL'] = math.sqrt(sl_t * sl_p)
    if joules:
        eta = n_pass / joules
        s['joules'] = round(joules); s['J_per_pass_item'] = round(joules / max(1, n_pass), 1); s['eta'] = eta
        s['sE'] = min(1.0, max(0.0, math.log(max(eta, 1e-12) / ETA_BASE) / math.log(3)))
        s['S_est'] = round(100 * 1.0 * s['sE'] ** 0.6 * s['sT'] ** 0.2 * s['sL'] ** 0.2, 2)
    return s


def real_items(log):
    R = item_lib.load(log)
    out = []
    for it in item_lib.items(R):
        out.append({'ts': it['ts'], 'turns': [{'ts': r['ts'], 'prompt_tokens': r['prompt_tokens'], 'out_tokens': r['out_tokens'],
                                               'ttft': r['ttft'], 'tpot_ms': r['tpot_ms'], 'ok': r['ok']} for r in it['turns']]})
    return out


# ---------------------------------------------------------------- live replay
def text_of(n, seed):
    """Deterministic word stream per seed; prompt of n tokens = first n-18 words (so longer turns extend shorter ones exactly)."""
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(max(n - 18, 1)))


def make_prompt(words, n):
    return " ".join(words[:max(n - 18, 1)]) + "\nContinue the list."


def chat_history(words, turns, k):
    """--chat-history: turn k = turn k-1's messages + assistant + user, so turn k's prompt extends turn k-1's prompt
    up to its generation prompt (LCP = P_{k-1} - 4 tokens, real grader tool turns: P_{k-1} - 1). The default mode
    rewrites the user text each turn (LCP = P_{k-1} - 14), which understates sub-block prefix-cache hits."""
    n0 = max(turns[0]['prompt_tokens'] - 18, 1)
    msgs = [{"role": "user", "content": " ".join(words[:n0])}]
    pos = n0
    for j in range(1, k + 1):
        grow = max(turns[j]['prompt_tokens'] - turns[j - 1]['prompt_tokens'] - 10, 1)
        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": " ".join(words[pos:pos + grow])}]
        pos += grow
    return msgs


async def send(client, url, model, prompt, out, t_ref):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "max_tokens": out, "min_tokens": out,
            "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = last = None; usage = {}; err = None
    try:
        async with client.stream("POST", url, json=body, timeout=900) as resp:
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}"
            async for line in resp.aiter_lines():
                if not line.startswith("data: {"):
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                if any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning")
                       or (c.get("delta") or {}).get("reasoning_content") for c in d.get("choices", [])):
                    now = time.time(); first = first or now; last = now
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    n = usage.get("completion_tokens") or 0
    ttft = first - t0 if first else None
    tpot = (last - first) / (n - 1) * 1000 if first and n > 1 else 0.0
    return {"sent": t0 - t_ref, "end": time.time() - t_ref, "ttft": ttft, "tpot_ms": tpot, "n": n,
            "p": usage.get("prompt_tokens") or 0, "err": err}


async def run_episode(client, url, model, ep, t_ref, scale, history=False):
    await asyncio.sleep(max(0.0, t_ref + ep['ts'] * scale - time.time()))
    words = text_of(sum(t['prompt_tokens'] for t in ep['turns']), 7919 * ep['id'] + 1).split(" ")
    res = []
    for k, t in enumerate(ep['turns']):
        if k:
            await asyncio.sleep(t['gap'])
        prompt = chat_history(words, ep['turns'], k) if history else make_prompt(words, t['prompt_tokens'])
        r = await send(client, url, model, prompt, t['out_tokens'], t_ref)
        ok = r['err'] is None and r['ttft'] is not None and r['ttft'] <= slo_ttft(t['prompt_tokens']) \
            and (t['out_tokens'] <= 1 or r['tpot_ms'] <= 50)
        res.append({**r, "ts": r['sent'], "prompt_tokens": t['prompt_tokens'], "out_tokens": t['out_tokens'], "ok": ok, "turn": k})
        if r['err']:
            break
    # unsent turns (after error) count as failed
    for k in range(len(res), len(ep['turns'])):
        t = ep['turns'][k]
        res.append({"ts": t['ts'], "prompt_tokens": t['prompt_tokens'], "out_tokens": t['out_tokens'], "ttft": None,
                    "tpot_ms": None, "ok": False, "turn": k, "err": "not sent"})
    return {"id": ep["id"], "kind": ep["kind"], "ts": ep["ts"] * scale, "turns": res}


async def run_probe(client, url, model, p, t_ref, scale):
    await asyncio.sleep(max(0.0, t_ref + p['ts'] * scale - time.time()))
    await send(client, url, model, make_prompt(text_of(p['prompt_tokens'] or 16, p['id']).split(" "), p['prompt_tokens'] or 16),
               p['out_tokens'], t_ref)


async def live(a, eps, pr):
    import httpx
    import pynvml
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    t_ref = time.time() + 1
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=2000)) as client:
        done = await asyncio.gather(*(run_episode(client, a.url, a.model, e, t_ref, a.scale, a.chat_history) for e in eps),
                                    *(run_probe(client, a.url, a.model, p, t_ref, a.scale) for p in pr))
    wall = time.time() - t_ref
    joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000
    done = [d for d in done if d]
    summ = score_items(done, joules)
    summ.update({"log": a.log, "wall_s": round(wall, 1), "probes": len(pr), "scale": a.scale})
    print(json.dumps(summ, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summ, "items": done}, f)


def dry_run(a, eps, pr):
    lo, hi = peak_window([e['ts'] for e in eps])
    print(f"# {a.log}: {len(eps)} items ({sum(e['kind']=='tool' for e in eps)} tool episodes), "
          f"{sum(len(e['turns']) for e in eps)} graded requests, {len(pr)} probe requests; peak window {lo:.1f}-{hi:.1f}s")
    print("# item kind turns | first-turn ts | per-turn: prompt_tokens/out_tokens (+gap s)")
    for e in eps:
        tl = " ".join(f"{t['prompt_tokens']}/{t['out_tokens']}" + (f"(+{t['gap']:.3f})" if k else "") for k, t in enumerate(e['turns']))
        print(f"{e['id']:3d} {e['kind']:5s} {len(e['turns']):2d} | t={e['ts']:8.2f}{'*' if lo <= e['ts'] < hi else ' '}| {tl}")
    gaps = sorted(t['gap'] for e in eps for t in e['turns'][1:])
    if gaps:
        print(f"# inter-turn gaps: n={len(gaps)} min={gaps[0]:.3f} median={gaps[len(gaps)//2]:.3f} max={gaps[-1]:.3f}")
    print(f"# probes: first {pr[0]['ts']:.1f}s .. last {pr[-1]['ts']:.1f}s" if pr else "# no probes")


GRADER = {  # run -> (items passed, p95 TTFT s, p95 TPOT ms) from the real grader
    'f5': (265, 6.98, 81), 'f6': (256, 18.45, 39), 'f7': (262, 7.76, 45), 'f8': (251, 16.99, 46)}


def score_real(logs):
    print("| log | items | passed (scorer) | grader | p95 TTFT item s | p95 TTFT req s | grader | p95 TPOT item-mean ms | p95 TPOT item-worst ms | p95 TPOT req ms | grader | sT | sL |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for log in logs:
        tag = next((k for k in GRADER if os.path.basename(log).startswith(f"reqlog_{k}_")), None)
        g = GRADER.get(tag, (None, None, None))
        s = score_items(real_items(log))
        print(f"| {tag or os.path.basename(log)} | {s['items']} | {s['items_pass']} | {g[0]} | {s['p95_ttft_s_item']:.2f} | "
              f"{s['p95_ttft_s_req']:.2f} | {g[1]} | {s['p95_tpot_ms_item']:.1f} | {s['p95_tpot_ms_item_worstturn']:.1f} | {s['p95_tpot_ms_req']:.1f} | {g[2]} | "
              f"{s['sT']:.3f} | {s['sL']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="*")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="mango")
    ap.add_argument("--no-probes", action="store_true", help="skip the probe phase (graded items only, offsets re-based to first graded item)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply item start offsets (e.g. 0.5 = twice the arrival rate)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-real", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--chat-history", action="store_true", default=os.environ.get("GM_REPLAY_CHAT") == "1",
                    help="multi-turn items resend the chat history (exact prompt prefixes, for prefix-cache A/B; "
                         "env GM_REPLAY_CHAT=1)")
    a = ap.parse_args()
    if a.score_real:
        return score_real(a.log)
    a.log = a.log[0]
    eps, pr = build_schedule(a.log, probes=not a.no_probes)
    if a.no_probes:
        base = eps[0]['ts']
        for e in eps:
            e['ts'] -= base
    if a.dry_run:
        return dry_run(a, eps, pr)
    asyncio.run(live(a, eps, pr))


if __name__ == "__main__":
    main()
