"""Reconstruct probe/graded items from a metadata-only grader request log (see docs/METHOD.md)."""
import json
RUNS = {'f2': 'results/reqlog_f2_think_b2048.jsonl',
        'f3': 'results/reqlog_f3_prio_think05_b8192.jsonl',
        'v3': 'results/reqlog_v3_fcfs8192.jsonl'}

def load(path):
    R = [json.loads(l) for l in open(path)]
    R.sort(key=lambda r: r['t0'])
    t = R[0]['t0']
    for r in R:
        if r['ttft'] is None: r['ttft'] = r['e2e']; r['tpot_ms'] = 0.0
        r['ts'] = r['t0'] - t
        r['end'] = r['ts'] + r['e2e']
        r['probe'] = r['max_tokens'] != 8192
        r['ttft_slo'] = 0.5 + 0.25 * r['prompt_tokens'] / 1000
        r['ok_ttft'] = r['ttft'] <= r['ttft_slo']
        r['ok_tpot'] = r['out_tokens'] <= 1 or r['tpot_ms'] <= 50
        r['ok_trunc'] = r['finish'] != 'length'
        r['ok'] = r['ok_ttft'] and r['ok_tpot'] and r['ok_trunc']
    return R

def items(R):
    """Graded items: list of dicts {kind, turns:[req...], t0}. Tool episodes via prefix-roles chaining."""
    G = [r for r in R if not r['probe']]
    used = set(); eps = []
    tools = [r for r in G if r['n_tools'] > 0]
    heads = [r for r in tools if r['roles'] == 'su']
    chain = {id(h): [h] for h in heads}
    last = {id(h): h for h in heads}
    for r in tools:
        if r['roles'] == 'su': continue
        best = None
        for h in heads:
            p = last[id(h)]
            rem = r['roles'][len(p['roles']):]
            if (p['n_tools'] == r['n_tools'] and r['roles'].startswith(p['roles'])
                    and len(rem) >= 2 and rem[0] == 'a' and set(rem[1:]) == {'t'}
                    and p['finish'] == 'tool_calls' and -0.1 <= r['ts'] - p['end'] <= 0.3
                    and p['prompt_tokens'] < r['prompt_tokens']):
                g = abs(r['ts'] - p['end'] - 0.03) + 1e-5 * (r['prompt_tokens'] - p['prompt_tokens'])
                if best is None or g < best[0]: best = (g, h)
        if best:
            chain[id(best[1])].append(r); last[id(best[1])] = r
        else:
            print('UNMATCHED', r['roles'], r['ts'])
    out = [{'kind': 'tool', 'turns': chain[id(h)], 'ts': h['ts']} for h in heads]
    out += [{'kind': 'plain', 'turns': [r], 'ts': r['ts']} for r in G if r['n_tools'] == 0]
    out.sort(key=lambda i: i['ts'])
    return out
