"""CPU-only differential test of slo_scheduler.SLOScheduler against vLLM 0.32's own Scheduler / AsyncScheduler.

No GPU, no model weights: a tiny fake llama config (written to a temp dir), real vLLM Scheduler classes, fake
Request objects and a fake engine loop that turns every scheduled request into sampled tokens (with random
spec-decode acceptance). The same load runs through the stock class and the SLO class; both must finish every
request with the right number of output tokens, and the SLO-specific checks must hold:
  * deadline order: a short urgent request is admitted ahead of an older long one,
  * late requests are held back while feasible ones wait, nothing is ever dropped,
  * doomed decodes are skipped when the measured TPOT is high and resume afterwards,
  * max_num_seqs / token-budget invariants (vLLM's own asserts) never fire.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts .venv-vllm/bin/python scripts/test_slo_scheduler.py
"""
import json
import os
import random
import sys
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GM_SCHED_LOG", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402,F401
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec  # noqa: E402
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402

import slo_scheduler  # noqa: E402,F401

TMP = tempfile.mkdtemp(prefix="slo_sched_test_")
with open(os.path.join(TMP, "config.json"), "w") as f:
    json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
               "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
               "num_hidden_layers": 2, "vocab_size": 1000, "max_position_embeddings": 40960,
               "torch_dtype": "float16", "eos_token_id": 2, "bos_token_id": 1}, f)

BLOCK = 16
MBLOCK = 64   # mamba/attention block in the hybrid case (multiple of the hash block)
init_none_hash(sha256)
HASHER = get_request_block_hasher(BLOCK, sha256)


def make(cls_name, async_sched, spec_k=0, max_seqs=16, budget=1024, thr=512, prefix=True, mamba=False):
    model = ModelConfig(model=TMP, skip_tokenizer_init=True, dtype="float16", max_model_len=40960)
    sc = SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=budget, max_model_len=40960,
                         long_prefill_token_threshold=thr, enable_chunked_prefill=True,
                         is_encoder_decoder=False, async_scheduling=async_sched)
    if cls_name == "slo":
        sc.scheduler_cls = "slo_scheduler.SLOScheduler"
    cc = CacheConfig(block_size=MBLOCK if mamba else BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
                     enable_prefix_caching=prefix)
    if mamba:
        cc.mamba_cache_mode = "align"
    cc.num_gpu_blocks = 20000
    spec = None
    if spec_k:
        spec = SpeculativeConfig(method="ngram_gpu" if async_sched else "ngram", num_speculative_tokens=spec_k,
                                 prompt_lookup_max=3,
                                 prompt_lookup_min=1)
    vc = VllmConfig(scheduler_config=sc, model_config=model, cache_config=cc, speculative_config=spec)
    vc.scheduler_config.async_scheduling = async_sched  # VllmConfig may have overridden it for ngram
    bs = MBLOCK if mamba else BLOCK
    groups = [KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=bs, num_kv_heads=1, head_size=8,
                                                              dtype=torch.float16))]
    if mamba:   # hybrid like Qwen3.5 GDN: attention + mamba-type state, align-mode prefix caching
        groups.append(KVCacheGroupSpec(["layer.1"], MambaSpec(
            block_size=bs, shapes=((4, 8),), dtypes=(torch.float16,), mamba_cache_mode="align",
            num_speculative_blocks=spec_k)))
    kv = KVCacheConfig(num_blocks=20000, kv_cache_tensors=[], kv_cache_groups=groups)
    cls = vc.scheduler_config.get_scheduler_cls()
    return cls(vllm_config=vc, kv_cache_config=kv, structured_output_manager=StructuredOutputManager(vc),
               block_size=bs, hash_block_size=BLOCK, log_stats=False)


def req(i, prompt_len, out_len, arrival):
    sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
    toks = [random.Random(i).randrange(3, 1000) for _ in range(prompt_len)]
    return Request(request_id=f"r{i}", prompt_token_ids=toks, sampling_params=sp, pooling_params=None,
                   arrival_time=arrival, block_hasher=HASHER)


def fake_output(sched, so, rnd, spec_k):
    """Sample tokens for every scheduled request: prefill-completing chunks and decodes produce 1 + accepted."""
    req_ids, out = [], []
    for rid, n in so.num_scheduled_tokens.items():
        r = sched.requests.get(rid)
        req_ids.append(rid)
        if r is None:
            out.append([])
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid, [])
        # after _update_after_schedule: still prefilling -> no sample
        if r.is_prefill_chunk:   # evaluated right after schedule(): chunk does not finish the prompt
            out.append([])
            continue
        acc = rnd.randint(0, len(spec)) if spec else 0
        out.append([rnd.randrange(3, 1000) for _ in range(1 + acc)])
    return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                             sampled_token_ids=out)


def run(sched, workload, async_sched, spec_k, seed=0, max_steps=20000, step_sleep=0.0, hooks=None):
    rnd = random.Random(seed)
    pending = sorted(workload, key=lambda x: x[0])   # (step_to_add, Request)
    inflight = []
    outputs = {}
    step = 0
    while step < max_steps:
        while pending and pending[0][0] <= step:
            sched.add_request(pending.pop(0)[1])
        if not pending and not sched.has_requests():
            break   # remaining in-flight steps are drained below
        so = sched.schedule()
        if hooks:
            hooks(sched, so, step)
        inflight.append((so, fake_output(sched, so, rnd, spec_k)))   # "GPU result" fixed at schedule time
        if async_sched and len(inflight) < 2 and (pending or sched.has_requests()):
            step += 1
            continue
        so, mro = inflight.pop(0)
        if step_sleep:
            time.sleep(step_sleep)
        eco = sched.update_from_output(so, mro)
        for v in eco.values():
            for o in v.outputs:
                outputs.setdefault(o.request_id, []).extend(o.new_token_ids)
        if spec_k and not async_sched:
            ids = [rid for rid in so.num_scheduled_tokens if rid in sched.requests]
            sched.update_draft_token_ids(DraftTokenIds(ids, [[rnd.randrange(3, 1000) for _ in range(spec_k)]
                                                             for _ in ids]))
        step += 1
    while inflight:   # drain async pipeline
        so, mro = inflight.pop(0)
        eco = sched.update_from_output(so, mro)
        for v in eco.values():
            for o in v.outputs:
                outputs.setdefault(o.request_id, []).extend(o.new_token_ids)
    return outputs, step


def workload(n=60, seed=1, now=None):
    rnd = random.Random(seed)
    now = now or time.time()
    w = []
    for i in range(n):
        p = rnd.choice([30, 200, 900, 3000, 7000])
        o = rnd.choice([5, 40, 200])
        # some requests arrived long ago -> already late (doomed) when they reach the scheduler
        arr = now - (rnd.random() * 20 if i % 4 == 0 else 0)
        w.append((i // 3, req(i, p, o, arr)))
    return w


def check_complete(outputs, w, label):
    for _, r in w:
        n = len(outputs.get(r.request_id, []))
        assert n == r.sampling_params.max_tokens, f"{label}: {r.request_id} got {n} tokens"


def test_differential():
    # force every SLO path on in the fast fake loop: doomed skipping at any TPOT, budget controller shrinking
    os.environ.update(GM_SCHED_DOOMED_TPOT="0", GM_SCHED_TPOT="0.00001", GM_SCHED_ADMIT="budget")
    try:
        for mamba in (False, True):
            for async_sched in (False, True):
                for spec_k in (0, 3):
                    for budget, thr in ((1024, 512), (2048, 0)):
                        for name in ("stock", "slo"):
                            _diff_case(name, mamba, async_sched, spec_k, budget, thr)
    finally:
        for k in ("GM_SCHED_DOOMED_TPOT", "GM_SCHED_TPOT", "GM_SCHED_ADMIT"):
            os.environ.pop(k, None)


def _diff_case(name, mamba, async_sched, spec_k, budget, thr):
    s = make(name, async_sched, spec_k, budget=budget, thr=thr, mamba=mamba)
    assert s.need_mamba_block_aligned_split == mamba
    seen = {"hidden_steps": 0, "late_held": 0, "min_budget": budget}

    def hook(sched, so, step):
        if name == "slo":
            seen["hidden_steps"] += sched._slo.stats["hidden"] > 0
            seen["min_budget"] = min(seen["min_budget"], int(sched._slo.budget))
            # invariants after restore
            assert len(sched.running) <= sched.max_num_running_reqs
            assert sched.max_num_scheduled_tokens == budget or spec_k  # restored (spec may lower base)
        assert so.total_num_scheduled_tokens <= budget
    if name == "slo":
        assert type(s).__name__ == ("SLOAsyncScheduler" if async_sched else "SLOScheduler"), type(s).__name__
    w = workload()
    out, steps = run(s, w, async_sched, spec_k, hooks=hook)
    check_complete(out, w, f"{name} mamba={mamba} async={async_sched} k={spec_k} b={budget}")
    extra = ""
    if name == "slo":
        seen["late_held"] = s._slo.stats["late_held"]
        assert seen["hidden_steps"] > 0 and seen["late_held"] > 0, seen
        extra = f" hidden_steps={seen['hidden_steps']} late_held_steps={seen['late_held']} " \
                f"min_budget={seen['min_budget']}"
    print(f"ok {name:5s} mamba_align={mamba!s:5s} async={async_sched!s:5s} spec_k={spec_k} "
          f"budget={budget}/{thr} steps={steps}{extra}")


def test_edf_order_and_late_hold():
    now = time.time()
    s = make("slo", False, budget=2048, thr=0, max_seqs=16)
    old_long = req(1, 6000, 5, now - 0.05)     # SLO 2.0 s, arrived 50 ms ago, deadline now+1.95
    urgent = req(2, 50, 5, now)                # SLO 0.51 s, deadline now+0.51
    doomed = req(3, 100, 5, now - 30)          # missed long ago -> late, goes behind
    for r in (doomed, old_long, urgent):
        s.add_request(r)
    so = s.schedule()
    order = [r.req_id for r in so.scheduled_new_reqs]
    assert order[0] == "r2", order                       # urgent first
    assert order[1] == "r1", order                       # then the feasible long one
    assert "r3" not in order, order                      # late held back while feasible ones waited
    assert len(s.waiting) == 1 and s.waiting[0].request_id == "r3"
    print("ok EDF order", order, "late held:", [r.request_id for r in s.waiting])
    # once nothing feasible waits, the late request is admitted (never dropped)
    rnd = random.Random(0)
    for k in range(10):
        s.update_from_output(so, fake_output(s, so, rnd, 0))
        so = s.schedule()
        if "r3" in [r.req_id for r in so.scheduled_new_reqs]:
            break
    else:
        raise AssertionError("late request never admitted")
    print(f"ok late request admitted {k + 1} steps later, once the budget was free (not dropped)")


def test_doomed_skip():
    now = time.time()
    s = make("slo", False, budget=2048, thr=0, max_seqs=16)
    pol = s._slo
    good = [req(10 + i, 40, 400, now) for i in range(4)]
    bad = [req(20 + i, 40, 400, now - 60) for i in range(4)]   # TTFT long gone
    for r in good + bad:
        s.add_request(r)
    rnd = random.Random(0)
    for _ in range(5):   # good ones first (late ones held back one step), then the rest
        so = s.schedule()
        s.update_from_output(so, fake_output(s, so, rnd, 0))
        if all(r.num_output_tokens > 0 for r in good + bad):
            break
    assert all(r.request_id in pol.first_tok for r in good + bad)
    pol.tpot = 0.080    # pretend the measured step TPOT is high
    so = s.schedule()
    sched_ids = set(so.num_scheduled_tokens)
    assert all(r.request_id in sched_ids for r in good), sched_ids
    assert not any(r.request_id in sched_ids for r in bad), sched_ids
    assert len(s.running) == 8                                 # hidden ones restored to running
    assert [r.request_id for r in s.running[-4:]] == [r.request_id for r in bad]   # restored at the end
    s.update_from_output(so, fake_output(s, so, rnd, 0))
    pol.tpot = 0.010    # load gone -> doomed decodes resume
    pol.observe = lambda *a, **k: None
    so = s.schedule()
    assert all(r.request_id in so.num_scheduled_tokens for r in bad)
    print("ok doomed decodes skipped under high TPOT and resumed after")


def test_admission_controllers_run():
    # tiny TPOT target -> the controllers clamp down; everything must still complete
    for mode in ("budget", "cap"):
        os.environ.update(GM_SCHED_ADMIT=mode, GM_SCHED_TPOT="0.00001", GM_SCHED_MIN_CAP="4")
        try:
            s = make("slo", True, 3, budget=1024, thr=512)
            seen = {"cap": 10 ** 9, "budget": 10 ** 9, "running": 0}

            def hook(sched, so, step):
                seen["cap"] = min(seen["cap"], int(sched._slo.cap))
                seen["budget"] = min(seen["budget"], int(sched._slo.budget))
                seen["running"] = max(seen["running"], len(sched.running))
            w = workload(40, seed=3)
            out, _ = run(s, w, True, 3, hooks=hook)
            check_complete(out, w, f"admit={mode}")
            if mode == "cap":
                assert seen["cap"] == 4, seen
            else:
                assert seen["budget"] == 512, seen
            print(f"ok admit={mode} clamps (min cap={seen['cap']}, min budget={seen['budget']}) and completes")
        finally:
            for k in ("GM_SCHED_ADMIT", "GM_SCHED_TPOT", "GM_SCHED_MIN_CAP"):
                os.environ.pop(k, None)


def test_priority_policy_replaced():
    model = None  # noqa: F841
    s = make("slo", False)
    s2 = None
    os.environ["GM_SCHED_EDF"] = "1"
    try:
        sc_model = ModelConfig(model=TMP, skip_tokenizer_init=True, dtype="float16", max_model_len=40960)
        sc = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=1024, max_model_len=40960,
                             enable_chunked_prefill=True, is_encoder_decoder=False, policy="priority",
                             scheduler_cls="slo_scheduler.SLOScheduler", async_scheduling=False)
        cc = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto")
        cc.num_gpu_blocks = 2000
        vc = VllmConfig(scheduler_config=sc, model_config=sc_model, cache_config=cc)
        kv = KVCacheConfig(num_blocks=2000, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8,
                                                            dtype=torch.float16))])
        s2 = vc.scheduler_config.get_scheduler_cls()(vllm_config=vc, kv_cache_config=kv,
                                                     structured_output_manager=StructuredOutputManager(vc),
                                                     block_size=BLOCK)
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        assert isinstance(s2.waiting, FCFSRequestQueue)
        w = workload(20, seed=5)
        out, _ = run(s2, w, False, 0)
        check_complete(out, w, "priority policy")
        print("ok --scheduling-policy priority: waiting queue replaced by deque, completes")
    finally:
        del s, s2


HDR = [7, 8, 9, 10]   # stand-in for the tool system header token ids
ASST = (12, 13)       # stand-in for "<|im_start|>assistant" (the chat template's generation prompt ends a prompt with it)


def req_toks(rid, toks, out_len, arrival=None):
    sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
    return Request(request_id=rid, prompt_token_ids=list(toks), sampling_params=sp, pooling_params=None,
                   arrival_time=arrival or time.time(), block_hasher=HASHER)


def _prefill_chunks(s, r, rnd):
    """Add r, run steps until its first token; -> (cache hit tokens, list of prefill chunk sizes)."""
    s.add_request(r)
    hit, chunks = None, []
    for _ in range(50):
        so = s.schedule()
        for nr in so.scheduled_new_reqs:
            if nr.req_id == r.request_id:
                hit = nr.num_computed_tokens
        if r.request_id in so.num_scheduled_tokens:
            chunks.append(so.num_scheduled_tokens[r.request_id])
        s.update_from_output(so, fake_output(s, so, rnd, 0))
        if r.num_output_tokens > 0:
            return hit, chunks
    raise AssertionError("prefill did not finish")


def _finish(s, rnd):
    for _ in range(200):
        if not s.has_requests():
            return
        so = s.schedule()
        s.update_from_output(so, fake_output(s, so, rnd, 0))
    raise AssertionError("requests did not finish")


def test_tail_checkpoint():
    """GM_SCHED_TAIL_CKPT: which prompts keep vLLM's extra prefill stop at the prompt-tail hash boundary.

    Hybrid align mode, mamba/attention block 64, hash (prefix_match_unit) 16 -> fine-grained hits are on and vLLM
    ends a prefill chunk at the last 16-aligned position before the prompt end so the follow-up turn can reuse the
    state there. Default ("" = vLLM) keeps that for every prompt; "tools" only for prompts with the tool header,
    "followups" only for prompts with an earlier assistant turn, "<N>" by length, "off" never."""
    rnd = random.Random(0)
    base = [random.Random(99).randrange(20, 1000) for _ in range(200)]
    gen = list(ASST) + [14]              # generation prompt "<|im_start|>assistant\n"
    tool = HDR + base[:93] + gen         # 100-token tool prompt (turn 1): block state at 64, tail boundary at 96
    plain = base[:97] + gen              # 100-token plain prompt
    env = {"GM_SCHED_TAIL_PREFIX_IDS": ",".join(map(str, HDR)), "GM_SCHED_TAIL_ASSISTANT_IDS": ",".join(map(str, ASST)),
           "GM_SCHED_EDF": "0", "GM_SCHED_LATE": "0", "GM_SCHED_DOOMED": "0", "GM_SCHED_ADMIT": "0"}
    expect = {  # mode -> (plain chunks, tool turn-1 chunks, tool turn-2 hit, turn-2 chunks)
        "": ([64, 32, 4], [64, 32, 4], 96, [32, 16, 4]),
        "tools": ([64, 36], [64, 32, 4], 96, [32, 16, 4]),
        "followups": ([64, 36], [64, 36], 64, [64, 16, 4]),   # turn 1 not split -> turn 2 hits the 64-token block
        "off": ([64, 36], [64, 36], 64, [64, 20]),
        "90": ([64, 32, 4], [64, 32, 4], 96, [32, 16, 4]),
        "101": ([64, 36], [64, 36], 64, [64, 16, 4]),
    }
    for mode, (exp_plain, exp_tool, exp_hit2, exp_ch2) in expect.items():
        os.environ.update(env, GM_SCHED_TAIL_CKPT=mode)
        try:
            s = make("slo", False, budget=1024, thr=512, mamba=True)
        finally:
            for k in list(env) + ["GM_SCHED_TAIL_CKPT"]:
                os.environ.pop(k, None)
        assert s.mamba_partial_cache_hit, "test needs fine-grained (hash < block) hits"
        hit, ch = _prefill_chunks(s, req_toks(f"p-{mode}", [5] + plain[1:], 3), rnd)
        assert ch == exp_plain, (mode, "plain", ch)
        _finish(s, rnd)
        hit1, ch1 = _prefill_chunks(s, req_toks(f"t1-{mode}", tool, 3), rnd)
        assert hit1 == 0 and ch1 == exp_tool, (mode, "tool turn 1", hit1, ch1)
        _finish(s, rnd)
        # turn 2 = turn 1 minus its last token + assistant/tool-response tokens (what the chat template renders)
        turn2 = tool[:-1] + base[100:146] + gen     # 148 tokens: block state at 128, tail boundary at 144
        hit2, ch2 = _prefill_chunks(s, req_toks(f"t2-{mode}", turn2, 3), rnd)
        assert hit2 == exp_hit2 and ch2 == exp_ch2, (mode, "tool turn 2", hit2, ch2)
        _finish(s, rnd)
        assert sum(ch2) + hit2 == len(turn2)
        turn3 = turn2[:-1] + base[146:170] + gen
        hit3, ch3 = _prefill_chunks(s, req_toks(f"t3-{mode}", turn3, 3), rnd)
        assert hit3 == (128 if mode == "off" else 144), (mode, "tool turn 3", hit3, ch3)
        _finish(s, rnd)
        print(f"ok tail checkpoint mode={mode or 'vllm-default':9s} plain chunks={ch} tool turn-1 chunks={ch1} "
              f"turn-2 hit={hit2}/{len(turn2)} chunks={ch2} turn-3 hit={hit3}/{len(turn3)}")
    # full mixed workload under a non-default mode: completes, budget invariants hold
    os.environ.update(env, GM_SCHED_TAIL_CKPT="tools")
    try:
        for async_sched, spec_k in ((False, 0), (True, 3)):
            s = make("slo", async_sched, spec_k, budget=1024, thr=512, mamba=True)
            w = workload(40, seed=7)
            for j, (st, r) in enumerate(w[::2]):    # half of the requests carry the tool header
                w[2 * j] = (st, req_toks(r.request_id, HDR + list(r.prompt_token_ids[len(HDR):]),
                                         r.sampling_params.max_tokens, r.arrival_time))
            out, _ = run(s, w, async_sched, spec_k)
            check_complete(out, w, f"tail tools async={async_sched}")
        print("ok tail checkpoint mode=tools: mixed workload completes (sync k=0, async k=3)")
    finally:
        for k in list(env) + ["GM_SCHED_TAIL_CKPT"]:
            os.environ.pop(k, None)


def det_output(sched, so, spec_k):
    """Like fake_output, but every sampled token and every acceptance count is a pure function of (request,
    output position): any correct schedule (chunking, batching) yields the same token sequence per request.
    Sync scheduling only (output positions are exact at schedule time)."""
    req_ids, out = [], []
    for rid, n in so.num_scheduled_tokens.items():
        r = sched.requests.get(rid)
        req_ids.append(rid)
        if r is None or r.is_prefill_chunk:
            out.append([])
            continue
        pos = r.num_output_tokens
        spec = so.scheduled_spec_decode_tokens.get(rid, [])
        acc = random.Random(f"{rid}/{pos}/acc").randint(0, len(spec)) if spec else 0
        out.append([random.Random(f"{rid}/{pos + j}").randrange(3, 1000) for j in range(1 + acc)])
    return ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                             sampled_token_ids=out)


def run_det(sched, workload, spec_k, max_steps=20000):
    """Sync engine loop with det_output; -> (outputs, {request_id: [chunk sizes while prefilling]},
    {request_id: step of the first token})."""
    pending = sorted(workload, key=lambda x: x[0])
    outputs, chunks, first = {}, {}, {}
    rnd = random.Random(0)
    step = 0
    while step < max_steps:
        while pending and pending[0][0] <= step:
            sched.add_request(pending.pop(0)[1])
        if not pending and not sched.has_requests():
            break
        so = sched.schedule()
        for rid, n in so.num_scheduled_tokens.items():
            r = sched.requests[rid]
            if r.num_computed_tokens - n < r.num_prompt_tokens:   # chunk started inside the prompt
                chunks.setdefault(rid, []).append(n)
        eco = sched.update_from_output(so, det_output(sched, so, spec_k))
        for v in eco.values():
            for o in v.outputs:
                if o.new_token_ids and o.request_id not in first:
                    first[o.request_id] = step
                outputs.setdefault(o.request_id, []).extend(o.new_token_ids)
        if spec_k:
            ids = [rid for rid in so.num_scheduled_tokens if rid in sched.requests]
            sched.update_draft_token_ids(DraftTokenIds(ids, [[rnd.randrange(3, 1000) for _ in range(spec_k)]
                                                             for _ in ids]))
        step += 1
    return outputs, chunks, first


def _chunk_ends_ok(chunks, prompt_len, block, tail_positions):
    """Every non-final prefill chunk must end on a block boundary or on one of vLLM's mandatory stops.
    The first chunk starts after the prefix-cache hit (prompt_len - computed chunks, hash-block aligned)."""
    end = prompt_len - sum(chunks)
    if end < 0 or end % BLOCK:
        return False
    for n in chunks[:-1]:
        end += n
        if end % block and end not in tail_positions:
            return False
    return end + chunks[-1] == prompt_len


CHUNK_ENV = {"GM_SCHED_CHUNK": "1", "GM_SCHED_STEP_PRIOR": "0.3", "GM_SCHED_EWMA": "0",   # frozen step/rate/TPOT
             "GM_SCHED_ADMIT": "0", "GM_SCHED_DOOMED": "0"}


def test_chunk():
    """GM_SCHED_CHUNK: urgent long prefills get larger, block-aligned chunks; nothing else changes."""
    from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position as ckpt

    def load():
        now = time.time()   # per run: deadlines are wall-clock, scheduler construction takes a while
        # one long urgent prompt (SLO 0.5 + 0.25*3 = 1.25 s) + short prompts arriving while it prefills
        w = [(0, req_toks("long", [random.Random(f"L{j}").randrange(20, 1000) for j in range(3000)], 20, now))]
        for i in range(12):
            w.append((i, req_toks(f"s{i}", [random.Random(f"s{i}/{j}").randrange(20, 1000) for j in range(40 + 9 * i)],
                                  30, now + 0.001 * i)))
        return w

    res = {}
    for chunk in ("0", "1"):
        for spec_k in (0, 3):
            os.environ.update(CHUNK_ENV, GM_SCHED_CHUNK=chunk, GM_SCHED_CHUNK_MAX="512")
            try:
                s = make("slo", False, spec_k, budget=1024, thr=100, mamba=True)   # thr 100 -> 64-token chunks
            finally:
                for k in list(CHUNK_ENV) + ["GM_SCHED_CHUNK_MAX"]:
                    os.environ.pop(k, None)
            w = load()
            out, ch, first = run_det(s, w, spec_k)
            check_complete(out, w, f"chunk={chunk} k={spec_k}")
            assert s.scheduler_config.long_prefill_token_threshold == 100      # restored after every step
            res[chunk, spec_k] = (out, ch, first, s._slo.stats["boost_steps"])
            tails = {ckpt(3000, BLOCK, s.use_eagle_block_drop), ckpt(3000, MBLOCK, s.use_eagle_block_drop)}
            assert _chunk_ends_ok(ch["long"], 3000, MBLOCK, tails), (chunk, ch["long"])
    for spec_k in (0, 3):
        out0, ch0, f0, b0 = res["0", spec_k]
        out1, ch1, f1, b1 = res["1", spec_k]
        assert b0 == 0 and b1 > 0, (b0, b1)
        assert out0 == out1, "outputs differ with GM_SCHED_CHUNK=1"
        assert set(ch0["long"][:10]) == {64}, ch0["long"]                   # static: 100 -> 64 after alignment
        assert max(ch1["long"]) <= 512 and max(ch1["long"]) > 64, ch1["long"]
        assert f1["long"] < f0["long"], (f1["long"], f0["long"])           # first token earlier
        for i in range(12):   # other prompts: never a later first token (one-chunk prompts: identical chunks)
            rid = f"s{i}"
            assert f1[rid] <= f0[rid], (rid, ch0[rid], ch1[rid], f0[rid], f1[rid])
            assert len(ch0[rid]) > 1 or ch1[rid] == ch0[rid], (rid, ch0[rid], ch1[rid])
        print(f"ok chunk spec_k={spec_k}: long chunks {ch0['long'][:3]}.. ({len(ch0['long'])} steps, 1st token step "
              f"{f0['long']}) -> {ch1['long'][:3]}.. ({len(ch1['long'])} steps, step {f1['long']}); "
              f"shorts never later, outputs identical")

    # a long prompt that already missed its deadline gets no boost
    os.environ.update(CHUNK_ENV)
    try:
        s = make("slo", False, budget=1024, thr=100, mamba=True)
    finally:
        for k in CHUNK_ENV:
            os.environ.pop(k, None)
    now = time.time()
    w = [(0, req_toks("late", [random.Random(f"X{j}").randrange(20, 1000) for j in range(3000)], 5, now - 30)),
         (0, req_toks("other", [random.Random(f"Y{j}").randrange(20, 1000) for j in range(30)], 50, now))]
    out, ch, _ = run_det(s, w, 0)
    check_complete(out, w, "late long")
    assert set(ch["late"][:5]) == {64} and s._slo.stats["boost_steps"] == 0, ch["late"]
    print("ok chunk: a long prompt past its deadline keeps the static chunk")

    # full mixed workload, every scheduler flavour: completes, vLLM asserts hold, chunk ends stay aligned
    for mamba in (False, True):
        for async_sched in (False, True):
            for spec_k in (0, 3):
                os.environ.update(CHUNK_ENV)
                try:
                    s = make("slo", async_sched, spec_k, budget=2048, thr=512 if not mamba else 100, mamba=mamba)
                finally:
                    for k in CHUNK_ENV:
                        os.environ.pop(k, None)
                seen = {}

                def hook(sched, so, step):
                    assert sched.scheduler_config.long_prefill_token_threshold == (100 if mamba else 512)
                    assert so.total_num_scheduled_tokens <= 2048
                    for rid, n in so.num_scheduled_tokens.items():
                        r = sched.requests[rid]
                        if r.num_computed_tokens - n < r.num_prompt_tokens:   # chunk started inside the prompt
                            seen.setdefault(rid, []).append(n)
                w = workload(40, seed=11)
                out, _ = run(s, w, async_sched, spec_k, hooks=hook)
                check_complete(out, w, f"chunk mamba={mamba} async={async_sched} k={spec_k}")
                assert s._slo.stats["boost_steps"] > 0
                bs = MBLOCK if mamba else 1
                for _, r in w:
                    P = r.num_prompt_tokens
                    tails = {ckpt(P, BLOCK, s.use_eagle_block_drop), ckpt(P, MBLOCK, s.use_eagle_block_drop)}
                    assert _chunk_ends_ok(seen[r.request_id], P, bs, tails), (r.request_id, P, seen[r.request_id])
                big = max(max(v) for v in seen.values())
                print(f"ok chunk workload mamba_align={mamba!s:5s} async={async_sched!s:5s} spec_k={spec_k} "
                      f"boost_steps={s._slo.stats['boost_steps']} largest chunk={big}")


if __name__ == "__main__":
    test_chunk()
    test_tail_checkpoint()
    test_edf_order_and_late_hold()
    test_doomed_skip()
    test_differential()
    test_admission_controllers_run()
    test_priority_policy_replaced()
    print("ALL OK")
