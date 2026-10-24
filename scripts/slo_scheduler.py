"""SLO-aware (goodput-maximizing) scheduler for vLLM 0.32.

    vllm serve ... --scheduler-cls slo_scheduler.SLOScheduler     (scripts/ on PYTHONPATH)

The grader counts a request only if TTFT <= 0.5 s + 0.25 s per 1K prompt tokens AND average TPOT <= 50 ms.
In an overloaded peak FCFS (or newest-first) lets every request be a little late, so most of them miss.
This scheduler keeps vLLM's scheduling loop untouched and only steers it before each step:

1. Deadline order (GM_SCHED_EDF=1): the waiting queue is re-ordered every step by TTFT deadline
   (arrival + 0.5 + 0.25*prompt/1000). Requests whose own remaining prefill at the measured prefill rate can
   no longer finish before the deadline are "late" and go behind the feasible ones. GM_SCHED_MH=1 instead
   runs a Moore-Hodgson pass over the EDF order (cumulative work, drop the largest prefill first); it was
   worse than the simple test in the sim at budget 1024, so it is off by default.
   Late requests are served oldest-first after that; one that waited GM_SCHED_MAX_WAIT s jumps to the
   front (no starvation, no 900 s timeouts). Preempted requests (already streaming) always go first.
2. Late throttle (GM_SCHED_LATE=1): while any feasible request is waiting, late ones are not admitted
   (admission cap), and running prefills that are already late are skipped for the step, so doomed work
   only takes capacity nobody on time can use.
3. Doomed decodes (GM_SCHED_DOOMED=1): while the measured step TPOT (step wall time / tokens accepted per
   decoding request, EWMA) is above GM_SCHED_DOOMED_TPOT, decodes of requests that already missed TTFT are
   skipped for the step (at most GM_SCHED_MAX_HIDE s in a row and GM_SCHED_MAX_HIDE_TOTAL s per request),
   which makes the steps of the requests that can still pass shorter. Biggest single gain in the sim.
4. TPOT admission control (GM_SCHED_ADMIT=cap, default): the same TPOT estimate drives a dynamic cap on running
   requests (dynamic max_num_seqs): over GM_SCHED_TPOT the cap drops to 97% of the decoding batch per step (never
   below GM_SCHED_MIN_CAP), under 85% of it the cap grows by 1 per step. New requests then wait in the EDF queue
   instead of slowing every running decode past 50 ms. In the sim it costs ~0-10 passes at the real Final load
   and wins ~+100 at 1.15x load. GM_SCHED_ADMIT=budget (Sarathi-style dynamic token budget) is the alternative.
5. Prefix-cache tail checkpoint (GM_SCHED_TAIL_CKPT, default "" = vLLM behaviour; only does something with
   --prefix-match-unit < 288): restricts vLLM's extra prefill stop at the prompt-tail hash boundary (which caches the
   GDN state for the next turn but costs one engine step of TTFT) to prompts that will be extended: "tools",
   "followups", or a minimum prompt length "<N>".
6. Deadline-aware long-prefill chunks (GM_SCHED_CHUNK=1, default off; needs --long-prefill-token-threshold > 0):
   each step a prefill that cannot finish by its TTFT deadline at the static threshold's chunk (288 tokens at
   threshold 512 in mamba align mode) gets a larger per-request threshold, ceil(remaining * step_time / slack *
   GM_SCHED_CHUNK_SAFETY) rounded up to the block, at most GM_SCHED_CHUNK_MAX. Extra budget is handed out in
   deadline order and only from what is left after decodes, every visible running prefill's normal chunk and the
   normal chunks of admissible waiting requests with a closer deadline. Requests that already missed (or can no
   longer make) their deadline get no boost. See _mamba_block_aligned_split for the hook.
Nothing is ever dropped or aborted: every request is still served to completion, only later.

Works with sync and async scheduling (SLOScheduler picks AsyncScheduler as the base when vLLM enabled async
scheduling), chunked prefill, mamba align-mode prefix caching and spec decode: it only (a) re-orders the
FCFS waiting deque, (b) hides some running requests from one schedule() call (same state as a request that
did not fit the token budget), (c) lowers max_num_active_reqs / max_num_scheduled_tokens for one step, (d) only with GM_SCHED_TAIL_CKPT set:
skips the prompt-tail chunk stop for some prompts (exactly the chunking vLLM uses without --prefix-match-unit),
(e) only with GM_SCHED_CHUNK=1: gives some prefills a larger per-request long_prefill_token_threshold for one step
(chunks still go through vLLM's own block-aligned split, so chunk ends stay 288-aligned).

The policy (SLOPolicy) is pure Python and is reused by sched_sim.py.
"""
from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass, field


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _flag(name, default):
    return _env(name, "1" if default else "0", str) not in ("0", "false", "False", "")


@dataclass
class SLOConfig:
    edf: bool = True            # GM_SCHED_EDF     deadline-ordered waiting queue
    mh: bool = False            # GM_SCHED_MH      Moore-Hodgson feasibility (else: own prefill only)
    late: bool = True           # GM_SCHED_LATE    hold back late requests while feasible ones wait
    admit: str = "cap"          # GM_SCHED_ADMIT   cap | budget | 0
    doomed: bool = True         # GM_SCHED_DOOMED  skip decodes of TTFT-missed requests when TPOT is high
    ttft_base: float = 0.5      # GM_SCHED_TTFT_BASE
    ttft_per_k: float = 0.25    # GM_SCHED_TTFT_PER_K
    margin: float = 0.10        # GM_SCHED_MARGIN  s reserved for client/HTTP/detok/async lag
    prefill_tps: float = 9000.  # GM_SCHED_PREFILL_TPS prior for the measured prefill rate (tok/s)
    max_wait: float = 120.      # GM_SCHED_MAX_WAIT  s; a late request older than this goes first
    doomed_tpot: float = 0.015  # GM_SCHED_DOOMED_TPOT s; skip doomed decodes only while step TPOT EWMA > this
    tpot_target: float = 0.045  # GM_SCHED_TPOT    s (target of the cap/budget controllers)
    min_budget: int = 512       # GM_SCHED_MIN_BUDGET tokens (kept >= 2 mamba blocks by the scheduler)
    min_cap: int = 32           # GM_SCHED_MIN_CAP running requests the cap controller never goes below
    max_hide: float = 10.       # GM_SCHED_MAX_HIDE s a doomed decode may be skipped in a row
    max_hide_total: float = 60. # GM_SCHED_MAX_HIDE_TOTAL s of skipping per request, then never again
    doomed_guard: float = 0.0   # GM_SCHED_DOOMED_GUARD s; >0: hide a doomed decode only while its average TPOT so far
                                # stays below this, so skipped requests don't inflate the grader's p95 TPOT
    ewma: float = 0.2           # GM_SCHED_EWMA    weight of the newest step
    log_every: float = 10.      # GM_SCHED_LOG     s between log lines (0 = off)
    tail_ckpt: str = ""         # GM_SCHED_TAIL_CKPT  which prompts get vLLM's extra prefill stop at the prompt-tail
                                # hash boundary (only active with --prefix-match-unit < block size, see
                                # see docs/EXPERIMENTS.md): "" = all (vLLM default), "tools" = only
                                # prompts with a tool system header (multi-turn tool episodes), "followups" = only
                                # prompts that already contain an earlier assistant turn (turn >= 2), "<N>" = only
                                # prompts with >= N tokens, "off" = none
    chunk: bool = False         # GM_SCHED_CHUNK   deadline-aware per-request long-prefill chunk size
    chunk_max: int = 1152       # GM_SCHED_CHUNK_MAX  largest per-request chunk (tokens; 0 = up to the token budget)
    chunk_safety: float = 1.3   # GM_SCHED_CHUNK_SAFETY  multiplier on the required tokens per step
    step_prior: float = 0.08    # GM_SCHED_STEP_PRIOR  s, prior of the measured engine step time (EWMA)
    chunk_tpot: float = 0.045   # GM_SCHED_CHUNK_TPOT s; extra chunk tokens per step are limited so the step TPOT
                                # estimate stays below this (0 = no TPOT guard)

    def keeps_tail(self, prompt_ids, n_prompt, tool_hdr, asst=None) -> bool:
        """Whether a prompt keeps the prefill stop that checkpoints its mamba state at the prompt tail.
        tool_hdr: token ids a tool-bearing prompt starts with; asst: (im_start id, 'assistant' id)."""
        mode = self.tail_ckpt
        if mode == "":
            return True
        if mode == "off":
            return False
        if mode == "tools":
            return bool(tool_hdr) and list(prompt_ids[:len(tool_hdr)]) == tool_hdr
        if mode == "followups":   # an assistant turn before the generation prompt's own "<|im_start|>assistant"
            if not asst:
                return False
            a, b = asst
            n = sum(1 for i in range(len(prompt_ids) - 1) if prompt_ids[i] == a and prompt_ids[i + 1] == b)
            return n >= 2
        return n_prompt >= int(mode)

    @classmethod
    def from_env(cls) -> "SLOConfig":
        d = cls()
        return cls(
            edf=_flag("GM_SCHED_EDF", d.edf), mh=_flag("GM_SCHED_MH", d.mh), late=_flag("GM_SCHED_LATE", d.late),
            admit=_env("GM_SCHED_ADMIT", d.admit, str), doomed=_flag("GM_SCHED_DOOMED", d.doomed),
            ttft_base=_env("GM_SCHED_TTFT_BASE", d.ttft_base), ttft_per_k=_env("GM_SCHED_TTFT_PER_K", d.ttft_per_k),
            margin=_env("GM_SCHED_MARGIN", d.margin), prefill_tps=_env("GM_SCHED_PREFILL_TPS", d.prefill_tps),
            max_wait=_env("GM_SCHED_MAX_WAIT", d.max_wait), tpot_target=_env("GM_SCHED_TPOT", d.tpot_target),
            doomed_tpot=_env("GM_SCHED_DOOMED_TPOT", d.doomed_tpot),
            doomed_guard=_env("GM_SCHED_DOOMED_GUARD", d.doomed_guard),
            max_hide_total=_env("GM_SCHED_MAX_HIDE_TOTAL", d.max_hide_total),
            min_budget=_env("GM_SCHED_MIN_BUDGET", d.min_budget, int), min_cap=_env("GM_SCHED_MIN_CAP", d.min_cap, int),
            max_hide=_env("GM_SCHED_MAX_HIDE", d.max_hide), ewma=_env("GM_SCHED_EWMA", d.ewma),
            log_every=_env("GM_SCHED_LOG", d.log_every), tail_ckpt=_env("GM_SCHED_TAIL_CKPT", d.tail_ckpt, str),
            chunk=_flag("GM_SCHED_CHUNK", d.chunk), chunk_max=_env("GM_SCHED_CHUNK_MAX", d.chunk_max, int),
            chunk_safety=_env("GM_SCHED_CHUNK_SAFETY", d.chunk_safety),
            step_prior=_env("GM_SCHED_STEP_PRIOR", d.step_prior), chunk_tpot=_env("GM_SCHED_CHUNK_TPOT", d.chunk_tpot))


@dataclass
class Plan:
    waiting: list | None = None      # new waiting order (None = keep)
    hidden: list = field(default_factory=list)  # running requests to skip this step
    active_cap: int | None = None    # max running requests after admission this step
    token_budget: int | None = None  # token budget this step


class SLOPolicy:
    """Request objects need: request_id, arrival_time, num_prompt_tokens, num_computed_tokens, num_output_tokens."""

    def __init__(self, cfg: SLOConfig, max_seqs: int, max_budget: int):
        self.cfg = cfg
        self.max_seqs = max_seqs
        self.max_budget = max_budget
        self.budget = float(max_budget)
        self.cap = float(max_seqs)
        self.rate = cfg.prefill_tps      # measured prefill tokens per wall second while prefilling
        self.tpot = 0.0                  # EWMA of the per-step TPOT of decoding requests
        self.step = cfg.step_prior       # EWMA of the engine step wall time
        self.first_tok: dict = {}        # request_id -> time of first output token
        self.hidden_since: dict = {}     # request_id -> time it was first skipped (current streak)
        self.hidden_total: dict = {}     # request_id -> seconds skipped so far
        self._last_plan = None
        self.stats = {"steps": 0, "hidden": 0, "late_waiting": 0, "late_held": 0, "boosted": 0, "boost_steps": 0}
        self._last_log = 0.0

    # ---- SLO arithmetic -------------------------------------------------------------------------
    def slo(self, r) -> float:
        return self.cfg.ttft_base + self.cfg.ttft_per_k * r.num_prompt_tokens / 1000.0

    def deadline(self, r) -> float:
        return r.arrival_time + self.slo(r)

    def remaining_prefill(self, r) -> int:
        return max(r.num_prompt_tokens - r.num_computed_tokens, 0)

    def ttft_missed(self, r, now) -> bool:
        """True once the request can no longer meet its TTFT SLO."""
        ft = self.first_tok.get(r.request_id)
        if ft is not None:
            return ft - r.arrival_time > self.slo(r)
        return now + self.remaining_prefill(r) / self.rate > self.deadline(r)

    # ---- per step -------------------------------------------------------------------------------
    def plan(self, now: float, running: list, waiting: list) -> Plan:
        cfg = self.cfg
        p = Plan()
        self.stats["steps"] += 1
        prefilling = [r for r in running if r.request_id not in self.first_tok and self.remaining_prefill(r) > 0]
        n_feasible = 0
        hold_cap = None
        if cfg.edf and waiting:
            resumed, starving, cand = [], [], []
            for r in waiting:
                if r.num_output_tokens > 0:
                    resumed.append(r)          # preempted while streaming: TTFT is history, resume first
                elif now - r.arrival_time > cfg.max_wait:
                    starving.append(r)
                else:
                    cand.append(r)
            cand.sort(key=self.deadline)
            feasible, late = self._feasible(now, cand, prefilling)
            resumed.sort(key=lambda r: r.arrival_time)
            starving.sort(key=lambda r: r.arrival_time)
            late.sort(key=lambda r: r.arrival_time)
            n_feasible = len(feasible)
            p.waiting = resumed + starving + feasible + late
            self.stats["late_waiting"] = len(late)
            if cfg.late and late and feasible:
                # hold back late admissions while on-time work waits: admit at most the requests ahead of them
                hold_cap = len(resumed) + len(starving) + len(feasible)
                self.stats["late_held"] += 1
        hidden = []
        if cfg.late and n_feasible:
            # a running prefill that is already late does not get budget while feasible requests wait
            hidden += [r for r in prefilling
                       if self.ttft_missed(r, now) and now - r.arrival_time <= cfg.max_wait]
        step = now - self._last_plan if self._last_plan is not None else 0.0
        self._last_plan = now
        for rid in self.hidden_since:
            self.hidden_total[rid] = self.hidden_total.get(rid, 0.0) + step
        if cfg.doomed and self.tpot > cfg.doomed_tpot:
            for r in running:
                rid = r.request_id
                if rid in self.first_tok and self.ttft_missed(r, now) \
                        and self.hidden_total.get(rid, 0.0) < cfg.max_hide_total \
                        and (cfg.doomed_guard <= 0 or (now - self.first_tok[rid]) / max(r.num_output_tokens, 1)
                             < cfg.doomed_guard):
                    t0 = self.hidden_since.setdefault(rid, now)
                    if now - t0 < cfg.max_hide:
                        hidden.append(r)
                    else:
                        self.hidden_since.pop(rid, None)   # serve one step, then a new streak
        hid = {r.request_id for r in hidden}
        for rid in [k for k in self.hidden_since if k not in hid]:
            self.hidden_since.pop(rid, None)
        if len(hidden) >= len(running):
            hidden = []   # never hide everything: keep the engine making progress
        p.hidden = hidden
        self.stats["hidden"] = len(hidden)
        if hold_cap is not None:
            p.active_cap = len(running) - len(hidden) + hold_cap   # counts visible running requests
        if cfg.admit == "budget":
            p.token_budget = int(self.budget)
        elif cfg.admit == "cap":
            cap = int(self.cap)
            p.active_cap = cap if p.active_cap is None else min(cap, p.active_cap)
        return p

    def _feasible(self, now, cand, prefilling):
        """EDF list -> (feasible in EDF order, late). Work ahead = running prefills (served first by vLLM)."""
        cfg = self.cfg
        work = sum(self.remaining_prefill(r) for r in prefilling if not self.ttft_missed(r, now))
        t = now + cfg.margin
        if not cfg.mh:
            ok, late = [], []
            for r in cand:
                (ok if t + self.remaining_prefill(r) / self.rate <= self.deadline(r) else late).append(r)
            return ok, late
        chosen, heap, late = [], [], []   # heap of (-prefill, idx) over chosen
        for r in cand:
            w = self.remaining_prefill(r)
            chosen.append(r)
            heapq.heappush(heap, (-w, len(chosen) - 1))
            work += w
            if t + work / self.rate > self.deadline(r):
                negw, i = heapq.heappop(heap)    # drop the largest job so far (Moore-Hodgson)
                work += negw
                late.append(chosen[i])
                chosen[i] = None
        return [r for r in chosen if r is not None], late

    def chunk_caps(self, now, running, waiting, room, budget, thr, block, dec_tokens) -> dict:
        """GM_SCHED_CHUNK: per-request long-prefill thresholds for this step, {request_id: tokens > thr}.

        running: the running requests vLLM will see this step (hidden ones already removed), in vLLM's order;
        waiting: the waiting queue in the order vLLM will admit from; room: how many of them can be admitted;
        budget: token budget; thr: the static long_prefill_token_threshold; block: chunk alignment (mamba block,
        1 without align mode); dec_tokens: tokens a decode takes (1 + spec k). Requests not in the result keep thr.
        """
        cfg = self.cfg
        self.stats["boosted"] = 0
        if not cfg.chunk or thr <= 0 or budget <= 0:
            return {}
        block = max(int(block), 1)
        base = thr // block * block or thr      # what thr really gives a non-final chunk after alignment

        def base_chunk(r):
            return min(self.remaining_prefill(r), base)

        reserved = 0
        cands = []                       # (request, is_waiting)
        for r in running:
            rem = self.remaining_prefill(r)
            if r.request_id in self.first_tok or rem == 0:
                reserved += dec_tokens   # decode (or a resumed request replaying output: small)
                continue
            reserved += base_chunk(r)    # served before any waiting request, in list order
            cands.append((r, False))
        closer = []                      # (deadline, base chunk) of admissible on-time waiting prefills
        for r in waiting[:max(int(room), 0)]:
            if r.num_output_tokens > 0:
                reserved += base_chunk(r)    # resumed after preemption: goes first, never boosted
            elif not self.ttft_missed(r, now):
                closer.append((self.deadline(r), base_chunk(r)))
                cands.append((r, True))
        if reserved >= budget:
            return {}
        cmax = cfg.chunk_max if cfg.chunk_max > 0 else budget
        urgent = []
        for r, is_wait in cands:
            rem = self.remaining_prefill(r)
            if rem <= base:
                continue                 # one normal chunk finishes it
            d = self.deadline(r)
            slack = d - now - cfg.margin
            if slack <= 0 or self.ttft_missed(r, now):
                continue                 # will fail anyway: no boost
            need = rem * self.step / slack * cfg.chunk_safety
            if need <= base:
                continue                 # on time with the normal chunk
            want = min(-(-int(need) // block) * block, cmax // block * block or block)
            if want <= base:
                continue
            urgent.append((d, r, is_wait, want))
        if not urgent:
            return {}
        # TPOT guard: every extra prefill token lengthens the step by ~1/rate s for every decoding request. Keep the
        # step TPOT estimate (step time / accepted tokens per decode) under chunk_tpot.
        guard = budget
        if cfg.chunk_tpot > 0 and self.tpot > 0:
            guard = int(max(self.step * (cfg.chunk_tpot / self.tpot - 1.0), 0.0) * self.rate)
        urgent.sort(key=lambda x: x[0])
        caps, given = {}, 0
        for d, r, is_wait, want in urgent:
            free = min(budget - reserved, guard) - given - sum(b for dd, b in closer if dd < d)
            if is_wait:
                free -= base_chunk(r)    # its own normal chunk is not in `reserved`
            extra = min(want - base, free) // block * block
            if extra <= 0:
                continue
            caps[r.request_id] = base + extra
            given += extra
        if caps:
            self.stats["boosted"] = len(caps)
            self.stats["boost_steps"] += 1
        return caps

    def observe(self, now, dt, n_decode, decode_tokens, prefill_tokens, newly_started=()):
        """After a step: dt = wall seconds of the step, n_decode requests produced decode_tokens tokens."""
        cfg, a = self.cfg, self.cfg.ewma
        for r in newly_started:
            self.first_tok.setdefault(r.request_id, now)
        if dt <= 0 or dt > 5:
            return
        self.step = (1 - a) * self.step + a * dt
        if prefill_tokens >= 0.5 * self.budget:   # budget-bound steps measure the queue drain rate
            self.rate = (1 - a) * self.rate + a * min(max(prefill_tokens / dt, 1000.0), 50000.0)
        if n_decode > 0 and decode_tokens > 0:
            inst = dt * n_decode / decode_tokens
            self.tpot = (1 - a) * self.tpot + a * inst
        if cfg.admit == "budget":
            # multiplicative feedback on the TPOT estimate, clipped (Sarathi-style token budget)
            ratio = cfg.tpot_target / self.tpot if self.tpot > 0 else 2.0
            self.budget = min(max(self.budget * min(max(ratio, 0.7), 1.1), cfg.min_budget), self.max_budget)
        elif cfg.admit == "cap":
            if self.tpot > cfg.tpot_target:
                self.cap = max(cfg.min_cap, min(self.cap, n_decode) * 0.97)
            elif self.tpot < 0.85 * cfg.tpot_target:
                self.cap = min(self.max_seqs, self.cap + 1)

    def forget(self, request_id):
        self.first_tok.pop(request_id, None)
        self.hidden_since.pop(request_id, None)
        self.hidden_total.pop(request_id, None)

    def summary(self) -> str:
        return (f"rate={self.rate:.0f}tok/s tpot~{self.tpot * 1000:.1f}ms budget={int(self.budget)} cap={int(self.cap)} "
                f"hidden={self.stats['hidden']} late_waiting={self.stats['late_waiting']} "
                + (f"step~{self.step * 1000:.0f}ms boosted={self.stats['boosted']} "
                   f"boost_steps={self.stats['boost_steps']} " if self.cfg.chunk else "") +
                f"late_held_steps={self.stats['late_held']} steps={self.stats['steps']}")


# ------------------------------------------------------------------------------------------------------
# vLLM integration
# ------------------------------------------------------------------------------------------------------
_CLASSES: dict = {}


def _make_class(base):
    from vllm.logger import init_logger
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    logger = init_logger("slo_scheduler")

    class _SLOScheduler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = SLOConfig.from_env()
            block = int(getattr(self.cache_config, "block_size", 16) or 16)
            cfg.min_budget = max(cfg.min_budget, 2 * block)
            self._slo = SLOPolicy(cfg, self.max_num_running_reqs, self.max_num_scheduled_tokens)
            self._slo_base_budget = self.max_num_scheduled_tokens
            self._slo_base_active = self.max_num_active_reqs
            self._slo_last_t = None         # wall time of the previous update_from_output
            self._slo_sched_t: dict = {}     # id(scheduler_output) -> wall time it was built
            if cfg.edf and not isinstance(self.waiting, FCFSRequestQueue):
                # --scheduling-policy priority: we impose our own order, so use a plain deque
                assert not self.waiting
                self.waiting = FCFSRequestQueue()
            self._tail_hdr, self._tail_asst = [], None
            self._tail_keep: dict = {}      # request_id -> keeps the prompt-tail prefill stop
            if cfg.tail_ckpt not in ("", "off", "tools", "followups"):
                int(cfg.tail_ckpt)           # fail at startup, not mid-run
            if cfg.tail_ckpt in ("tools", "followups"):
                self._tail_hdr, self._tail_asst = self._chat_token_ids(logger)
            # GM_SCHED_CHUNK: per-request long-prefill thresholds of the current schedule() call
            self._slo_caps: dict = {}
            self._slo_thr = int(self.scheduler_config.long_prefill_token_threshold or 0)
            mc = getattr(self.vllm_config, "model_config", None)
            self._slo_chunk_ok = cfg.chunk and self._slo_thr > 0 \
                and not self.adaptive_long_prefill_threshold \
                and not getattr(mc, "is_multimodal_model", False)
            if cfg.chunk and not self._slo_chunk_ok:
                logger.warning("SLOScheduler: GM_SCHED_CHUNK=1 needs --long-prefill-token-threshold > 0, no adaptive "
                               "threshold and a text-only model; deadline-aware chunks are OFF")
            logger.info("SLOScheduler(base=%s) %s partial_tail_hits=%s", base.__name__, cfg,
                        getattr(self, "mamba_partial_cache_hit", None))

        def _chat_token_ids(self, logger):
            """(ids every tool-bearing prompt starts with, (im_start, 'assistant') ids) for the served chat template."""
            env = os.environ.get("GM_SCHED_TAIL_PREFIX_IDS", "")
            if env:
                asst = [int(x) for x in os.environ.get("GM_SCHED_TAIL_ASSISTANT_IDS", "").split(",") if x.strip()]
                return [int(x) for x in env.split(",") if x.strip()], (tuple(asst) if len(asst) == 2 else None)
            try:
                from transformers import AutoTokenizer
                mc = self.vllm_config.model_config
                tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=mc.trust_remote_code)
                hdr = tok("<|im_start|>system\n# Tools\n\nYou have access to the following functions:",
                          add_special_tokens=False)["input_ids"]
                asst = tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
                logger.info("SLOScheduler tail checkpoint: tool header ids %s, assistant ids %s", hdr, asst)
                return list(hdr), (tuple(asst) if len(asst) == 2 else None)
            except Exception as e:  # noqa: BLE001
                logger.warning("SLOScheduler: no chat token ids (%r); GM_SCHED_TAIL_CKPT falls back to vLLM default", e)
                self._slo.cfg.tail_ckpt = ""
                return [], None

        def _slo_cap(self, request, start):
            """GM_SCHED_CHUNK: this request's long-prefill threshold for the current step, None = vLLM's own."""
            if not self._slo_caps:
                return None
            if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
                return None                  # not prefilling (decode / spec tokens): untouched
            return self._slo_caps.get(request.request_id, self._slo_thr)

        def _mamba_block_aligned_split(self, request, num_new_tokens, *args, **kwargs):
            # GM_SCHED_CHUNK: while caps are active the step's threshold (read once by schedule() into a local)
            # is raised to the largest cap, so here every prefill chunk is clipped back to its own threshold
            # (the static one unless boosted) BEFORE vLLM's block-aligned split, and the split itself sees that
            # threshold too (it reads scheduler_config.long_prefill_token_threshold for the "block wider than
            # a chunk" rule). Net effect: identical to running vLLM with long_prefill_token_threshold = cap for
            # this request, including 288-aligned non-final chunk ends, prompt-tail/junction stops and the
            # MTP lookahead reservation that runs afterwards.
            cap = self._slo_cap(request, request.num_computed_tokens + sum(args) + sum(kwargs.values()))
            if cap is None:
                return self._slo_tail_split(request, num_new_tokens, *args, **kwargs)
            sc = self.scheduler_config
            old = sc.long_prefill_token_threshold
            sc.long_prefill_token_threshold = cap
            try:
                return self._slo_tail_split(request, min(num_new_tokens, cap), *args, **kwargs)
            finally:
                sc.long_prefill_token_threshold = old

        def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
            # GM_SCHED_CHUNK without mamba align mode (plain attention: no split hook): clip here, the last
            # per-request step before the chunk is committed (no encoder inputs: text-only models only).
            if self._slo_caps and not self.need_mamba_block_aligned_split:
                cap = self._slo_cap(request, num_computed_tokens)
                if cap is not None:
                    num_new_tokens = min(num_new_tokens, cap)
            return super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)

        def _slo_tail_split(self, request, num_new_tokens, *args, **kwargs):
            # With --prefix-match-unit finer than the 288-token block, vLLM ends a prefill chunk at the prompt's last
            # hash boundary so the GDN state there is cached for a follow-up turn (GDN has no in-kernel checkpoint).
            # That costs every prompt one extra engine step before its first token. GM_SCHED_TAIL_CKPT keeps the stop
            # only for prompts that will be extended (tool episodes); for the others the chunk simply runs to the end,
            # exactly as with prefix_match_unit == block size (no state is cached at the tail, nothing else changes).
            if not self.mamba_partial_cache_hit or self._slo.cfg.tail_ckpt == "":
                return super()._mamba_block_aligned_split(request, num_new_tokens, *args, **kwargs)
            keep = self._tail_keep.get(request.request_id)
            if keep is None:
                keep = self._slo.cfg.keeps_tail(request.prompt_token_ids or (), request.num_prompt_tokens,
                                                self._tail_hdr, self._tail_asst)
                if len(self._tail_keep) > 8 * self.max_num_running_reqs:
                    self._tail_keep = {k: v for k, v in self._tail_keep.items() if k in self.requests}
                self._tail_keep[request.request_id] = keep
            if keep:
                return super()._mamba_block_aligned_split(request, num_new_tokens, *args, **kwargs)
            self.mamba_partial_cache_hit = False
            try:
                return super()._mamba_block_aligned_split(request, num_new_tokens, *args, **kwargs)
            finally:
                self.mamba_partial_cache_hit = True

        def schedule(self, *args, **kwargs):
            now = time.time()
            pol = self._slo
            plan = pol.plan(now, self.running, list(self.waiting))
            if plan.waiting is not None:
                self.waiting.clear()
                self.waiting.extend(plan.waiting)
            hidden = plan.hidden
            active = self._slo_base_active
            if plan.active_cap is not None:
                active = min(active, max(plan.active_cap, 1))
            if hidden:
                hid = {id(r) for r in hidden}
                self.running = [r for r in self.running if id(r) not in hid]
                # hidden requests still hold model-runner slots
                active = min(active, self.max_num_running_reqs - len(hidden))
            self.max_num_active_reqs = active
            if plan.token_budget is not None:
                self.max_num_scheduled_tokens = max(min(plan.token_budget, self._slo_base_budget), 1)
            if self._slo_chunk_ok and sum(self.get_request_counts()) > 1:   # vLLM drops the threshold when alone
                self._slo_caps = pol.chunk_caps(
                    now, self.running, plan.waiting if plan.waiting is not None else list(self.waiting),
                    active - len(self.running), self.max_num_scheduled_tokens, self._slo_thr,
                    self.cache_config.block_size if self.need_mamba_block_aligned_split else 1,
                    1 + self.num_spec_tokens)
                if self._slo_caps:
                    self.scheduler_config.long_prefill_token_threshold = max(self._slo_caps.values())
            try:
                out = super().schedule(*args, **kwargs)
            finally:
                if self._slo_caps:
                    self._slo_caps = {}
                    self.scheduler_config.long_prefill_token_threshold = self._slo_thr
                if hidden:
                    # back at the end: they are the first preemption victims if KV runs short
                    self.running.extend(hidden)
                self.max_num_active_reqs = self._slo_base_active
                self.max_num_scheduled_tokens = self._slo_base_budget
            if out.total_num_scheduled_tokens > 0:
                self._slo_sched_t[id(out)] = time.time()
                if len(self._slo_sched_t) > 64:   # outputs that never came back (should not happen)
                    self._slo_sched_t.clear()
            if pol.cfg.log_every and now - pol._last_log > pol.cfg.log_every:
                pol._last_log = now
                logger.info("SLOScheduler running=%d waiting=%d %s", len(self.running), len(self.waiting),
                            pol.summary())
            return out

        def update_from_output(self, scheduler_output, model_runner_output):
            now = time.time()
            # snapshot before the base class mutates/finishes requests
            n_sched = scheduler_output.num_scheduled_tokens
            spec = scheduler_output.scheduled_spec_decode_tokens
            prefill_tokens = 0
            for rid, n in n_sched.items():
                extra = n - 1 - len(spec.get(rid, ()))
                if extra > 0:
                    prefill_tokens += extra
            before = {}
            for rid in n_sched:
                r = self.requests.get(rid)
                if r is not None:
                    before[rid] = r.num_output_tokens
            result = super().update_from_output(scheduler_output, model_runner_output)
            pol = self._slo
            n_dec, dec_tok, started = 0, 0, []
            for rid, b in before.items():
                r = self.requests.get(rid)
                if r is None:
                    pol.forget(rid)
                    continue
                gained = r.num_output_tokens - b
                if gained <= 0:
                    continue
                if b == 0:
                    started.append(r)
                else:
                    n_dec += 1
                    dec_tok += gained
            # GPU time of this step ~ now - max(when it was scheduled, when the previous step finished):
            # with async scheduling step N is built while step N-1 still runs.
            t_sched = self._slo_sched_t.pop(id(scheduler_output), None)
            dt = 0.0
            if t_sched is not None:
                dt = now - max(t_sched, self._slo_last_t or 0.0)
                self._slo_last_t = now
            pol.observe(now, dt, n_dec, dec_tok, prefill_tokens, started)
            # drop state of requests that finished outside this step's batch
            if len(pol.first_tok) > 4 * self.max_num_running_reqs:
                for rid in [k for k in pol.first_tok if k not in self.requests]:
                    pol.forget(rid)
            return result

    _SLOScheduler.__name__ = _SLOScheduler.__qualname__ = f"SLO{base.__name__}"
    return _SLOScheduler


def SLOScheduler(*args, **kwargs):
    """Factory used as --scheduler-cls slo_scheduler.SLOScheduler: keeps async scheduling when vLLM enabled it."""
    vllm_config = kwargs.get("vllm_config", args[0] if args else None)
    use_async = bool(vllm_config.scheduler_config.async_scheduling)
    if use_async:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as base
    else:
        from vllm.v1.core.sched.scheduler import Scheduler as base
    if base not in _CLASSES:
        _CLASSES[base] = _make_class(base)
    return _CLASSES[base](*args, **kwargs)
