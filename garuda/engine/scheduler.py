"""Iteration-level scheduler: continuous batching (Orca, OSDI '22) with a
Sarathi-Serve (OSDI '24) token budget and chunked prefill.

Every step:
  1. RUNNING sequences are served first. Decodes cost 1 budget token;
     partially-prefilled sequences continue with a chunk that fits the
     remaining budget.
  2. Remaining budget admits WAITING sequences (FCFS), chunking their prompts.
  3. If a decode can't get a KV block, the newest running sequence is
     preempted (blocks freed, recomputed later) — discard-and-recompute
     preemption, as in vLLM.

Chunked prefill keeps every step's token count bounded, so a long prompt
can't stall in-flight decodes — this is what tames p99 inter-token latency.
"""

from __future__ import annotations

from collections import deque

from garuda.engine.block_manager import BlockManager
from garuda.engine.request import Request, Status


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_tokens_per_step: int = 512,
        max_running: int = 256,
    ):
        self.bm = block_manager
        self.max_tokens_per_step = max_tokens_per_step
        self.max_running = max_running
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt_for(self, req: Request) -> bool:
        """Free blocks by preempting the newest running request. Returns False
        if req itself became the victim (caller must skip it this step)."""
        victim = self.running.pop()
        self.num_preemptions += 1
        self.bm.free(victim)
        victim.num_computed_tokens = 0
        victim.status = Status.WAITING
        self.waiting.appendleft(victim)
        return victim is not req

    def schedule(self) -> list[tuple[Request, int]]:
        batch: list[tuple[Request, int]] = []
        prefill_batch: list[tuple[Request, int]] = []
        budget = self.max_tokens_per_step

        # 1. running sequences: decodes and in-progress prefills
        for req in list(self.running):
            if budget == 0:
                break
            if req.status is not Status.RUNNING:  # preempted earlier this step
                continue
            n_new = 1 if req.prefill_done else min(
                req.num_prompt_tokens - req.num_computed_tokens, budget
            )
            while not self.bm.can_append(req, n_new):
                if not self._preempt_for(req):
                    n_new = 0
                    break
            if n_new == 0:
                continue
            self.bm.append_blocks(req, n_new)
            (batch if n_new == 1 else prefill_batch).append((req, n_new))
            budget -= n_new

        # 2. admit waiting sequences with prefill chunks
        while self.waiting and budget > 0 and len(self.running) < self.max_running:
            req = self.waiting[0]
            if not req.block_table and self.bm.enable_prefix_caching:
                cached_blocks, cached_tokens = self.bm.match_prefix(req.token_ids[: req.num_prompt_tokens])
                req.block_table.extend(cached_blocks)
                req.num_computed_tokens = cached_tokens
            n_new = min(req.num_prompt_tokens - req.num_computed_tokens, budget)
            if not self.bm.can_append(req, n_new):
                break  # FCFS: don't skip ahead of the head request
            self.waiting.popleft()
            self.bm.append_blocks(req, n_new)
            req.status = Status.RUNNING
            self.running.append(req)
            (batch if n_new == 1 else prefill_batch).append((req, n_new))
            budget -= n_new

        return batch + prefill_batch  # decodes first (ModelRunner convention)

    def postprocess(self, batch: list[tuple[Request, int]]) -> None:
        """Advance computed-token counts; register full blocks for prefix cache."""
        for req, n_new in batch:
            req.num_computed_tokens += n_new
            self.bm.register_full_blocks(req)

    def finish(self, req: Request, reason: str) -> None:
        req.status = Status.FINISHED
        req.finish_reason = reason
        self.running.remove(req)
        self.bm.free(req)
