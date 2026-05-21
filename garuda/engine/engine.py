"""LLMEngine: ties model runner + scheduler + sampler into a step loop.

step() = schedule -> one batched forward -> sample -> append/finish.
Synchronous by design; the async serving layer drives it from a background
task and fans tokens out to per-request queues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from garuda.engine.block_manager import BlockManager
from garuda.engine.model_runner import ModelRunner
from garuda.engine.request import Request, SamplingParams, Status
from garuda.engine.sampler import sample
from garuda.engine.scheduler import Scheduler
from garuda.models.qwen import QwenForCausalLM


@dataclass
class StepOutput:
    req_id: int
    token_id: int
    finished: bool
    finish_reason: str | None


@dataclass
class EngineStats:
    steps: int = 0
    tokens_generated: int = 0
    started_at: float = field(default_factory=time.perf_counter)


class LLMEngine:
    def __init__(
        self,
        model_dir: str,
        block_size: int = 16,
        gpu_mem_util: float = 0.9,
        max_tokens_per_step: int = 512,
        max_running: int = 256,
        enable_prefix_caching: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = QwenForCausalLM.load(model_dir)
        self.runner = ModelRunner(self.model, block_size, gpu_mem_util)
        self.block_manager = BlockManager(
            self.runner.num_blocks, block_size, enable_prefix_caching
        )
        self.scheduler = Scheduler(self.block_manager, max_tokens_per_step, max_running)
        self.eos_ids = set(self.model.cfg.eos_token_ids)
        self.requests: dict[int, Request] = {}
        self.stats = EngineStats()

    def add_request(self, prompt: str | list[int], sampling: SamplingParams, chat: bool = True) -> int:
        if isinstance(prompt, str):
            if chat:
                token_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                )["input_ids"]
            else:
                token_ids = self.tokenizer(prompt).input_ids
        else:
            token_ids = list(prompt)
        req = Request(token_ids=token_ids, sampling=sampling)
        self.requests[req.req_id] = req
        self.scheduler.add(req)
        return req.req_id

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    def step(self) -> list[StepOutput]:
        batch = self.scheduler.schedule()
        if not batch:
            return []
        logits, sample_reqs = self.runner.run(batch)
        self.scheduler.postprocess(batch)
        outputs: list[StepOutput] = []
        for req, token in zip(sample_reqs, sample(logits, sample_reqs)):
            req.token_ids.append(token)
            self.stats.tokens_generated += 1
            finished, reason = False, None
            if token in self.eos_ids and not req.sampling.ignore_eos:
                finished, reason = True, "stop"
            elif req.num_generated >= req.sampling.max_new_tokens:
                finished, reason = True, "length"
            if finished:
                self.scheduler.finish(req, reason)
                del self.requests[req.req_id]
            outputs.append(StepOutput(req.req_id, token, finished, reason))
        self.stats.steps += 1
        return outputs

    def generate(
        self, prompts: list[str | list[int]], sampling: SamplingParams, chat: bool = True
    ) -> list[list[int]]:
        """Offline batch API for benchmarks: returns generated token ids per prompt."""
        id_map = {self.add_request(p, sampling, chat): i for i, p in enumerate(prompts)}
        results: list[list[int]] = [[] for _ in prompts]
        while self.has_work:
            for out in self.step():
                results[id_map[out.req_id]].append(out.token_id)
        return results

    def metrics(self) -> dict:
        bm, sched = self.block_manager, self.scheduler
        elapsed = time.perf_counter() - self.stats.started_at
        return {
            "steps": self.stats.steps,
            "tokens_generated": self.stats.tokens_generated,
            "avg_tokens_per_sec": round(self.stats.tokens_generated / elapsed, 1),
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "kv_blocks_total": bm.num_blocks,
            "kv_blocks_free": bm.num_free,
            "preemptions": sched.num_preemptions,
            "prefix_cache_hit_rate": round(
                bm.cache_hit_blocks / max(1, bm.cache_query_blocks), 3
            ),
        }
