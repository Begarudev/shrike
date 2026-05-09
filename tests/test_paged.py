"""Paging + continuous batching correctness: the full engine (paged KV,
chunked prefill, batched decode) must reproduce the naive single-sequence
cache's greedy outputs token-for-token.

max_tokens_per_step=48 forces prompts to be chunk-prefilled across steps
while other sequences decode, exercising the mixed-batch path.
"""

import pytest
import torch

MODEL_DIR = "models_cache/qwen2.5-0.5b-instruct"

PROMPTS = [
    "Explain the difference between a process and a thread.",
    "Explain the difference between a mutex and a semaphore in detail.",
    "Write a haiku about memory fragmentation.",
    "What is the capital of France?",
]


@pytest.fixture(scope="module")
def engine():
    from garuda.engine.engine import LLMEngine

    return LLMEngine(MODEL_DIR, max_tokens_per_step=48, gpu_mem_util=0.8)


def naive_greedy(engine, token_ids: list[int], max_new: int) -> list[int]:
    from garuda.models.qwen import NaiveKVBackend

    backend = NaiveKVBackend(engine.model.cfg)
    n = len(token_ids)
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        logits = engine.model(ids, torch.arange(n, device="cuda"), backend)
        out = []
        tok = logits[-1].argmax()
        for i in range(max_new):
            out.append(tok.item())
            if out[-1] in engine.eos_ids:
                break
            logits = engine.model(tok.view(1), torch.tensor([n + i], device="cuda"), backend)
            tok = logits[-1].argmax()
    return out


def test_engine_matches_naive(engine):
    from garuda.engine.request import SamplingParams

    params = SamplingParams(max_new_tokens=48, temperature=0.0)
    tokenized = [
        engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )
        for p in PROMPTS
    ]
    got = engine.generate(tokenized, params, chat=False)
    for prompt_ids, engine_out in zip(tokenized, got):
        ref = naive_greedy(engine, prompt_ids, 48)
        assert engine_out == ref


def test_all_blocks_freed(engine):
    assert engine.block_manager.num_free == engine.block_manager.num_blocks
