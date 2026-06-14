# garuda

An LLM inference engine built from scratch in pure PyTorch — no vLLM, no
flash-attn, no custom CUDA. Loads Qwen2.5-0.5B-Instruct safetensors directly
and serves it over an async HTTP API with the same core machinery as
production engines:

- **KV cache** — the textbook decode optimization
- **Paged KV cache** — block-table memory management (PagedAttention, [vLLM, SOSP '23](https://arxiv.org/abs/2309.06180))
- **Continuous batching** — iteration-level scheduling ([Orca, OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu))
- **Chunked prefill** — token-budget steps so long prompts can't stall decodes ([Sarathi-Serve, OSDI '24](https://arxiv.org/abs/2403.02310))
- **Prefix caching** — hash-chained block reuse across requests (vLLM v1 APC / [RadixAttention](https://arxiv.org/abs/2312.07104))
- **Async serving layer** — FastAPI + asyncio, SSE token streaming, hundreds of concurrent requests multiplexed onto one GPU decode loop

Everything measured on a **4GB RTX 3050 Laptop GPU** — memory pressure is the
point: paging matters most when VRAM is scarce.

## Results

All rungs run the same seeded variable-length workload (32–256 output tokens
per request, greedy, HF transformers bf16 as the baseline implementation):

| Rung | Configuration | tok/s | vs rung 1 |
|---|---|---|---|
| 1 | HF, no KV cache, batch=1 (full-prefix recompute) | 43.0 | 1× |
| 2 | HF, KV cache, batch=1 | 68.1 | 1.6× |
| 3 | HF, static batching (64 reqs, padded to longest) | 831.0 useful (1477 raw) | 19.3× |
| 4 | **garuda** (paged KV + continuous batching + chunked prefill) | **818.5** | **19.0×** |

Two honest observations, because benchmarks that only flatter are worthless:

- **Static batching ties the engine (±1.5%) on this workload** — all 64
  requests fit in one batch, which is static batching's best case. Its raw
  throughput (1477 tok/s) is 44% padding waste decoding rows that already
  hit their target length; continuous batching backfills that waste, which
  is why the *useful* numbers converge. The engine's win is everything
  static batching cannot do at all: requests arriving over time, streaming,
  per-request lengths, admission control.
- **The ~19× is against a recompute-everything baseline** on short-ish
  generations; it grows with sequence length (the recompute cost is
  quadratic). The vLLM-comparable number is rung 4 vs rung 2: **~12×**.

**Load test** — 512 concurrent streaming HTTP clients, single burst, 64
tokens each, `max_running=256`:

| metric | value |
|---|---|
| success | 512/512, 0 failures, 0 preemptions |
| aggregate throughput | **1302 tok/s** (25.2s wall) |
| TTFT | p50 7.8s · p99 15.0s (burst queueing behind admission control) |
| inter-token latency | p50 162ms · p99 613ms |
| prefix cache hit rate | 95.5% (shared chat-template preamble) |

**Speculative decoding** (`spec_ngram=2`): on repetition-friendly output
(echo/summarize/extract), 26 → 140 tok/s single-stream (**5.4×**) with 100%
draft acceptance and provably identical outputs; on non-repetitive text it
degrades gracefully toward baseline.

![throughput ladder](bench/results/throughput_ladder.png)
![latency CDF](bench/results/latency_cdf.png)

## Architecture

```
HTTP (FastAPI, SSE)  ──►  AsyncEngine (asyncio bridge, per-request queues)
                                 │ background task
                                 ▼
                          LLMEngine.step()
                 schedule ─► forward ─► sample ─► stream
                    │            │
             Scheduler       ModelRunner
      (continuous batching,  (flat token batch, paged
   chunked prefill budget,    attention: scatter K/V to
   preemption, FCFS admit)    block pool + gather/SDPA)
                    │            │
              BlockManager   PagedKVBackend
       (free list, refcounts,  [L, 2, slots, H_kv, D]
        prefix-cache hashes)      bf16 KV pool
```

- One `step()` = one flat `[N_tokens]` forward mixing decodes (1 token/seq)
  and prefill chunks, capped by a token budget (Sarathi). Dense layers don't
  care about sequence boundaries; only attention reads the batch metadata.
- Block size 16; blocks are ref-counted and content-hashed
  (`h_i = hash(h_{i-1}, tokens_i)`), so shared prompt prefixes are served
  from cache with zero recompute.
- Preemption = discard-and-recompute (free victim's blocks, requeue).

### Honest limitations

Attention gathers each sequence's KV blocks into a contiguous tensor before
`scaled_dot_product_attention` — a production engine reads the block table
inside a fused paged-attention kernel instead. That's the deliberate
trade-off for readable pure-PyTorch code; the scheduler/memory-manager
design is unchanged by it. Future work: Triton paged-attention kernel,
CUDA graphs for decode steps, draft-model speculative decoding.

## Run it

```bash
uv venv .venv && uv pip install -p .venv/bin/python --torch-backend=auto torch && uv pip install -p .venv/bin/python -e .
.venv/bin/python scripts/download_model.py
.venv/bin/python -m pytest tests/ -x        # parity vs HF + paging correctness
.venv/bin/python -m garuda.server.api       # serve on :8000
curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "Explain paged attention briefly.", "stream": true, "max_tokens": 128}'
```

Benchmarks: `python -m bench.baselines --rung 1|2|3`, `python -m bench.bench_engine`,
`python -m bench.load_gen --concurrency 512`, `python -m bench.plots`.
