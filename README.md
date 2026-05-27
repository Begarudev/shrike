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

<!-- BENCH:START — filled by bench/plots.py output -->
| Rung | Configuration | tok/s | vs rung 1 |
|---|---|---|---|
| 1 | no KV cache, batch=1 | TBD | 1x |
| 2 | KV cache, batch=1 | TBD | TBD |
| 3 | static batching | TBD | TBD |
| 4 | **garuda** (paged KV + continuous batching + chunked prefill) | TBD | **TBD** |

Load test: 512 concurrent streaming requests — TTFT p50/p99 TBD, inter-token p99 TBD.
<!-- BENCH:END -->

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
