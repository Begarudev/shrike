"""Batched forward over a paged KV cache.

Each engine step runs ONE flat token batch through the model (Orca-style
iteration-level batching): all scheduled sequences' new tokens are
concatenated to shape [N_tokens]. Dense layers (embeddings, projections, MLP)
are sequence-agnostic; only attention consults the batch metadata.

Token order convention: [all decode tokens (1 per seq)] + [prefill chunks].

Attention is pure PyTorch: new K/V are scattered into the paged pool via a
slot mapping, then per-sequence context is gathered from block tables and fed
to scaled_dot_product_attention. A production engine fuses this into a paged
attention kernel (vLLM SOSP '23); the gather is our deliberate simplicity
trade-off and is documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from garuda.engine.request import Request
from garuda.models.qwen import QwenConfig, QwenForCausalLM


@dataclass
class BatchMeta:
    slot_mapping: torch.Tensor  # [N] long: flat pool slot for each new token
    num_decode_seqs: int
    decode_block_tables: torch.Tensor | None  # [B_dec, max_blocks] long, 0-padded
    decode_ctx_lens: torch.Tensor | None  # [B_dec] long, includes the new token
    # per prefill seq: (q_start, q_len, ctx_len, block_table [num_blocks] long)
    prefill: list[tuple[int, int, int, torch.Tensor]] = field(default_factory=list)


class PagedKVBackend:
    """KV pool: [num_layers, 2, num_slots, num_kv_heads, head_dim]."""

    def __init__(self, cfg: QwenConfig, num_blocks: int, block_size: int, device: str = "cuda"):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pool = torch.zeros(
            cfg.num_layers, 2, num_blocks * block_size, cfg.num_kv_heads, cfg.head_dim,
            dtype=cfg.dtype, device=device,
        )
        self.meta: BatchMeta | None = None  # set by ModelRunner before each forward

    def run(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        meta = self.meta
        k_pool, v_pool = self.pool[layer_idx, 0], self.pool[layer_idx, 1]
        k_pool.index_copy_(0, meta.slot_mapping, k)
        v_pool.index_copy_(0, meta.slot_mapping, v)

        out = torch.empty_like(q)
        bs = self.block_size
        k_blocks = k_pool.view(self.num_blocks, bs, *k_pool.shape[1:])
        v_blocks = v_pool.view(self.num_blocks, bs, *v_pool.shape[1:])

        n_dec = meta.num_decode_seqs
        if n_dec:
            # [B, max_b, bs, H_kv, D] -> [B, H_kv, S, D]
            kg = k_blocks[meta.decode_block_tables].flatten(1, 2).transpose(1, 2)
            vg = v_blocks[meta.decode_block_tables].flatten(1, 2).transpose(1, 2)
            qd = q[:n_dec].unsqueeze(2)  # [B, H, 1, D]
            mask = (
                torch.arange(kg.shape[2], device=q.device)[None, :] < meta.decode_ctx_lens[:, None]
            )[:, None, None, :]
            attn = F.scaled_dot_product_attention(qd, kg, vg, attn_mask=mask, enable_gqa=True)
            out[:n_dec] = attn.squeeze(2)

        for q_start, q_len, ctx_len, block_table in meta.prefill:
            kg = k_blocks[block_table].flatten(0, 1)[:ctx_len].transpose(0, 1)[None]  # [1,H_kv,C,D]
            vg = v_blocks[block_table].flatten(0, 1)[:ctx_len].transpose(0, 1)[None]
            qp = q[q_start : q_start + q_len].permute(1, 0, 2)[None]  # [1, H, L, D]
            mask = torch.ones(q_len, ctx_len, dtype=torch.bool, device=q.device).tril(
                diagonal=ctx_len - q_len
            )
            attn = F.scaled_dot_product_attention(qp, kg, vg, attn_mask=mask, enable_gqa=True)
            out[q_start : q_start + q_len] = attn[0].permute(1, 0, 2)
        return out


class ModelRunner:
    def __init__(self, model: QwenForCausalLM, block_size: int, gpu_mem_util: float = 0.9):
        self.model = model
        self.cfg = model.cfg
        self.block_size = block_size
        free, _total = torch.cuda.mem_get_info()
        block_bytes = (
            self.cfg.num_layers * 2 * block_size * self.cfg.num_kv_heads * self.cfg.head_dim
            * self.pool_dtype_size()
        )
        self.num_blocks = int(free * gpu_mem_util) // block_bytes
        self.backend = PagedKVBackend(self.cfg, self.num_blocks, block_size)

    def pool_dtype_size(self) -> int:
        return torch.tensor([], dtype=self.cfg.dtype).element_size()

    @torch.inference_mode()
    def run(self, batch: list[tuple[Request, int]]) -> tuple[torch.Tensor, list[Request]]:
        """Execute one step. batch = [(req, num_new_tokens)], decodes first.

        Returns (logits [num_sampling_seqs, vocab], sampling_reqs) — logits of
        the last computed token for every request that finished its prompt
        this step (all decodes + final prefill chunks).
        """
        device = "cuda"
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sample_rows: list[int] = []
        sample_reqs: list[Request] = []

        decode_tables, decode_ctx = [], []
        prefill_meta: list[tuple[int, int, int, torch.Tensor]] = []
        num_decode_seqs = sum(1 for _, n in batch if n == 1)

        for req, n_new in batch:
            start = req.num_computed_tokens
            toks = req.token_ids[start : start + n_new]
            q_start = len(input_ids)
            input_ids.extend(toks)
            positions.extend(range(start, start + n_new))
            slots.extend(
                req.block_table[p // self.block_size] * self.block_size + p % self.block_size
                for p in range(start, start + n_new)
            )
            ctx_len = start + n_new
            table = torch.tensor(req.block_table, dtype=torch.long, device=device)
            if n_new == 1:
                decode_tables.append(table)
                decode_ctx.append(ctx_len)
            else:
                prefill_meta.append((q_start, n_new, ctx_len, table))
            if ctx_len >= req.num_tokens:  # computed through last token -> sample
                sample_rows.append(q_start + n_new - 1)
                sample_reqs.append(req)

        meta = BatchMeta(
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            num_decode_seqs=num_decode_seqs,
            decode_block_tables=(
                torch.nn.utils.rnn.pad_sequence(decode_tables, batch_first=True)
                if decode_tables else None
            ),
            decode_ctx_lens=(
                torch.tensor(decode_ctx, dtype=torch.long, device=device) if decode_ctx else None
            ),
            prefill=prefill_meta,
        )
        self.backend.meta = meta
        logits = self.model(
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(positions, dtype=torch.long, device=device),
            self.backend,
        )
        return logits[sample_rows], sample_reqs
