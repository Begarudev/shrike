"""Triton paged-decode kernel vs the grouped-einsum reference."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from shrike.engine.triton_attention import triton_paged_decode

BLOCK_SIZE = 16
H, H_KV, D = 14, 2, 64
NUM_BLOCKS = 64


def einsum_reference(q, k_pool, v_pool, tables, ctx_lens):
    b = q.shape[0]
    groups = H // H_KV
    max_blocks = tables.shape[1]
    k_blocks = k_pool.view(NUM_BLOCKS, BLOCK_SIZE, H_KV, D)
    v_blocks = v_pool.view(NUM_BLOCKS, BLOCK_SIZE, H_KV, D)
    kg = k_blocks[tables].flatten(1, 2)  # [b, S, H_kv, D]
    vg = v_blocks[tables].flatten(1, 2)
    qg = q.view(b, H_KV, groups, D)
    scores = torch.einsum("bkgd,bskd->bkgs", qg.float(), kg.float()) * D**-0.5
    pad = torch.arange(kg.shape[1], device=q.device)[None, :] >= ctx_lens[:, None]
    scores.masked_fill_(pad[:, None, None, :], float("-inf"))
    probs = scores.softmax(-1)
    out = torch.einsum("bkgs,bskd->bkgd", probs, vg.float())
    return out.reshape(b, H, D)


@pytest.mark.parametrize("ctx_lens_list", [[1], [15], [16], [17], [100], [1, 15, 16, 17, 100, 333, 512, 7]])
def test_triton_matches_einsum(ctx_lens_list):
    torch.manual_seed(0)
    b = len(ctx_lens_list)
    ctx_lens = torch.tensor(ctx_lens_list, dtype=torch.long, device="cuda")
    max_blocks = -(-max(ctx_lens_list) // BLOCK_SIZE)

    k_pool = torch.randn(NUM_BLOCKS * BLOCK_SIZE, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_pool = torch.randn_like(k_pool)
    q = torch.randn(b, H, D, dtype=torch.bfloat16, device="cuda")
    # random DISTINCT blocks per sequence, 0-padded like the runner builds them
    tables = torch.zeros(b, max_blocks, dtype=torch.long, device="cuda")
    for i, ctx in enumerate(ctx_lens_list):
        n = -(-ctx // BLOCK_SIZE)
        tables[i, :n] = torch.randperm(NUM_BLOCKS, device="cuda")[:n]

    got = triton_paged_decode(q, k_pool, v_pool, tables, ctx_lens, BLOCK_SIZE)
    ref = einsum_reference(q, k_pool, v_pool, tables, ctx_lens)
    assert torch.allclose(got.float(), ref, atol=2e-2, rtol=2e-2)
