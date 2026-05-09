"""Correctness gate: our pure-PyTorch Qwen must match HF transformers.

1. Prefill logits allclose (bf16 tolerance).
2. 50-token greedy continuation exact-match, generated token-by-token through
   our NaiveKVBackend (exercises the decode path + causal offset mask).
"""

import pytest
import torch

MODEL_DIR = "models_cache/qwen2.5-0.5b-instruct"
PROMPT = "The key idea behind paged attention in LLM serving is"


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from garuda.models.qwen import QwenForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="bfloat16").cuda().eval()
    ours = QwenForCausalLM.load(MODEL_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    return hf, ours, ids


def test_prefill_logits_close(setup):
    from garuda.models.qwen import NaiveKVBackend

    hf, ours, ids = setup
    with torch.inference_mode():
        ref = hf(ids).logits[0]
        n = ids.shape[1]
        got = ours(ids[0], torch.arange(n, device="cuda"), NaiveKVBackend(ours.cfg))
    # bf16 accumulation-order differences: compare top-1 agreement + magnitude
    assert (ref.argmax(-1) == got.argmax(-1)).float().mean().item() > 0.99
    assert torch.allclose(ref.float(), got.float(), atol=1.0, rtol=0.05)


def test_greedy_50_tokens_exact(setup):
    from garuda.models.qwen import NaiveKVBackend

    hf, ours, ids = setup
    with torch.inference_mode():
        ref = hf.generate(ids, max_new_tokens=50, do_sample=False)[0, ids.shape[1] :]

        backend = NaiveKVBackend(ours.cfg)
        n = ids.shape[1]
        logits = ours(ids[0], torch.arange(n, device="cuda"), backend)
        out = []
        tok = logits[-1].argmax()
        for i in range(50):
            out.append(tok.item())
            logits = ours(tok.view(1), torch.tensor([n + i], device="cuda"), backend)
            tok = logits[-1].argmax()
    assert out == ref.tolist()
