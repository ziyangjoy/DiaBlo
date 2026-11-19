"""CPU sanity tests for the DiaBlo layer, merge and checkpoint filtering.

Run from the repo root:  python -m pytest -q tests
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diablo import (BlockLinear, BlockAdapter, replace_linear_with_blocklinear,
                    replace_blocklinear_with_linear)
from diablo.trainer import adapter_state_dict


class Tiny(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.other = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x):
        return self.other(self.q_proj(x))


# (in, out, N) incl. non-divisible shapes and GQA-like k/v (out << in)
SHAPES = [(64, 64, 8), (64, 16, 8), (96, 160, 8), (50, 30, 7), (1536, 256, 64)]


@pytest.mark.parametrize("d_in,d_out,n", SHAPES)
def test_zero_init_is_identity(d_in, d_out, n):
    torch.manual_seed(0)
    m = Tiny(d_in, d_out)
    x = torch.randn(2, 5, d_in)
    ref = m(x)
    replace_linear_with_blocklinear(m, num_blocks=n, target_modules=["q_proj"])
    assert isinstance(m.q_proj, BlockLinear) and isinstance(m.other, nn.Linear)
    assert torch.allclose(m(x), ref, atol=1e-6)


@pytest.mark.parametrize("d_in,d_out,n", SHAPES)
def test_merge_matches_adapter_forward(d_in, d_out, n):
    torch.manual_seed(0)
    m = Tiny(d_in, d_out)
    replace_linear_with_blocklinear(m, num_blocks=n, target_modules=["q_proj"])
    m.q_proj.block_A.data.normal_()
    m.eval()
    x = torch.randn(3, 4, d_in)
    y_adapter = m(x)
    y2d = m.q_proj(x[0])  # 2-D input path
    replace_blocklinear_with_linear(m)
    assert isinstance(m.q_proj, nn.Linear)
    assert torch.allclose(m(x), y_adapter, atol=1e-4, rtol=1e-4)
    assert torch.allclose(m.q_proj(x[0]), y2d, atol=1e-4, rtol=1e-4)


def test_update_is_block_diagonal():
    """Merged delta W must be zero outside the diagonal blocks."""
    d, n = 32, 4
    m = Tiny(d, d)
    w0 = m.q_proj.weight.data.clone()
    replace_linear_with_blocklinear(m, num_blocks=n, target_modules=["q_proj"])
    m.q_proj.block_A.data.normal_()
    replace_blocklinear_with_linear(m)
    delta = (m.q_proj.weight.data - w0).T  # (in, out)
    b = d // n
    mask = torch.block_diag(*[torch.ones(b, b) for _ in range(n)]).bool()
    assert delta[~mask].abs().max() == 0
    assert delta[mask].abs().min() > 0


def test_block_adapter_equals_block_linear():
    torch.manual_seed(0)
    base = nn.Linear(96, 160, bias=False)
    bl = BlockLinear(96, 160, num_blocks=8)
    bl.linear.weight.data = base.weight.data.clone()
    ba = BlockAdapter(base, num_blocks=8)
    A = torch.randn_like(bl.block_A)
    bl.block_A.data, ba.block_A.data = A.clone(), A.clone()
    bl.eval(), ba.eval()
    x = torch.randn(2, 3, 96)
    assert torch.allclose(bl(x), ba(x), atol=1e-6)


def test_param_count_formula():
    # trainable params per layer ~= in*out/N when divisible
    bl = BlockLinear(1536, 8960, num_blocks=64)
    assert bl.block_A.numel() == 1536 * 8960 // 64


def test_adapter_state_dict_filter():
    sd = {"model.layers.0.self_attn.q_proj.block_A": 1,
          "model.layers.0.self_attn.q_proj.linear.weight": 2,
          "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": 3,
          "model.embed_tokens.weight": 4}
    assert set(adapter_state_dict(sd)) == {
        "model.layers.0.self_attn.q_proj.block_A",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"}


def test_replacement_reuses_frozen_layer_without_copy():
    """The wrapper keeps the original nn.Linear (no copy, bias kept); block_A is zero in the layer's dtype."""
    lin = nn.Linear(48, 40, bias=True).half()

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = lin

    h = Holder()
    replace_linear_with_blocklinear(h, num_blocks=8, target_modules=["q_proj"])
    assert h.q_proj.linear is lin and h.q_proj.linear.bias is not None
    assert h.q_proj.block_A.dtype == torch.float16 and h.q_proj.block_A.device == lin.weight.device
    assert h.q_proj.block_A.abs().sum() == 0 and not h.q_proj.block_A.is_meta


def test_param_count_matches_paper_table10():
    """Table 10 (Llama3.2-3B) lists N=64 -> 27.56M, N=8 -> 220.50M: q,k,v,up,down in units of 2^20."""
    hidden, inter, kv, layers = 3072, 8192, 1024, 28
    shapes = [(hidden, hidden), (hidden, kv), (hidden, kv), (hidden, inter), (inter, hidden)]   # q k v up down
    for n, paper in [(64, 27.56), (8, 220.50), (512, 3.45)]:
        with torch.device("meta"):
            per_layer = sum(BlockLinear(i, o, num_blocks=n).block_A.numel() for i, o in shapes)
        assert round(per_layer * layers / 2**20, 2) == paper


def test_eval_prompts_are_training_prefixes():
    from diablo.prompts import (commonsense_eval_prompt, commonsense_train_prompt, metamath_eval_prompt,
                                metamath_train_prompt)
    q = "What is 2+2?"
    assert metamath_train_prompt(q, "4. The answer is: 4").startswith(metamath_eval_prompt(q))
    assert metamath_train_prompt(q, "X")[len(metamath_eval_prompt(q)):].startswith("X")
    assert not metamath_train_prompt(q, "X").startswith(metamath_eval_prompt(q, "upstream"))   # the upstream mismatch
    assert commonsense_train_prompt(q, "", "the correct answer is true").startswith(commonsense_eval_prompt(q))


def _tokenizer(vocab, **special):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel(vocab, unk_token=special.get("unk_token", "<x>")))
    return PreTrainedTokenizerFast(tokenizer_object=tk, **special)


def test_setup_pad_token_never_grows_vocab():
    from diablo import setup_pad_token
    llama2 = _tokenizer({"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3}, unk_token="<unk>", eos_token="</s>")
    setup_pad_token(llama2)
    assert llama2.pad_token_id == 0 and len(llama2) == 4
    llama3 = _tokenizer({"<|end_of_text|>": 0, "a": 1}, eos_token="<|end_of_text|>")
    setup_pad_token(llama3)
    assert llama3.pad_token_id == llama3.eos_token_id and len(llama3) == 2
