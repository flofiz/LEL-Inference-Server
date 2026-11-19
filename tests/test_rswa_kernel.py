"""Smoke tests for vLLM 0.34 on CUDA, independent of Ray and model weights."""

import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def visible_keys(q, prefix, window):
    """Reference mask in absolute token coordinates (current token included)."""
    return [k for k in range(q + 1) if k < prefix or q - k < window]


def test_reference_mask_boundaries():
    assert visible_keys(q=3, prefix=37, window=2) == [0, 1, 2, 3]
    assert visible_keys(q=128, prefix=37, window=32) == [
        *range(37),
        *range(97, 129),
    ]


@pytest.fixture
def cuda_torch():
    if sys.platform != "linux":
        pytest.skip("The project installs vLLM on Linux only")
    import torch

    assert torch.cuda.is_available(), "CUDA is not available in the project environment"
    assert torch.cuda.is_bf16_supported(), "The test GPU must support BF16"
    return torch


def test_runtime_imports_and_bf16(cuda_torch, monkeypatch):
    torch = cuda_torch
    assert version("vllm") == "0.34.0"
    assert torch.__version__ == "2.14.0+cu130"
    assert version("torchvision") == "0.29.0+cu130"

    # In vLLM 0.34, CUDA ops moved from _C to _C_stable_libtorch.
    import vllm._C_stable_libtorch  # noqa: F401

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from main import VLLMPredictDeployment  # noqa: F401 — import without starting Ray

    x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(x @ x, torch.full_like(x, 16))


def run_attention_check(torch, *, seq_len, query_len, prefix, poison_gap=False):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    block_size, head_size, window = 32, 128, 32
    num_blocks = (seq_len + block_size - 1) // block_size
    rng = torch.Generator(device="cuda").manual_seed(42)
    opts = {"device": "cuda", "dtype": torch.bfloat16, "generator": rng}
    q = torch.randn((query_len, 2, head_size), **opts)
    k = torch.randn((num_blocks, block_size, 1, head_size), **opts)
    v = torch.randn(k.shape, **opts)
    out = torch.empty_like(q)
    scale = head_size**-0.5

    def int_tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    if poison_gap:
        # In the decode case block 2 covers positions 64..95: after the
        # prompt and outside the window of every valid query row.
        assert all(
            not set(range(64, 96)).intersection(visible_keys(pos, prefix, window))
            for pos in range(seq_len - query_len, seq_len)
        )
        v[2].fill_(float("nan"))

    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=int_tensor([0, query_len]),
        max_seqlen_q=query_len,
        seqused_k=int_tensor([seq_len]),
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=int_tensor([list(range(num_blocks))]),
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        rswa_prefix_lens=int_tensor([prefix]),
        rswa_window=window,
    )

    # Gather only visible keys: masked NaNs never enter the FP32 oracle.
    expected = []
    flat_k, flat_v = k.view(-1, head_size), v.view(-1, head_size)
    for query, pos in zip(q, range(seq_len - query_len, seq_len)):
        keys = visible_keys(pos, prefix, window)
        logits = (query.float() @ flat_k[keys].float().T) * scale
        expected.append(logits.softmax(dim=-1) @ flat_v[keys].float())

    assert torch.isfinite(out).all(), "RSWA attention returned NaN or Inf"
    torch.testing.assert_close(out.float(), torch.stack(expected), atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    ("seq_len", "query_len"), [(16, 16), (130, 1), (130, 2)],
    ids=["causal-prompt-chunk", "decode", "decode-multiple-queries"],
)
def test_rswa_triton_matches_reference(cuda_torch, seq_len, query_len):
    run_attention_check(cuda_torch, seq_len=seq_len, query_len=query_len, prefix=37)


def test_rswa_triton_recycled_gap(cuda_torch):
    run_attention_check(cuda_torch, seq_len=130, query_len=2, prefix=37, poison_gap=True)
