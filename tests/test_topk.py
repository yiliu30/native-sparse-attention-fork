# -*- coding: utf-8 -*-

import os

import pytest
import torch

from native_sparse_attention.ops.naive import naive_nsa_with_compression
from native_sparse_attention.ops.parallel import parallel_nsa_with_compression

@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [512, 1024, 2000])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("window_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("scale", [0.1])
def test_topk(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    S: int,
    block_size: int,
    window_size: int,
    dtype: torch.dtype,
    scale: float
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda')
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda')
    g_cmp = torch.rand((B, T, HQ), dtype=dtype, device='cuda')
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda')
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda')
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    block_counts = torch.randint(1, S + 1, (B, T, H), dtype=torch.long, device='cuda')

    ref, ref_topk = naive_nsa_with_compression(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale
    )

    tri, tri_topk = parallel_nsa_with_compression(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale
    )

    assert (ref_topk != tri_topk[:, :, :, :ref_topk.shape[-1]]).float().mean() < 0.02
