# -*- coding: utf-8 -*-

import os

import pytest
import torch
import triton

from fla.ops.common.utils import prepare_token_indices
from native_sparse_attention.ops.naive import naive_nsa_with_compression
from native_sparse_attention.ops.parallel import parallel_nsa_with_compression


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [256, 1024, 2000])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("window_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("scale", [0.1])
def test_parallel(
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

    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    g_cmp = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    block_counts = torch.randint(1, S + 1, (B, T, H), device='cuda')

    ref = naive_nsa_with_compression(
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
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg_cmp, g_cmp.grad = g_cmp.grad.clone(), None
    ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None
    if window_size > 0:
        ref_dg_swa, g_swa.grad = g_swa.grad.clone(), None
    '''
    tri = parallel_nsa_with_compression(
        q=q,
        k=k,
        v=v,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    '''
