# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Union

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor,
    block_size: int = 64,
    scale: Optional[float] = None,
    head_first: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (torch.LongTensor):
            Block counts of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat([block_indices.new_tensor(range(0, B*T, T)), block_indices.new_tensor([B*T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b, s_b = q[i], k[i], v[i], block_indices[i], block_counts[i]
        else:
            T = cu_seqlens[i+1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b, s_b = map(
                lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]],
                (q, k, v, block_indices, block_counts)
            )

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [1, HQ]
            s_i = s_b[i_q]
            # [S*BS, HQ, -1]
            k_i, v_i = map(lambda x: x.gather(0, i_i.clamp(0, T-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn = torch.einsum('h d, n h d -> n h', q_i, k_i).masked_fill((i_i > i_q) | (c >= s_i), float('-inf')).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)
            else:
                o[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn, v_i)

    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o.to(dtype)

def compression(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    """
    Currently, we set mean pooling as our basic compression function.
    We pad the incomplete blocks during compression for consistency, but the incomplete blocks won't be used later.
    """
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, block_size, num_block, H, -1).mean(dim=1)
    v_cmp = v.view(B, block_size, num_block, H, -1).mean(dim=1)
    return k_cmp, v_cmp

def naive_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
) -> torch.LongTensor:
    B, T = q.shape[0], q.shape[1]
    H, HQ = k.shape[2], q.shape[2]
    G = HQ//H
    BS = block_size
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    k_cmp, v_cmp = compression(k, v, BS)
    C = k_cmp.shape[1]
    S = min(S, C)
    k_cmp, v_cmp = map(lambda x: repeat(x, 'b c h d -> b c (h g) d', g=G), (k_cmp, v_cmp))
    q, k_cmp, v_cmp = map(lambda x: x.float(), (q, k_cmp, v_cmp))

    casual_mask = ((torch.arange(T) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device)
    local_mask = (torch.arange(T)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)

    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    attn_cmp = attn_cmp.masked_fill(casual_mask, float('-inf'))
    attn_cmp = attn_cmp.softmax(-1)
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp).nan_to_num()
    attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
    attn_select = attn_select.view(B, H, G, T, C).sum(2)
    block_indices = attn_select.topk(S, -1)[1]

    block_indices = block_indices.masked_fill(block_indices > (block_indices.new_tensor(range(T))[:, None]//BS), 0)
    block_indices = block_indices.transpose(1, 2)
    return block_indices, o_cmp


def naive_nsa_with_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        raise NotImplementedError("Variable-length mode is not supported for naive NSA with compression")
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        if not isinstance(block_counts, int):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    block_indices, o_cmp = naive_nsa_compression(q, k, v, block_counts, block_size, scale)
    o = naive_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
        head_first=head_first,
        cu_seqlens=cu_seqlens
    )
 
    return o + o_cmp
