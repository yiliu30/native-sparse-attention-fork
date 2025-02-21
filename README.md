<div align="center">

# 🐳 Native Sparse Attention

[![arxiv](https://img.shields.io/badge/arXiv-2502.11089-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.11089)

</div>

Efficient Triton implementations for [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089).

<div align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/ace2920d-3894-4556-8039-b70861742551">
</div>

## News

- [2024/02/21] We support a variable number of selected blocks for queries across different positions and batches.

## Usage

```py
from native_sparse_attention.ops.parallel import parallel_nsa

B, T, H, HQ, D = 4, 2048, 4, 64, 64
block_size = 64
q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
# randomly generated block indices
indices = torch.full((B, T, H, S), T, dtype=torch.long, device='cuda')
s = torch.randint(1, S + 1, (B, T, H), device='cuda')
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            indices[b, t, h, :len(i_i)] = i_i
indices = indices.sort(-1)[0]

parallel_nsa(
  q=q,
  k=k,
  v=v,
  indices=indices,
  s=s,
  block_size=block_size
)

# variable-length inputs are supported as well
# randomly split the sequence into N segments
N, T = 4, 2048
offsets = torch.cat([
    torch.tensor([0], dtype=torch.long),
    torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
    torch.tensor([T], dtype=torch.long)
], 0).cuda().sort()[0]
# seq-first required for inputs with variable lengths
q = torch.randn((1, T, HQ, D), dtype=dtype, device='cuda').requires_grad_()
k = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
v = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()

indices = torch.full((1, T, H, S), T, dtype=torch.long, device='cuda')
s = torch.randint(1, S + 1, (B, T, H), device='cuda')
seq_indices = prepare_token_indices(offsets).tolist()
for i in range(T):
    _, t = seq_indices[i]
    for h in range(H):
        i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
        indices[0, i, h, :len(i_i)] = i_i
indices = indices.sort(-1)[0]
parallel_nsa(
    q=q,
    k=k,
    v=v,
    indices=indices,
    s=s,
    block_size=block_size,
    cu_seqlens=offsets
)
```

## Benchmarks

```sh
Performance:
         T        nsa     nsa_bwd      flash   flash_bwd
0    128.0   0.116224    0.561968   0.019552    0.123888
1    256.0   0.216896    0.963808   0.041472    0.223840
2    512.0   0.414688    1.951680   0.093168    0.486176
3   1024.0   0.813952    4.039584   0.260000    1.252896
4   2048.0   1.672784    9.081648   0.855856    3.794176
5   4096.0   3.518624   19.852303   3.196768   12.965824
6   8192.0   7.535328   43.620705  12.336976   47.652878
7  16384.0  16.107521  102.203011  48.110847  186.464386
```
<div align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/efc25313-b058-47ae-b96e-ed67c62c134d">
</div>

## Citations

```bibtex
@inproceedings{Yuan2025NativeSA,
    title   = {Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention},
    author  = {Jingyang Yuan and Huazuo Gao and Damai Dai and Junyu Luo and Liang Zhao and Zhengyan Zhang and Zhenda Xie and Y. X. Wei and Lean Wang and Zhiping Xiao and Yuqing Wang and Chong Ruan and Ming Zhang and Wenfeng Liang and Wangding Zeng},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276408911}
}
```
