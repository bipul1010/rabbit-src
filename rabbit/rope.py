import torch
import torch.nn as nn
from torch import bfloat16


def precompute_theta_pos(dim: int, seq_len: int, theta_base: float = 10000):
    theta_pow = torch.arange(0, dim, 2)  ##head_dim/2

    theta_vector = 1 / (theta_base ** (theta_pow / dim))  ##head_dim/2

    m = torch.arange(0, seq_len)  ##seq

    prod = torch.outer(m, theta_vector).float()  ##(seq,head_dim/2)

    ##(seq,head_dim/2)
    return torch.polar(torch.ones(*prod.shape), prod)


def apply_rotary_embed(x: torch.Tensor, freq_cis: torch.Tensor):
    x_ = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  ## (b,seq,h,head_dim) -> (b,seq,h,head_dim/2,2)
    x_ = torch.view_as_complex(x_)  ##(b,seq,h,head_dim/2,2) -> (b,seq,h,head_dim/2)

    # (b,seq,h,head_dim/2) * (1,seq,1,head_dim/2) -> (b,seq,h,head_dim/2)
    x_ = x_ * freq_cis.unsqueeze(0).unsqueeze(2)

    # (b,seq,h,head_dim/2) -> (b,seq,h,head_dim/2,2)
    x_out = torch.view_as_real(x_)

    # (b,seq,h,head_dim)
    return x_out.reshape(*x.shape).type_as(x)


if __name__ == "__main__":
    torch.manual_seed(1000)
    dim, seq_len = 8, 4
    freq_cis = precompute_theta_pos(dim=dim, seq_len=seq_len)
    print(freq_cis, freq_cis.shape)

    x = torch.randn(2, seq_len, dim).to(dtype=bfloat16)
    print(x, x.shape)
    y = apply_rotary_embed(x, freq_cis=freq_cis)
    print(y, y.shape, y.dtype)
    # y = y.reshape(2, seq_len, 2, -1)
    # print(y, y.shape, y.dtype)
