import torch
import torch.nn as nn
from config import ModelArgs


class Cache(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        cache_shape = (
            args.train_batch_size,
            args.n_kv_heads,
            args.seq_len * 2,
            args.heads_dim,
        )

        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(self, input_pos: int, k: torch.Tensor, v: torch.Tensor):

        batch_size, _, seq_len, _ = k.shape
        self.k_cache[:batch_size, :, input_pos : input_pos + seq_len] = k
        self.v_cache[:batch_size, :, input_pos : input_pos + seq_len] = v

        return (
            self.k_cache[:batch_size, :, : input_pos + seq_len],
            self.v_cache[:batch_size, :, : input_pos + seq_len],
        )

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()


if __name__ == "__main__":
    from config import get_default_config

    args = get_default_config()
    args.no_layers = 2
    args.seq_len = 10
    args.dim, args.n_heads = 48, 8
    args.n_kv_heads = args.n_heads // 4
    args.heads_dim = args.dim // args.n_heads
    args.hidden_dim = args.dim * 4
    cache = Cache(args=args)

    x = torch.rand(1, args.n_kv_heads, args.seq_len // 2, args.heads_dim)
    print(x.shape)

    input_pos = 0
    k, v = cache.update(input_pos=input_pos, k=x, v=x)
    print(f"KV Cache: {k}", k.shape)

    y = torch.rand(1, args.n_kv_heads, 1, args.heads_dim)
    print(y, y.shape)
    k, v = cache.update(input_pos=x.shape[-2], k=y, v=y)
    print(f"KV Cache: {v}", v.shape)
