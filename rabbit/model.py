import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
from config import ModelArgs
from typing import Optional
from rabbit.rope import apply_rotary_embed, precompute_theta_pos
from rabbit.cache import Cache
from rabbit.moe import MoE


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        ## (b,seq) -> (b,seq,dim)
        return self.embed(x)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w3 = nn.Linear(args.hidden_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.wt_gain = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        ## (b,seq,dim) -> (b,seq,dim)
        output = self.norm(x.float()).type_as(x)
        return output * self.wt_gain


class Attention(nn.Module):
    def __init__(self, modelargs: ModelArgs) -> None:
        super().__init__()

        self.n_heads = modelargs.n_heads
        self.n_kv_heads = modelargs.n_kv_heads
        self.heads_dim = modelargs.heads_dim

        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            modelargs.dim, modelargs.n_heads * modelargs.heads_dim, bias=False
        )
        self.wk = nn.Linear(
            modelargs.dim, modelargs.n_kv_heads * modelargs.heads_dim, bias=False
        )
        self.wv = nn.Linear(
            modelargs.dim, modelargs.n_kv_heads * modelargs.heads_dim, bias=False
        )

        self.wo = nn.Linear(
            modelargs.n_heads * modelargs.heads_dim, modelargs.dim, bias=False
        )
        self.cache = Cache(args=modelargs)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        input_pos: int,
        is_cache: Optional[bool] = False,
        mask: Optional[bool] = False,
    ):

        batch, seq_len, _ = x.shape
        q = self.wq(x)  # (b,seq,dim) - > (b,seq,n_heads*head_dim)
        k = self.wk(x)  # (b,seq,dim) - > (b,seq,n_kv_heads*head_dim)
        v = self.wv(x)  # (b,seq,dim) - > (b,seq,n_kv_heads*head_dim)

        q = q.reshape(batch, seq_len, self.n_heads, self.heads_dim)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.heads_dim)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.heads_dim)

        q = apply_rotary_embed(q, freq_cis=freq_cis)  # (b,seq_len,n_heads,head_dim)
        k = apply_rotary_embed(k, freq_cis=freq_cis)  # (b,seq_len,n_kv_heads,head_dim)

        q, k, v = map(
            lambda x: x.transpose(1, 2), (q, k, v)
        )  # (b,n_heads(q)/n_kv_heads(k,v),seq_len,head_dim)

        if is_cache is True:
            k, v = self.cache.update(input_pos=input_pos, k=k, v=v)

        k, v = map(
            lambda x: torch.repeat_interleave(x, repeats=self.repeats, dim=1), (k, v)
        )

        if mask is True:
            attn_output = F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=0.0, is_causal=True, attn_mask=None
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=0.0, is_causal=False, attn_mask=None
            )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.n_heads * self.heads_dim)
        )

        return self.wo(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

        # self.feedforward: nn.Module
        if args.moe is True:
            experts = nn.ModuleList([FeedForward(args) for i in range(args.no_experts)])
            gate = nn.Linear(args.dim, args.no_experts)
            self.feedforward = MoE(experts=experts, gate=gate, args=args)
        else:
            self.feedforward = FeedForward(args)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        input_pos: int,
        is_cache: Optional[bool] = False,
        mask: Optional[bool] = False,
    ):

        # x - > (b,seq,dim)

        # 1st residual connection
        h = x + self.attention(
            x=self.attention_norm(x),
            freq_cis=freq_cis,
            mask=mask,
            input_pos=input_pos,
            is_cache=is_cache,
        )

        # 2nd residual connection
        output = h + self.feedforward(self.ffn_norm(h))
        return output


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.no_layers)]
        )
        self.norm = RMSNorm(args.dim)

        self.projection_layer = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.token_embedding.embed.weight = (
            self.projection_layer.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.freq_cis = precompute_theta_pos(
            dim=args.heads_dim, seq_len=args.seq_len * 2
        ).to(args.device)
        self.args = args
        self.apply(self._init_weights_)

    def _init_weights_(self, module):
        # for p in module.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.args.device == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
        return optimizer

    @property
    def n_params(self):
        params = sum([p.numel() for p in self.parameters()])
        return params

    @property
    def n_trainable_params(self):
        params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return params

    def _clear_cache(self):
        for layer in self.layers:
            layer.attention.cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        input_pos: int = 0,
        is_cache: Optional[bool] = False,
        mask: Optional[bool] = False,
        targets: torch.Tensor = None,
        pad_token_id: int = -1,
    ):
        h = self.token_embedding(x)  # (b,seq) - > (b,seq,dim)

        freq_cis = self.freq_cis[input_pos : input_pos + h.shape[1], :]

        for layer in self.layers:
            h = layer.forward(
                x=h,
                freq_cis=freq_cis,
                mask=mask,
                is_cache=is_cache,
                input_pos=input_pos,
            )

        h = self.norm(h)
        if targets is not None:
            # (b,seq,dim) -> (b,seq,vocab_size)
            logits = self.projection_layer(h)
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(
                logits, targets, label_smoothing=0.1, ignore_index=pad_token_id
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.projection_layer(
                h[:, [-1], :]
            )  # (b,seq,dim) -> (b,1,vocab_size)
            loss = None

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_length: int = -1,
        temperature=1.0,
        top_k: int = None,
        pad_token_id=-1,
        eos_token_id=-1,
    ):
        # x = (b,seq), batch of list of tokens

        _, seq_len = prompt_tokens.shape
        if seq_len >= self.args.seq_len:
            raise ValueError(
                f"Input Seq should contain less than {self.args.seq_len} tokens"
            )

        self._clear_cache()

        input_pos = 0
        temperature = max(temperature, 0.1)
        n_max_tokens = (
            min(max_length, self.args.seq_len) if max_length >= 0 else self.args.seq_len
        )
        mask = True

        x = prompt_tokens.clone()
        while prompt_tokens.shape[1] < n_max_tokens:

            logits, _ = self.forward(
                x=x,
                input_pos=input_pos,
                is_cache=True,
                mask=mask,
                pad_token_id=pad_token_id,
            )  # (b,1,vocab)

            logits = logits[:, -1, :] / temperature  # (b,vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))  # v- > (b,topk)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)  # (b,vocab)
            next_token = torch.multinomial(probs, num_samples=1)  # (b,1)
            next_token_id = next_token[:, 0][
                0
            ]  ##just assuming as of now we are getting one batch.

            ##update parameters
            x = next_token.to(self.args.device)
            input_pos = prompt_tokens.shape[1]
            prompt_tokens = torch.cat([prompt_tokens, next_token], dim=1)
            mask = False
            if next_token_id == eos_token_id:
                break

        return prompt_tokens


if __name__ == "__main__":

    torch.manual_seed(1000)
    import sys
    from config import get_default_config
    from utils import generate_square_subsequent_mask, memory_footpring_during_runtime
    from rabbit.rope import precompute_theta_pos
    from rabbit.tokenizer import Tokenizer
    from pathlib import Path

    args = get_default_config()
    model = Transformer(args=args)
    print(model)
    print(model.n_params, model.n_trainable_params)
    print(memory_footpring_during_runtime(model.n_params))
    sys.exit()
    args.no_layers = 2
    args.dim, args.n_heads = 48, 8
    args.n_kv_heads = args.n_heads // 4
    args.heads_dim = args.dim // args.n_heads
    args.hidden_dim = args.dim * 4

    tokenizer = Tokenizer(model_path=str(Path("./tokenizer/tokenizer.model")))
    # args.vocab_size = tokenizer.n_words

    prompt = [
        "How are you, all ok?",
        "all well?",
        "ok great",
        "i am doing really good",
    ]

    x = [tokenizer.encode(x) for x in prompt]
    print(x)
    seq_len = max([len(j) for j in x])
    print(seq_len)
    x = [
        j + (seq_len - len(j)) * [tokenizer.pad_id] if len(j) < seq_len else j
        for j in x
    ]
    x = torch.tensor(x)
    print(x, x.shape)
    args.train_batch_size = x.shape[0]
    args.seq_len = x.shape[1]

    # # Expand to fit the attention mask dimensions

    attn_mask = generate_square_subsequent_mask(size=x.shape[1])
    attn_mask = attn_mask.expand(x.shape[0], -1, -1)
    key_padding_mask = x == tokenizer.pad_id
    key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
    print(key_padding_mask)
    print(attn_mask, attn_mask.shape)
    attn_mask = attn_mask.masked_fill(key_padding_mask == True, float("-inf"))
    print(attn_mask, attn_mask.shape)
    attn_mask = attn_mask.unsqueeze(1)
    # # attention = Attention(modelargs=args)
    # # # x = torch.rand(args.train_batch_size, args.seq_len, args.dim)
    # # # print(x.shape)
    # freq_cis = precompute_theta_pos(args.heads_dim, args.seq_len)
    # print(freq_cis.shape)

    print("==============")
    model = Transformer(args=args)
    for idx, layer in enumerate(model.layers):
        print(f"Layer: {idx}")
        print(layer.attention.cache.k_cache)
        print(layer.attention.cache.v_cache.shape)
    # print(model.parameters())
    # for pn, p in model.named_parameters():
    #     print(pn, p, p.shape, p.dim())
    print(model)
    print(model.n_params, model.n_trainable_params)
    # print(
    #     f"Approximate memory footprint (MB):{memory_footpring_during_runtime(model.n_params)}"
    # )

    logits, loss = model.forward(
        x=x, input_pos=0, is_cache=False, mask=None, pad_token_id=tokenizer.pad_id
    )
    print(logits, logits.shape, loss)

    print("BEfore=======")
    for idx, layer in enumerate(model.layers):
        print(f"Layer: {idx}")
        print(layer.attention.cache.k_cache)
        print(layer.attention.cache.v_cache.shape)

    print("After=======")
    for idx, layer in enumerate(model.layers):
        layer.attention.cache.clear()

    print("After=======")
    for idx, layer in enumerate(model.layers):
        print(f"Layer: {idx}")
        print(layer.attention.cache.k_cache)
        print(layer.attention.cache.v_cache.shape)
# ff = FeedForward(args=modelargs)
# ff_output = ff(output)
# print(ff_output, ff_output.shape)
