from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelArgs


class MoE(nn.Module):
    def __init__(
        self, experts: nn.ModuleList, gate: nn.Module, args: ModelArgs
    ) -> None:
        super().__init__()

        self.gate = gate
        self.experts = experts
        self.args = args

    def forward(self, x: torch.Tensor):
        # x - (b,seq,dim)

        gate_logits = self.gate(x)  # (b,seq,dim) -> (b,seq,no_experts)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.no_experts_per_token
        )  # weights,selected_experts -> (b,seq,no_experts_per_token)
        weights = F.softmax(weights, dim=2)
        output = torch.zeros_like(x)  # (b,seq,dim)
        for i, expert in enumerate(self.experts):
            batch_idx, token_idx, nth_expert = torch.where(selected_experts == i)
            output[batch_idx, token_idx] += weights[
                batch_idx, token_idx, nth_expert, None
            ] * expert(x[batch_idx, token_idx])

        return output


if __name__ == "__main__":

    from rabbit.model import FeedForward
    from config import get_default_config

    torch.manual_seed(1000)
    args = get_default_config()
    args.train_batch_size = 2
    args.seq_len = 8
    x = torch.rand(args.train_batch_size, args.seq_len, args.dim)
    print(x, x.shape)

    experts = [FeedForward(args) for i in range(args.no_experts)]

    gate = nn.Linear(args.dim, args.no_experts)

    moe = MoE(experts=experts, gate=gate, args=args)

    output = moe(x)
    print(output, output.shape)
