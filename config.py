from pathlib import Path
from dataclasses import dataclass

from torch._C import device


@dataclass
class ModelArgs:

    vocab_size: int  # vocab size of the tokenizers
    train_batch_size: int  # Batch size
    valid_batch_size: int
    seq_len: int  # Sequence length
    shift_len: int  # Shift Length
    no_layers: int  # number of layers
    dim: int  # Size of the embedding vector
    n_heads: int  # No of heads
    n_kv_heads: int  # No of KV_Heads
    heads_dim: int  # Size of Embedding vector per head
    hidden_dim: int  # Hidden dimensions of the Feedforward
    moe: bool  # Enable/Disable Mixture of Experts
    no_experts: int  # No of experts
    no_experts_per_token: int  # No of experts per token
    model_folder: str  # Folder where to save the checkpoints
    model_basename: str  # Basename of the checkpoint files
    tokenizer_file: str  # Path where to save the tokenizer
    device: str  # Device
    local_rank: int = -1  # LOCAL_RANK assigned by torchrun
    global_rank: int = -1  # RANK assigned by torchrun
    base_path: str = ""  ##Base Path


def get_default_config() -> ModelArgs:

    return ModelArgs(
        vocab_size=48064,
        train_batch_size=1,
        valid_batch_size=1,
        seq_len=1024,
        shift_len=100,
        no_layers=16,
        dim=1024,
        n_heads=16,
        n_kv_heads=4,
        heads_dim=64,
        hidden_dim=3072,
        moe=True,
        no_experts=2,
        no_experts_per_token=1,
        model_folder="weights",
        model_basename="ckpt",
        tokenizer_file="tokenizer/tokenizer.model",
        base_path="/Users/bipulvaibhav/Documents/AI/openai/rabbit-src",
        device="cpu",
    )


def check_or_make_model_folder_path(args: ModelArgs):
    model_folder_path = Path(f"{args.base_path}/{args.model_folder}")
    if model_folder_path.exists() is False:
        model_folder_path.mkdir()


def get_ckpt_model_path(args: ModelArgs, iter: int):
    model_folder_path = Path(f"{args.base_path}/{args.model_folder}")
    return model_folder_path / Path(f"{args.model_basename}_{iter}.pt")


def get_latest_model_path(args: ModelArgs):
    model_folder_path = Path(f"{args.base_path}/{args.model_folder}")
    weight_files = list(model_folder_path.glob("*.pt"))

    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return weight_files[-1]


if __name__ == "__main__":
    args = get_default_config()
    # print(get_model_path(args=args, iter=10))
    print(get_latest_model_path(args=args))
