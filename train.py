import math
import random
from pathlib import Path
from torch._C import BenchmarkConfig
from torch.serialization import load
from torch.utils.data import DataLoader
import wandb
import torch
import torch.nn.functional as F
from config import ModelArgs
from rabbit.model import Transformer
from rabbit.tokenizer import Tokenizer
from dataset.loader import Loader
from config import (
    get_ckpt_model_path,
    get_latest_model_path,
    check_or_make_model_folder_path,
)
from tqdm import tqdm
from contextlib import nullcontext


class TrainArgs:
    start_from = "scratch"
    checkpoint = -1
    training_dataset = "tinystories"
    learning_rate = 6e-4
    min_lr = 6e-5
    warmup_iters = 2000
    lr_decay_iters = 60000
    weight_decay = 1e-1
    epochs = 30
    is_lr_flexible = True
    eval_iter = 1
    mini_batch_size = 4
    num_mini_batches_for_train = 4
    compile = False


def get_lr(global_iter, train_args: TrainArgs):
    if global_iter < train_args.warmup_iters:
        return train_args.learning_rate * (global_iter / train_args.warmup_iters)
    elif global_iter > train_args.lr_decay_iters:
        return train_args.min_lr
    else:
        decay_ratio = (global_iter - train_args.warmup_iters) / (
            train_args.lr_decay_iters - train_args.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return train_args.min_lr + coeff * (
            train_args.learning_rate - train_args.min_lr
        )


def create_attn_mask(attn_mask: torch.Tensor, x: torch.Tensor, pad_token_id=-1):
    batch_size, seq_len = x.shape
    attn_mask = attn_mask.unsqueeze(0).expand(
        batch_size, -1, -1
    )  # (seq,seq) -> (b,seq,seq)
    if pad_token_id >= 0:
        key_padding_mask = x == pad_token_id  # (b,seq)
        key_padding_mask = key_padding_mask.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (b,seq)- > (b,seq,seq)
        attn_mask = attn_mask.masked_fill(key_padding_mask == True, float("-inf"))
    return attn_mask


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    train_losses: list,
    val_dataloader: DataLoader,
    eval_iter: int,
    device: str,
    pad_token_id=-1,
    autocast=None,
):
    model.eval()
    avg_train_loss = (
        sum(train_losses) / len(train_losses) if len(train_losses) > 0 else None
    )

    val_idx = 0
    val_losses = []

    for val_batch in val_dataloader:
        if val_idx >= eval_iter:
            break
        # x,y -> (b,seq)
        x, y = torch.tensor(val_batch["input_sequences"]), torch.tensor(
            val_batch["output_sequences"]
        )

        x = x[:6].to(device)
        y = y[:6].to(device)

        with autocast:
            _, loss = model.forward(
                x=x, targets=y, mask=True, pad_token_id=pad_token_id, is_cache=False
            )
        val_losses.append(loss.item())
        val_idx += 1
    model.train()
    avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else None
    return {"train": round(avg_train_loss, 2), "val": round(avg_val_loss, 2)}


def view_predicted_tokens(
    model: Transformer,
    test_dataloader: DataLoader,
    device: str,
    tokenizer: Tokenizer,
    print_msg=None,
    no_prints=1,
):
    line_seperation = "+" * 100
    print_idx = 0
    for test_batch in test_dataloader:
        if print_idx >= no_prints:
            break
        x = torch.tensor(test_batch["input_sequences"])

        randidx = torch.randint(0, x.shape[0], (1,))
        random_test_tokens = x[randidx].to(device)  # (b=1,seq)

        _, pad_idx = torch.where(random_test_tokens == tokenizer.pad_id)
        if pad_idx.numel() != 0:
            random_test_tokens = random_test_tokens[:, : pad_idx[0]]

        length_sampler = torch.randint(0, random_test_tokens.shape[1], (1,))
        prompt_tokens, targeted_tokens = (
            random_test_tokens[:, :length_sampler],
            random_test_tokens[:, length_sampler:],
        )

        predicted_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            temperature=0,
            top_k=None,
            pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id,
            max_length=1024,
        )
        print_idx += 1

        print_msg(f"--------------------PrintIDx: {print_idx}-----------------")
        print_msg(line_seperation + "\n")
        print_msg(
            f"Prompt Tokens: {tokenizer.decode(prompt_tokens[0,:].detach().cpu().tolist())}"
        )
        print_msg(line_seperation + "\n")
        print_msg(
            f"Targeted Tokens: {tokenizer.decode(targeted_tokens[0,:].detach().cpu().tolist())}"
        )
        print_msg(line_seperation + "\n")
        print_msg(
            f"Predicted Tokens: {tokenizer.decode(predicted_tokens[0,prompt_tokens.shape[1]:].detach().cpu().tolist())}"
        )


def train(
    args: ModelArgs,
    train_args: TrainArgs,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    args.device = device
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    precision_auto_ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    tokenizer = Tokenizer(model_path=str(Path(f"./{args.tokenizer_file}")))
    args.vocab_size = tokenizer.n_words
    loader = Loader(config=args, tokenizer=tokenizer)
    if train_args.training_dataset == "wikipedia":
        dataloader = loader.wikipedia()
    elif train_args.training_dataset == "tinystories":
        dataloader = loader.tinystories()

    train_dataloader, val_dataloader = dataloader.train, dataloader.validation
    val_dataloader_batches = []

    for idx, x in enumerate(val_dataloader):
        if idx >= 5:
            break
        val_dataloader_batches.append(x)

    val_dataloader = val_dataloader_batches

    initial_epoch = 0
    global_iter = 0
    model = Transformer(args=args)
    model.to(args.device)
    optimizer = model.configure_optimizers(
        weight_decay=train_args.weight_decay,
        learning_rate=train_args.learning_rate,
    )
    check_or_make_model_folder_path(args=args)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if train_args.compile:
        print("Compiling Model ... (takes a ~minute)")
        model = torch.compile(model)

    if train_args.start_from == "scratch":
        print("Initializing a new model from scratch")
    else:
        if train_args.start_from == "checkpoint":
            model_path = get_ckpt_model_path(args=args, iter=train_args.checkpoint)
            if model_path.exists() is False:
                raise FileNotFoundError(
                    f"Trained file with checkpoint {train_args.checkpoint} doesn't exist in Folder: {args.model_folder}"
                )
        elif train_args.start_from == "resume":
            model_path = get_latest_model_path(args=args)
            if model_path is None:
                raise FileNotFoundError(
                    f"No trained file .pt exist in folder: {args.model_folder}"
                )
        else:
            raise ValueError("Train args has to be one of [scratch,resume,checkpoint]")

        model_path = str(model_path)
        print(f"Loading existing trained model: {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_epoch = checkpoint["epoch"] + 1
        global_iter = checkpoint["global_iter"] + 1
        del checkpoint

    for epoch in range(initial_epoch, train_args.epochs):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoc: {epoch}")
        losses = []
        # index = 0
        for training_batch in batch_iterator:
            # x,y -> (b,seq)
            # index += 1
            # if index >= 4:
            #     break
            input_x, target_y = torch.tensor(
                training_batch["input_sequences"]
            ), torch.tensor(training_batch["output_sequences"])

            n_mini_batches = input_x.shape[0] // train_args.mini_batch_size
            shuffled_batch_idx = random.sample(
                list(range(n_mini_batches)),
                min(train_args.num_mini_batches_for_train, n_mini_batches),
            )
            # print(
            #     f"Total Batches: {input_x.shape[0]} | No mini batches: {n_mini_batches} | {shuffled_batch_idx}"
            # )
            for idx in shuffled_batch_idx:
                start_idx, end_idx = (
                    train_args.mini_batch_size * idx,
                    train_args.mini_batch_size * (idx + 1),
                )
                # print(f"Idx: {idx} | StartIdx: {start_idx} | EndIdx: {end_idx}")
                x = input_x[start_idx:end_idx].to(device)
                y = target_y[start_idx:end_idx].to(device)

                if x.size(0) == 0 or y.size(0) == 0:
                    continue  # Skip if there are no data points to process

                # print(x, y)

                if train_args.is_lr_flexible:
                    lr = get_lr(global_iter=global_iter, train_args=train_args)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                with precision_auto_ctx:
                    _, loss = model.forward(
                        x=x,
                        targets=y,
                        mask=True,
                        pad_token_id=tokenizer.pad_id,
                        is_cache=False,
                    )
                scaler.scale(loss).backward()
                batch_iterator.set_postfix({"loss": loss.item()})
                losses.append(loss.item())

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_iter += 1

            if global_iter % 1000 == 0:
                ##check some predicted tokens , how the model is learning.
                view_predicted_tokens(
                    model=model,
                    test_dataloader=dataloader.test,
                    device=device,
                    tokenizer=tokenizer,
                    print_msg=lambda msg: batch_iterator.write(msg),
                )

        train_val_losses = estimate_loss(
            model=model,
            train_losses=losses,
            val_dataloader=val_dataloader,
            eval_iter=train_args.eval_iter,
            device=device,
            pad_token_id=tokenizer.pad_id,
            autocast=precision_auto_ctx,
        )

        batch_iterator.write(
            f"Epoch:{epoch} | Global Iteration: {global_iter} | Train Loss: {train_val_losses['train']} | Val Loss: {train_val_losses['val']}"
        )

        ## save the model
        model_filename = get_ckpt_model_path(args=args, iter=epoch)
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_iter": global_iter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_filename,
            )


if __name__ == "__main__":
    from config import get_default_config

    args = get_default_config()
    train_args = TrainArgs()

    train(args=args, train_args=train_args)
