import pandas as pd
from dataset.wikidataset import WikipediaDataset
from dataset.tinystories import TinyStoriesDataset
from dataset.dahoas_fine_tune import DahosDataset
from config import ModelArgs
from datasets import load_dataset
from torch.utils.data import DataLoader
from rabbit.tokenizer import Tokenizer
from pathlib import Path
from torch.utils.data import random_split
from utils import apply_chat_template


class Loader:
    def __init__(self, config: ModelArgs, tokenizer: Tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train = None
        self.validation = None
        self.test = None

    def get_bs(self, split):
        if split == "train":
            return self.config.train_batch_size
        elif split == "validation":
            return self.config.valid_batch_size
        else:
            return 1

    def wikipedia(self):
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")

        def collate_fun(batch):
            output = dict()
            for key in batch[0]:
                output[key] = []
                for item in batch:
                    output[key] += item[key]
            return output

        for split in ["train", "validation", "test"]:

            merged_texts = []
            text_in_each_section = []
            for x in ds[split]["text"]:
                if x.strip().split().count("=") == 2:
                    if len(text_in_each_section) > 0:
                        merged_texts.append("\n\n".join(text_in_each_section))
                    text_in_each_section = [x.strip()]
                else:
                    if x.strip():
                        text_in_each_section.append(x.strip())
            df_raw = pd.DataFrame(merged_texts, columns=["text"])
            new_ds = WikipediaDataset(
                df_raw["text"],
                tokenizer=self.tokenizer,
                seq_len=self.config.seq_len,
                shift_len=self.config.shift_len,
            )
            # print(f"Dataset Length: {len(df_raw)}")
            batch_size = 1 if split == "train" else self.get_bs(split)
            loader = DataLoader(
                new_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fun,
            )

            setattr(self, split, loader)
        return self

    def tinystories(self):
        ds = load_dataset("roneneldan/TinyStories")

        def collate_fun(batch):
            output = dict()
            for key in batch[0]:
                output[key] = []
                for item in batch:
                    output[key].append(item[key])
            return output

        for split in ["train", "validation"]:

            ds_subset_size = int(len(ds[split]) * 0.1)
            remaining_ds = len(ds[split]) - ds_subset_size

            ds_raw = random_split(ds[split], [ds_subset_size, remaining_ds])

            new_ds = TinyStoriesDataset(
                ds_raw[0], tokenizer=self.tokenizer, seq_len=self.config.seq_len
            )
            print(f"{split}: ", len(new_ds))
            loader = DataLoader(
                new_ds,
                batch_size=self.get_bs(split),
                shuffle=True,
                collate_fn=collate_fun,
            )
            setattr(self, split, loader)

        self.test = self.validation

        return self

    def dahoas(self):
        ds = load_dataset("Dahoas/static-hh")

        def collate_fun(batch):
            output = dict()
            for key in ["input_sequences", "output_sequences"]:
                output[key] = []
                for item in batch:
                    if item.get(key):
                        output[key].append(item[key])
            return output

        for split in ["train", "test"]:
            # new_ds = DahosDataset(
            #     ds[split], tokenizer=self.tokenizer, seq_len=self.config.seq_len
            # )
            print(f"{split}: {len(ds[split])}")

            new_ds = DahosDataset(
                ds[split], tokenizer=self.tokenizer, seq_len=self.config.seq_len
            )

            loader = DataLoader(
                new_ds,
                batch_size=self.get_bs(split),
                shuffle=True,
                collate_fn=collate_fun,
            )

            setattr(self, split, loader)

        self.validation = self.test
        return self


if __name__ == "__main__":
    import sys
    import torch
    from config import get_default_config

    k = get_default_config()
    tokenizer = Tokenizer(model_path=str(Path(".") / k.tokenizer_file))
    loader = Loader(config=k, tokenizer=tokenizer).dahoas()
    # sys.exit()
    # random_iter = torch.randint(0, len(loader.validation), (1,))

    # print(random_iter[0])
    # x = loader.validation[random_iter[0]]
    # for x in loader.train:
    batch_length = []
    for k in range(4):
        print(f"K:{k}")
        for x in loader.train:
            input_sequences = torch.tensor(x["input_sequences"])
            output_sequences = torch.tensor(x["output_sequences"])

            print(input_sequences.shape, output_sequences.shape)
            batch_length.append(input_sequences.shape[0])
            print("Input Seq==\n\n")
            print(input_sequences[-2, 0::200])
            print("Output Seq==\n\n")
            print(output_sequences[-2:, 0:200])

            print("Input Seq Decode==\n\n")
            print(tokenizer.decode(input_sequences[-2].tolist()))

            print("Output Seq Decode==\n\n")
            print(tokenizer.decode(output_sequences[-2].tolist()))
            break

    print(
        f"Max _length: {max(batch_length)} | Avg Length: {sum(batch_length) / len(batch_length)}"
    )
