from torch.utils.data import Dataset
from rabbit.tokenizer import Tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(self, ds, tokenizer: Tokenizer, seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if index >= len(self.ds):
            return IndexError("Index out of bounds")

        text = self.ds[index]["text"]

        text_tokens = self.tokenizer.encode(text)

        if len(text_tokens) > self.seq_len:
            text_tokens = text_tokens[: self.seq_len + 1]
        else:
            padding_length = self.seq_len - len(text_tokens)
            text_tokens = text_tokens + (padding_length + 1) * [self.tokenizer.pad_id]

        return {
            "input_sequences": text_tokens[:-1],
            "output_sequences": text_tokens[1:],
        }


if __name__ == "__main__":
    print("hello")
