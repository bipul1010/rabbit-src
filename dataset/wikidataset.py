from torch.utils.data import Dataset
from rabbit.tokenizer import Tokenizer


class WikipediaDataset(Dataset):
    def __init__(self, ds, tokenizer: Tokenizer, seq_len: int, shift_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shift_len = shift_len

    def _expand_with_padding(self, tokenized_list: list):
        if len(tokenized_list) < self.seq_len:
            padding_len = self.seq_len - len(tokenized_list)
            tokenized_list = tokenized_list + [self.tokenizer.pad_id] * padding_len

        return tokenized_list

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if index >= len(self.ds):
            return IndexError("Index out of bounds")

        text = self.ds[index]
        text_tokens = self.tokenizer.encode(s=text)
        start_indices = [i for i in range(0, len(text_tokens), self.shift_len)]

        input_sequences = []
        output_sequences = []
        for start_idx in start_indices:
            input_tokens = self._expand_with_padding(
                text_tokens[start_idx : start_idx + self.seq_len]
            )
            output_tokens = self._expand_with_padding(
                text_tokens[start_idx + 1 : start_idx + 1 + self.seq_len]
            )
            input_sequences.append(input_tokens)
            output_sequences.append(output_tokens)
        return {
            "input_sequences": input_sequences,
            "output_sequences": output_sequences,
        }


if __name__ == "__main__":
    print("hello")
