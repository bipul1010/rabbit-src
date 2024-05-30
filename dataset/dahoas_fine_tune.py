import re
from torch.utils.data import Dataset
from rabbit.tokenizer import Tokenizer
from utils import apply_chat_template


class DahosDataset(Dataset):
    def __init__(self, ds, tokenizer: Tokenizer, seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def get_messages_for_chat_template(self, text):
        segments = re.split(r"(Human:|Assistant:)", text)
        messages = []

        # Process each segment
        for i in range(1, len(segments), 2):
            role = segments[i].strip().lower()
            content = segments[i + 1].strip()

            if role.find("human") != -1:
                role = "user"
            elif role.find("assistant") != -1:
                role = "assistant"

            messages.append({"role": role, "content": content})

        return messages

    def __getitem__(self, index):
        if index >= len(self.ds):
            return IndexError("Index out of bounds")

        prompt_text = self.ds[index]["prompt"]
        prompt_response = self.ds[index]["response"]

        messages = self.get_messages_for_chat_template(text=prompt_text)
        if prompt_response.strip():
            messages[-1]["content"] = prompt_response

        try:
            chat_text = apply_chat_template(messages=messages)
        except:
            return {}

        encoded_tokens = self.tokenizer.encode(chat_text)

        if len(encoded_tokens) > self.seq_len:
            text_tokens = encoded_tokens[: self.seq_len + 1]
        else:
            padding_length = self.seq_len - len(encoded_tokens)
            text_tokens = encoded_tokens + [self.tokenizer.pad_id] * (
                padding_length + 1
            )

        return {
            "input_sequences": text_tokens[:-1],
            "output_sequences": text_tokens[1:],
        }


if __name__ == "__main__":
    print("hello")
