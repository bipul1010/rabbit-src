import torch
from transformers import AutoTokenizer


mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    return mask


def memory_footpring_during_runtime(params):
    memory_bytes = params * 4
    memory_megabytes = memory_bytes / (1024**2)
    return memory_megabytes


def apply_chat_template(messages=[]):

    return mistral_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What should I look for in a therapist?"},
    ]
    print(apply_chat_template(messages))
