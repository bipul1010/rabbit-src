import torch


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    return mask


def memory_footpring_during_runtime(params):
    memory_bytes = params * 4
    memory_megabytes = memory_bytes / (1024**2)
    return memory_megabytes
