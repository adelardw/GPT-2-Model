import torch

def tril_mask(embedding):
    batch_size, seq_len, _ = embedding.size()

    mask = torch.tril(torch.ones(seq_len,seq_len)).expand(batch_size, 1,seq_len,seq_len).bool()

    return mask