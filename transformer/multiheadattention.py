import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, in_size, query_in_size, out_size, head_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.query_in_size = query_in_size if query_in_size is not None else in_size
        self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads

        self.q_matrix = nn.Linear(self.query_in_size, self.head_size * self.num_heads, bias=False)
        self.k_matrix = nn.Linear(self.in_size, self.head_size * self.num_heads, bias=False)
        self.v_matrix = nn.Linear(self.in_size, self.head_size * self.num_heads, bias=False)

        self.out_matrix = nn.Linear(self.head_size * self.num_heads, self.out_size)

    
    def forward(self, query, key, value, mask = None):
        # input size -> (batch, max_seq_len, query in size or in size)

        batch_size = query.size(0)
        query_seq_len = query.size(1)
        seq_len = key.size(1)

        q = self.q_matrix(query).view(batch_size, query_seq_len, self.num_heads, self.head_size)
        k = self.k_matrix(key).view(batch_size, seq_len, self.num_heads, self.head_size)
        v = self.v_matrix(value).view(batch_size, seq_len, self.num_heads, self.head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q.size() -> (batch, num_heads, query_seq_len, head_sise)
        # k.size() -> (batch, num_heads, seq_len, head_size)
        relevance = q @ k.transpose(2, 3) / math.sqrt(self.head_size)

        if mask is not None:
            relevance = relevance.masked_fill(mask, -1e20)

        # relevance.size() -> (bacth, num_heads, query_seq_len, seq_len)
        # v.size() -> (bacth, num_heads, seq_len, head_size)

        relevance = F.softmax(relevance, dim=-1)

        head_i = relevance @ v

        # head_i.size() -> (batch, num_heads, query_seq_len, head_size)

        out = head_i.transpose(1, 2).reshape(batch_size, query_seq_len, self.num_heads*self.head_size)

        return self.out_matrix(out)
