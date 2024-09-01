import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer.posencoding import PositionalEncoding
from transformer.multiheadattention import MultiheadAttention
from transformer.trilmask import tril_mask
from collections import OrderedDict


class GPTBlock(nn.Module):

    def __init__(self, in_size,
                       query_in_size,
                       out_size,
                       head_size,
                       num_heads,
                       dropout_p,
                       fc_hidden_size):
        
        super().__init__()
        
        self.in_size = in_size
        self.query_in_size = query_in_size if query_in_size is not None else in_size
        self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size
        

        self.attention = MultiheadAttention(self.in_size,
                                            self.query_in_size,
                                            self.out_size,
                                            self.head_size,
                                            self.num_heads)
        
        self.adapt_residual = nn.Linear(self.query_in_size, self.out_size, bias=False) if self.in_size != self.out_size \
                              else nn.Identity()
        
        self.norm1 = nn.LayerNorm(self.out_size)
        self.dropout1 = nn.Dropout(self.dropout_p)

        self.feed_forward = nn.Sequential(OrderedDict(
                                    [('1', nn.Linear(self.out_size, self.fc_hidden_size)),
                                     ('2', nn.GELU()),
                                     ('3',nn.Linear(self.fc_hidden_size, self.out_size))]))
        
        self.dropout2 = nn.Dropout(self.dropout_p)

    
    def forward(self, query, key, value):
        
        mask = tril_mask(key)
        attention = self.attention(query, key, value, mask)
        res_attention = self.dropout1(attention) + self.adapt_residual(query)
        norm = self.norm1(res_attention)
        ff = self.feed_forward(norm)

        return self.dropout2(ff + norm)


class GPTLayers(nn.Module):

    def __init__(self, vocab_size,
                       max_seq_len,
                       in_size,
                       out_size,
                       head_size,
                       num_heads,
                       dropout_p,
                       fc_hidden_size,
                       num_layers):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.in_size = in_size
        self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size


        self.embedding = nn.Embedding(self.vocab_size, self.in_size)
        self.pe = PositionalEncoding(self.max_seq_len, self.in_size)

        self.encoder_blocks = nn.ModuleDict(
            {f'encoder block {i}': GPTBlock(in_size= self.in_size if i==0 else self.out_size,
                                            out_size=self.out_size,
                                            head_size=self.head_size,
                                            num_heads=self.num_heads,
                                            dropout_p=self.dropout_p,
                                            fc_hidden_size=self.fc_hidden_size,
                                            query_in_size=None)
                                            for i in range(self.num_layers)})
        
    def forward(self, x):
        embed = self.embedding(x)
        out = self.pe(embed)

        for block in self.encoder_blocks.values():
            out = block(out, out, out)
        

        return out
    



