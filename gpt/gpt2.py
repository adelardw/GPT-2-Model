import torch
import torch.nn as nn
import torch.nn.functional as F
from gptblock import GPTLayers
from collections import OrderedDict

class GPT(nn.Module):
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

        self.gpt_blocks = GPTLayers(vocab_size,
                                    max_seq_len,
                                    in_size,
                                    out_size,
                                    head_size,
                                    num_heads,
                                    dropout_p,
                                    fc_hidden_size,
                                    num_layers)
        
        self.norm = nn.LayerNorm(out_size)
        self.fc = nn.Sequential(OrderedDict(
                                    [('1', nn.Linear(out_size, fc_hidden_size)),
                                     ('2', nn.ReLU()),
                                     ('3',nn.Linear(fc_hidden_size, vocab_size))]))

    
    def proba(self, x):
        return F.softmax(self.forward(x), dim=-1)
    
    def forward(self, x):
        
        out = self.gpt_blocks(x)
        out = self.norm(out)
        
        return self.fc(out)


