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

"""vocab_size=13
max_seq_len=29
in_size = 256
out_size = 256
head_size = 128
num_heads = 2
dropout_p = 0.5
fc_hidden_size = 128
num_layers = 4

gpt = GPT(vocab_size,
        max_seq_len,
        in_size,
        out_size,
        head_size,
        num_heads,
        dropout_p,
        fc_hidden_size,
        num_layers)


criterion = nn.CrossEntropyLoss()
x = torch.randint(3, 13, (64, 17))
trg = torch.randint(3, 13, (64, 17))
output = gpt(x).transpose(1,2)
print(output)
loss = criterion(output, trg)
print(loss)"""
