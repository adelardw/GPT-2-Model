import torch
import torch.nn as nn
from multiheadattention import MultiheadAttention
from encoder import TransformerEncoderBlock
from posencoding import PositionalEncoding
from trilmask import tril_mask



class TransformerDecoderBlock(nn.Module):

    def __init__(self, in_size,
                       out_size,
                       head_size,
                       num_heads,
                       dropout_p,
                       fc_hidden_size,
                       encoder_out_size):
        
        super().__init__()
        
        self.encoder_out_size = encoder_out_size
        self.in_size = in_size
        self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size
        

        self.masked_attention = MultiheadAttention(self.in_size,
                                                   None,
                                                   self.out_size,
                                                   self.head_size,
                                                   self.num_heads)
        
        self.adapt_residual = nn.Linear(self.in_size, self.out_size, bias=False) if self.in_size != self.out_size \
                              else nn.Identity()
        
        self.norm = nn.LayerNorm(self.out_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.encoder_block = TransformerEncoderBlock(in_size=self.encoder_out_size,
                                                     query_in_size=self.out_size,
                                                     out_size=self.out_size,
                                                     head_size=head_size,
                                                     num_heads=self.num_heads,
                                                     dropout_p=self.dropout_p,
                                                     fc_hidden_size=self.fc_hidden_size)
    
    def forward(self, encoder_output, input):
        
        mask = tril_mask(input)

        masked_attention = self.masked_attention(input,input,input,mask)
        norm = self.dropout(self.norm(masked_attention + self.adapt_residual(input)))
        encoded = self.encoder_block(query=norm, key=encoder_output, value=encoder_output)


        return encoded
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size,
                       max_seq_len,
                       in_size,
                       out_size,
                       head_size,
                       num_heads,
                       dropout_p,
                       fc_hidden_size,
                       encoder_out_size,
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
        self.encoder_out_size = encoder_out_size if encoder_out_size is not None else out_size


        self.embedding = nn.Embedding(self.vocab_size, self.in_size)
        self.pe = PositionalEncoding(self.max_seq_len, self.in_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.decoder_blocks = nn.ModuleDict(
            {f"decoder block{i}": TransformerDecoderBlock(in_size=self.in_size if i==0 else self.out_size,
                                                           out_size=self.out_size,
                                                           head_size=self.head_size,
                                                           num_heads=self.num_heads,
                                                           dropout_p=self.dropout_p,
                                                           fc_hidden_size=self.fc_hidden_size,
                                                           encoder_out_size=self.encoder_out_size)
              for i in range(self.num_layers)})


        self.out = nn.Linear(self.out_size, self.vocab_size)

    

    def forward(self, encoder_output, input):

        embedded = self.embedding(input)
        pe = self.pe(embedded)
        out = self.dropout(pe)

        for block in self.decoder_blocks.values():
            out = block(encoder_output, out) 

        return self.out(out)
    


