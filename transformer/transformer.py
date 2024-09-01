import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(
            self,
            max_seq_len,
            vocab_size,
            emb_size,
            
            num_encoder_layers,
            enc_att_out_size,
            enc_att_head_size,
            enc_num_heads,
            enc_ff_hidden_size,
            enc_dropout_p,
            
            num_decoder_layers,
            dec_att_out_size,
            dec_att_head_size,
            dec_num_heads,
            dec_ff_hidden_size,
            dec_dropout_p,
        ):
            super(Transformer, self).__init__()
            
            # Запишем все переданые гиперпараметры модели
            self.max_seq_len = max_seq_len
            self.vocab_size = vocab_size
            self.emb_size = emb_size
            
            self.num_encoder_layers = num_encoder_layers
            self.enc_att_out_size = enc_att_out_size
            self.enc_att_head_size = enc_att_head_size
            self.enc_num_heads = enc_num_heads
            self.enc_ff_hidden_size = enc_ff_hidden_size
            self.enc_dropout_p = enc_dropout_p
            
            self.num_decoder_layers = num_decoder_layers
            self.dec_att_out_size = dec_att_out_size
            self.dec_att_head_size = dec_att_head_size
            self.dec_num_heads = dec_num_heads
            self.dec_ff_hidden_size = dec_ff_hidden_size
            self.dec_dropout_p = dec_dropout_p

            # Encoder
            self.encoder = TransformerEncoder(
                max_seq_len=self.max_seq_len,
                vocab_size=self.vocab_size,
                in_size=self.emb_size,
                num_layers=self.num_encoder_layers,
                head_size=self.enc_att_head_size,
                num_heads=self.enc_num_heads,
                out_size=self.enc_att_out_size,
                fc_hidden_size=self.enc_ff_hidden_size,
                dropout_p=self.enc_dropout_p,
            )
            
            # Decoder
            self.decoder = TransformerDecoder(
                max_seq_len=self.max_seq_len,
                vocab_size=self.vocab_size,
                in_size=self.emb_size,
                num_layers=self.num_decoder_layers,
                head_size=self.dec_att_head_size,
                num_heads=self.dec_num_heads,
                out_size=self.dec_att_out_size,
                fc_hidden_size=self.dec_ff_hidden_size,
                dropout_p=self.dec_dropout_p,
                encoder_out_size=self.enc_att_out_size,
            )
        
    def forward(self, encoder_input, decoder_input):
        """
        Args:
            encoder_input: input to encoder 
            decoder_input: input to decoder
        out:
            out: final tensor with logits of each word in vocab
        """
        # Получаем на вход batch_size x enc_seq_len и batch_size x dec_seq_len
        encoder_output = self.encoder(encoder_input)  # (batch_size, enc_seq_len, enc_att_out_size)

        return self.decoder(encoder_output, decoder_input)

