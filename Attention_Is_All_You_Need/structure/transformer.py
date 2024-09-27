import torch
import torch.nn as nn

from structure.encoder import Encoder
from structure.Dataset import English_Hindi_Dataset
from structure.decoder import Decoder
from sub_modules.positional_encoding import Positional_Encoding
from sub_modules.embedding import Language_Embedding

class Transformer(nn.Module):
    def __init__(self, 
                 num_encoder_decoder_layers,
                 d_model,
                 sequence_length,
                 hidden_layers,
                 num_heads,
                 hi_voab_size,
                 dropout_ff,
                 dropout_attn,
                 activation = 'silu'
                 ):
        super().__init__()
        
        self.encoder = Encoder(num_encoder_layers=num_encoder_decoder_layers,
                               d_model=d_model,
                               hidden_layers=hidden_layers,
                               num_heads=num_heads,
                               dropout_ff=dropout_ff,
                               dropout_att=dropout_attn,
                               activation=activation)
        self.decoder = Decoder(num_decoder_layers=num_encoder_decoder_layers,
                               d_model=d_model,
                               hidden_layers=hidden_layers,
                               num_heads=num_heads,
                               dropout_ff=dropout_ff,
                               dropout_att=dropout_attn,
                               activation=activation)
        self.position_encoding = Positional_Encoding(sequence_length, d_model)
        
        self.last_layer = nn.Linear(d_model, hi_voab_size)
        
    def forward(self, x, y, dec_self, enc_self, enc_dec_cross):     
        # Encoder part
        x = self.position_encoding(x)
        x = self.encoder(x, enc_self)
        
        # Decoder part
        y = self.position_encoding(y)
        y = self.decoder(x, y, dec_self, enc_dec_cross)
        
        # last layers
        y = self.last_layer(y)
        return y
    


        
        
        
        
        