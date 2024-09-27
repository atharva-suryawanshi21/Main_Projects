import torch 
import torch.nn as nn

from sub_modules.self_attention import Multi_Headed_Attention
from sub_modules.cross_attention import Multi_Head_Cross_Attention
from sub_modules.feed_forward import Feed_Forward_Module
from sub_modules.layer_normalization import Layer_Normalization

class UnitDecoder(nn.Module):
    def __init__(self,d_model, hidden_layers, num_heads,dropout_ff, dropout_att, activation ):
        super(UnitDecoder, self).__init__()
        
        self.self_attention = Multi_Headed_Attention( d_model,  num_heads, dropout_att)
        self.feed_forward = Feed_Forward_Module(d_model, hidden_layers, dropout_ff, activation )
        self.layer_norm_1 = Layer_Normalization(d_model, epsilon= 1e-5)
        self.layer_norm_2 = Layer_Normalization(d_model, epsilon= 1e-5)
        self.layer_norm_3 = Layer_Normalization(d_model, epsilon= 1e-5)
        self.cross_attention = Multi_Head_Cross_Attention( d_model,  num_heads, dropout_att)
        
        self.Dropout_1 = nn.Dropout(dropout_ff)
        self.Dropout_2 = nn.Dropout(dropout_ff)
        self.Dropout_3 = nn.Dropout(dropout_ff)
        
    def forward(self, x, y, mask_ds, mask_en_ds):
        residual_y = y
        y = self.self_attention(y, to_mask = mask_ds)
        y = self.Dropout_1(y)
        y = residual_y + y
        y = self.layer_norm_1(y)
        
        residual_y = y
        y = self.cross_attention(x, y, to_mask =mask_en_ds )
        y = self.Dropout_2(y)
        y = residual_y + y
        y = self.layer_norm_2(y)
        
        residual_y = y
        y = self.feed_forward(y)
        y = self.Dropout_3(y)
        y = residual_y + y
        y = self.layer_norm_3(y)
        
        return y
    
class Sequential_for_Decoder(nn.Sequential):
    def forward(self, x, y, mask_ds, mask_en_ds):
        
        for module in self._modules.values():
            y = module(x, y, mask_ds, mask_en_ds)
        
        return y
    
    
class Decoder(nn.Module):
    def __init__(self, num_decoder_layers,  d_model, hidden_layers, num_heads,dropout_ff, dropout_att, activation ):
        super(Decoder, self).__init__()
        
        self.decoder = Sequential_for_Decoder(
            *[UnitDecoder(d_model, hidden_layers, num_heads,dropout_ff, dropout_att, activation ) for _ in range(num_decoder_layers)]
        )
        
    
    def forward(self, x, y, mask_ds, mask_en_ds):
        y = self.decoder(x, y, mask_ds, mask_en_ds)
        return y
        
       
        
        
        
        