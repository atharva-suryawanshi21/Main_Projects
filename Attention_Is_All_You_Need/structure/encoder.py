import torch
import torch.nn as nn

from sub_modules.self_attention import Multi_Headed_Attention
from sub_modules.layer_normalization import Layer_Normalization
from sub_modules.feed_forward import Feed_Forward_Module

class unit_encoder(nn.Module):
    def __init__(self, d_model, hidden_layers, num_heads,dropout_ff, dropout_att, activation):
        super(unit_encoder, self).__init__()
        
        self.attention_head = Multi_Headed_Attention( 
                                                 d_model = d_model,
                                                 num_heads=  num_heads,
                                                 dropout_ratio= dropout_att)
        self.layer_norm_1 = Layer_Normalization(d_model, epsilon= 1e-5)
        self.layer_norm_2 = Layer_Normalization(d_model, epsilon= 1e-5)
        self.ff = Feed_Forward_Module(d_model= d_model, hidden_layers= hidden_layers, drop_prob= dropout_ff, activation = activation)
        
        self.Dropout_1 = nn.Dropout(dropout_ff)
        self.Dropout_2 = nn.Dropout(dropout_ff)
        
    def forward(self, x, mask):
        residual_x = x
        x = self.attention_head(x, to_mask = mask)
        x = self.Dropout_1(x)
        x = x + residual_x
        x = self.layer_norm_1(x)
        
        residual_x = x
        x = self.ff(x)
        x = self.Dropout_2(x)
        x = residual_x + x
        x = self.layer_norm_2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_encoder_layers,  d_model, hidden_layers, num_heads, dropout_ff, dropout_att, activation):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([unit_encoder(d_model, hidden_layers, num_heads, dropout_ff, dropout_att, activation) 
                                     for _ in range(num_encoder_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

        
    
if __name__ == "__main__":
    batch_size = 30
    sequence_length = 50
    d_model = 512
    hidden_layers = 2048
    num_heads = 1
    dropout_ff = 0.1
    dropout_att = 0.1
    activation = 'silu'
    
    input = torch.rand(batch_size, sequence_length, d_model)
    num_encoder_layers = 3
    obj = Encoder(num_encoder_layers,  d_model, hidden_layers, num_heads,dropout_ff, dropout_att, activation)
    print(obj.forward(input).shape)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Example usage
    encoder = Encoder(num_encoder_layers=num_encoder_layers,  d_model=512, hidden_layers=2048, num_heads=8, dropout_ff=0.1, dropout_att=0.1, activation='relu')
    total_params = count_parameters(encoder)
    print(f'Total parameters: {total_params}')

        