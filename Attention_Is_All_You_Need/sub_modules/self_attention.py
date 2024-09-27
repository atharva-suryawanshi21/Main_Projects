import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class Multi_Headed_Attention(nn.Module):
    def __init__(self, d_model,  num_heads, dropout_ratio):
        super(Multi_Headed_Attention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.dropout_ratio = dropout_ratio
   
        self.qkv_layer = nn.Linear(self.d_model, self.d_model *3)
        self.refine_layer = nn.Linear(self.d_model, self.d_model)
        self.Dropout = nn.Dropout(self.dropout_ratio)
            
        
    
    def get_modified_values(self, q, k, v, to_mask):
        d_k = q.size(-1)
        k_t = k.transpose(-2,-1)
        
        product = torch.matmul(q,k_t) / math.sqrt(d_k)
        if to_mask is not None:
            to_mask = to_mask.unsqueeze(1)  # Shape becomes [batch_size, 1, seq_len, seq_len]
            to_mask = to_mask.repeat(1, self.num_heads, 1, 1)  # Shape becomes [batch_size, num_heads, seq_len, seq_len]

            assert product.shape == to_mask.shape, f"{product.shape} {to_mask.shape}"
        
            product = product + to_mask
            
        # some heads may get dispropotionally high attention scores, so we apply dropout
        attention = torch.softmax(product, dim=-1)
        attention = self.Dropout(attention)
        values = torch.matmul(attention, v)
        
        batch_size = q.size(0)
        sentence_length = q.size(2) # sentence length is at 3rd index
        
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size, sentence_length, self.num_heads * self.head_dim)
        
        values = self.refine_layer(values)
        return attention, values
        
    def get_q_k_v(self, input):
        batch_size = input.size(0)
        sentence_length = input.size(1)
        
        qkv = self.qkv_layer(input)

        qkv = qkv.reshape(batch_size, sentence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        
        return qkv.chunk(3, dim = -1)

        
    def forward(self, input, to_mask ):
        q, k, v = self.get_q_k_v(input)
        attention, values = self.get_modified_values(q,k,v, to_mask)
        return values
        
    
        
        
if __name__ == "__main__":
    # Dummy data
    batch_size = 30
    sentence_length = 50
    d_model = 512
    num_heads = 8

    # Create a dummy input tensor
    input_tensor = torch.rand(batch_size, sentence_length, d_model)

    # Initialize the Attention module
    attention = Multi_Headed_Attention(d_model,  num_heads,dropout_ratio=0.1)


    print(attention.forward(input_tensor, to_mask = None).shape)