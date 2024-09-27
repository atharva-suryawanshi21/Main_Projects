import torch 
import torch.nn as nn
import math

class Multi_Head_Cross_Attention(nn.Module):
    def __init__(self,  d_model,  num_heads, dropout_ratio):
        super(Multi_Head_Cross_Attention, self).__init__()
        
        self.d_model =  d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.dropout_ratio = dropout_ratio
        
        self.kv_layer = nn.Linear(self.d_model, 2*self.d_model)
        self.q_layer = nn.Linear(self.d_model, self.d_model)
        self.refine_layer = nn.Linear(self.d_model, self.d_model)
        self.Dropout = nn.Dropout(self.dropout_ratio)
    

        
    def get_modified_values(self, q, k, v, to_mask):
        d_k = q.size(-1)
        k_t = k.transpose(-2,-1)
        
        batch_size = q.size(0)
        sentence_length = q.size(2) # sentence length is at 3rd index
        
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
        
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size, sentence_length, self.num_heads * self.head_dim)
        
        values = self.refine_layer(values)
        return attention, values
    
    def forward(self, x, y, to_mask = None):
        batch_size = x.size(0)
        sentence_length = x.size(1) # sentence length is at 3rd index
        
        kv = self.kv_layer(x) # from encoder
        q = self.q_layer(y) # from decoder
        
        kv = kv.reshape(batch_size, sentence_length, self.num_heads, 2*self.head_dim)
        q = q.reshape(batch_size, sentence_length, self.num_heads, self.head_dim)       
        kv = kv.permute(0,2,1,3)
        q = q.permute(0,2,1,3)
        
        k,v = kv.chunk(2, dim = -1)
        attention, values = self.get_modified_values(q,k,v, to_mask)
        return values
    
    
if __name__ == "__main__":
    input_x = torch.rand(30,50,512)
    input_y = torch.rand(30,50,512)
    att = Multi_Head_Cross_Attention(512,8,0.1)
    
    print(att(input_x, input_y, to_mask = None).shape)
    
        
        
        
        