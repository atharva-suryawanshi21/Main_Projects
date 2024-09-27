import torch
import torch.nn as nn

class Feed_Forward_Module(nn.Module):
    def __init__(self, d_model, hidden_layers, drop_prob, activation):
        super(Feed_Forward_Module, self).__init__()

        self.d_model = d_model
        self.hidden_layers = hidden_layers
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.linear_1 = nn.Linear(self.d_model, self.hidden_layers)
        self.linear_2 = nn.Linear(self.hidden_layers, self.d_model)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, input):
        output = self.linear_1(input)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear_2(output)
        return output


if __name__ == "__main__":
    
    input = torch.rand(30,50,512)
    ff = Feed_Forward_Module(512, 2048, 0.1,'silu')
    print(ff.forward(input).shape)
    # print(input.shape)