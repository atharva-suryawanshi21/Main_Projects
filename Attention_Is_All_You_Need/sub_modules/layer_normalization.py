import torch
import torch.nn as nn

class Layer_Normalization(nn.Module):
    def __init__(self,d_model, epsilon = 1e-5 ):
        super(Layer_Normalization, self).__init__()
        self.epsilon = epsilon
        self.d_model = d_model
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        
    def forward(self, input):
        var, mean = torch.var_mean(input, dim= -1, unbiased=False, keepdim=True)
        std = (var+self.epsilon).sqrt()
        
        normalized = (input - mean)/std
        
        out = self.gamma * normalized + self.beta
        return out

        

if __name__ == "__main__":
    input = torch.tensor([[
        [0.2,0.1,0.3],
        [0.5,0.1,0.1]
    ]])
        
    model = Layer_Normalization(3, 1e-5)
    output = model(input)
    print(output)