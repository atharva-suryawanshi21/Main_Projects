import torch
import torch.nn as nn

class Positional_Encoding(nn.Module):
    def __init__(self, sequence_length, d_model):
        super(Positional_Encoding, self).__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model # no of embeddings
    
    def get_positional_encodings(self):
        
        positions = torch.arange(0, self.sequence_length, 1, dtype=torch.float32).unsqueeze(1)
        
        # according to paper, we need even numbers to get denominator values
        even_numbers = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        denominator = torch.pow(10_000, even_numbers / self.d_model )
        
        positional_encoding = torch.zeros(self.sequence_length, self.d_model)
        for index, pos in enumerate(positions):
            for i in range(self.d_model):
                if i%2 == 0:
                    positional_encoding[index][i] = torch.sin(pos / denominator[i//2])
                else:
                    positional_encoding[index][i] = torch.cos(pos / denominator[i//2])
                    
        return positional_encoding
        
    def forward(self, input): 
        embeddings = self.get_positional_encodings().to(input.device)
        assert input.shape[-2:] == embeddings.shape, f"Shapes of input {input.shape}, embeddings {embeddings.shape}"
        return  input + embeddings


            