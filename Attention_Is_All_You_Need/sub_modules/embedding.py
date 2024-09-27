import torch
import torch.nn as nn

class Language_Embedding(nn.Module):
    def __init__(self, en_vocab_size, hi_vocab_size, d_model):
        super().__init__()
        self.en_embeddings = nn.Embedding(en_vocab_size, d_model)
        self.hi_embeddings = nn.Embedding(hi_vocab_size, d_model)

    def forward(self, en_tokens, hi_tokens):

        en_embed = self.en_embeddings(en_tokens.to(self.hi_embeddings.weight.device))
        hi_embed = self.hi_embeddings(hi_tokens.to(self.hi_embeddings.weight.device))
        
        return en_embed, hi_embed
        