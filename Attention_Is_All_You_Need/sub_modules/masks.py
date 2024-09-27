import torch

def get_masks(dataset, en_batch, hi_batch):
        
        max_sequence_length = dataset.max_sequence_length
        num_of_sentences = len(en_batch)
        
        nil = -1e9
        
        decoder_self_attention_mask = torch.full([num_of_sentences, max_sequence_length, max_sequence_length] , nil)
        decoder_self_attention_mask = torch.triu(decoder_self_attention_mask , diagonal = 1)
        
        encoder_self_attention_mask = torch.full([num_of_sentences, max_sequence_length, max_sequence_length], 0.0)
        encoder_decoder_attention_mask = torch.full([num_of_sentences, max_sequence_length, max_sequence_length], 0.0)
        
        for index in range(num_of_sentences):
            num_of_en_tokens = (en_batch[index] != dataset.english_to_index[dataset.PADDING_TOKEN]).sum().item()
            num_of_hi_tokens = (hi_batch[index] != dataset.hindi_to_index[dataset.PADDING_TOKEN]).sum().item() 
            
            encoder_self_attention_mask[index, num_of_en_tokens:, : ] = nil
            encoder_self_attention_mask[index, :, num_of_en_tokens: ] = nil
            
            encoder_decoder_attention_mask[index, num_of_hi_tokens: , : ] = nil
            encoder_decoder_attention_mask[index, : , num_of_en_tokens: ] = nil
            
            
            ## can be commented out
            decoder_self_attention_mask[index, num_of_hi_tokens:, :] = nil
            
        return decoder_self_attention_mask,encoder_self_attention_mask, encoder_decoder_attention_mask