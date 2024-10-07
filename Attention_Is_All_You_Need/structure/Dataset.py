import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class English_Hindi_Dataset(Dataset):
    def __init__(self, en_path, hi_path, num_of_sentences, max_sequence_length=200, read_max =  5_00_000 ):

        self.en_path = en_path
        self.hi_path = hi_path
        self.START_TOKEN = '<START>'
        self.PADDING_TOKEN = '<PADDING>'
        self.END_TOKEN = '<END>'
        self.en_vocab, self.hi_vocab = self.get_vocab()
        self.index_to_hindi, self.hindi_to_index, self.index_to_english, self.english_to_index = self.language_and_index_map()
        
        self.max_sequence_length = max_sequence_length
        self.num_of_sentences = num_of_sentences
        
        self.raw_en_data, self.raw_hi_data = self.read_files(read_max)
                        
        self.en_data_tokenized, self.hi_data_tokenized = self.process_data()
        
        del self.raw_en_data
        del self.raw_hi_data
        
    def __len__(self):
        return len(self.en_data_tokenized)
    
    def __getitem__(self, index):
        en_sentence = self.en_data_tokenized[index]
        hi_sentence = self.hi_data_tokenized[index]
        return en_sentence, hi_sentence
        
    def get_vocab(self):

        english_vocabulary = [self.START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                ':', '<', '=', '>', '?', '@', 
                                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                                'Y', 'Z',
                                '[', '\\', ']', '^', '_', '`', 
                                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                                'y', 'z', 
                                '{', '|', '}', '~', self.PADDING_TOKEN, self.END_TOKEN]

        hindi_vocabulary = [self.START_TOKEN, self.PADDING_TOKEN, self.END_TOKEN, ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',]

        # Adding Devanagari characters (vowels and consonants)
        for code in range(0x0900, 0x097F):  # Unicode range for Devanagari
            hindi_vocabulary.append(chr(code))

        punctuation = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', 
            ',', '-', '.', '/', ' ', ':', '<', '=', '>', '?', '@', 
            '[', '\\', ']', '^', '_', '`', '।', '“', '”', '{', '|', '}', '~'
        ]

        hindi_vocabulary.extend(punctuation) 
        english_vocabulary = sorted(set(english_vocabulary))
        hindi_vocabulary = sorted(set(hindi_vocabulary))

        print(f"Total unique characters: English-> {len(english_vocabulary)} Hindi-> {len(hindi_vocabulary)}")
        return english_vocabulary, hindi_vocabulary
    
    def language_and_index_map(self):
        index_to_hindi = {k:v for k,v in enumerate(self.hi_vocab)}
        hindi_to_index = {v:k for k,v in enumerate(self.hi_vocab)}
        index_to_english = {k:v for k,v in enumerate(self.en_vocab)}
        english_to_index = {v:k for k,v in enumerate(self.en_vocab)}
        return  index_to_hindi, hindi_to_index, index_to_english, english_to_index

        
    def valid_length(self, en, hi):
        return len(en) <= (self.max_sequence_length-2) and len(hi) <= (self.max_sequence_length-2)
        
    def valid_vocab(self, en, hi):
        en_vocab = set(self.en_vocab)
        hi_vocab = set(self.hi_vocab)
        
        for ch in en:
            if ch not in en_vocab:
                return False
        
        for ch in hi:
            if ch not in hi_vocab:
                return False
        
        return True
    
    def get_masks(self, en_batch, hi_batch):
        nil = -1e9

        
        decoder_self_attention_mask = torch.full([self.num_of_sentences, self.max_sequence_length, self.max_sequence_length] , nil)
        decoder_self_attention_mask = torch.triu(decoder_self_attention_mask , diagonal = 1)
        
        encoder_self_attention_mask = torch.full([self.num_of_sentences, self.max_sequence_length, self.max_sequence_length], 0.0)
        encoder_decoder_attention_mask = torch.full([self.num_of_sentences, self.max_sequence_length, self.max_sequence_length], 0.0)
        
        for index in range(self.num_of_sentences):
            num_of_en_tokens, num_of_hi_tokens = len(en_batch[index]), len(hi_batch[index])
            
            encoder_self_attention_mask[index, num_of_en_tokens:, : ] = nil
            encoder_self_attention_mask[index, :, num_of_en_tokens: ] = nil
            
            encoder_decoder_attention_mask[index, num_of_hi_tokens: , : ] = nil
            encoder_decoder_attention_mask[index, : , num_of_en_tokens: ] = nil
            
        return decoder_self_attention_mask,encoder_self_attention_mask, encoder_decoder_attention_mask
    
    
    def get_clean_data(self, en_data, hi_data):
        en_sentences = []
        hi_sentences = []
        total = 0
        
        for _ , (en, hi) in enumerate(zip(en_data, hi_data)):
            if self.valid_length(en, hi) and self.valid_vocab(en, hi):
                en_sentences.append(en)
                hi_sentences.append(hi)
                total += 1       

            if total == self.num_of_sentences:
                break
        
        return en_sentences, hi_sentences
    
    def tokenize(self, sentence, language_to_index, start_token=False, end_token=False):
        sentence_indices = [language_to_index[token] for token in list(sentence)]
        
        if start_token:
            sentence_indices.insert(0, language_to_index[self.START_TOKEN])
        if end_token:
            sentence_indices.append(language_to_index[self.END_TOKEN])
            
        while len(sentence_indices) < self.max_sequence_length:
            sentence_indices.append(language_to_index[self.PADDING_TOKEN])
                
        return torch.tensor(sentence_indices)   
    
    def untokenize(self, sentence, index_to_language):
        tokens = [index_to_language[index.item()] for index in sentence]
        
        if self.START_TOKEN in tokens:
            tokens.remove(self.START_TOKEN)
        if self.END_TOKEN in tokens:
            tokens.remove(self.END_TOKEN)
            
        tokens = [ chr for chr in tokens if chr != self.PADDING_TOKEN]
        
        sentence = ''.join(tokens)
        return sentence
        
    
    def read_files(self, read_max):
        
        en_data = []
        hi_data = []

        # Read the first 3 lines from the English file
        with open(self.en_path, 'r', encoding='utf-8') as file:
            for _ in range(read_max):
                line = file.readline()
                if not line:  # Stop if there are fewer than 3 lines
                    break
                en_data.append(line.strip())  # Optional: strip newline characters

        # Read the first 3 lines from the Hindi file
        with open(self.hi_path, 'r', encoding='utf-8') as file:
            for _ in range(read_max):
                line = file.readline()
                if not line:  # Stop if there are fewer than 3 lines
                    break
                hi_data.append(line.strip())  # Optional: strip newline characters

        return en_data, hi_data
             
    def process_data(self, tokenize = True):
        

        en_data = [sentence.rstrip() for sentence in self.raw_en_data]
        hi_data = [sentence.rstrip() for sentence in self.raw_hi_data]
        
        en_data, hi_data = self.get_clean_data(en_data, hi_data)

        print("\tDataset Cleaned")
        en_tokenized = [self.tokenize(sentence, self.english_to_index, start_token=False, end_token=False) for sentence in en_data]
        hi_tokenized = [self.tokenize(sentence, self.hindi_to_index, start_token=True, end_token=True) for sentence in hi_data]
        print("\tDataset Tokenized and Pading is Done")
        return en_tokenized, hi_tokenized

    # def update_dicts_for_translation(self,en_vocab, hi_vocab,en_to_index,index_to_en,hi_to_index,index_to_hi):      
    #     self.en_vocab = en_vocab
    #     self.hi_vocab = hi_vocab
        
    #     self.index_to_hindi = index_to_hi  
    #     self.hindi_to_index = hi_to_index 
    #     self.index_to_english = index_to_en 
    #     self.english_to_index = en_to_index
        
        
        
        
if __name__ == "__main__":
    dataset = English_Hindi_Dataset('Dataset/train.en/train.en', 
                                    'Dataset/train.hi/train.hi',
                                    num_of_sentences = 100) 
    print(dataset[0])
    # print(dataset[1])