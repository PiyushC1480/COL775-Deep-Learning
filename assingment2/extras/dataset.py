import torch
import nltk
from nltk import word_tokenize
import json
import pickle
import re


import os
from torch.utils.data import Dataset, DataLoader
from utils import *

"""
TextToMath converter:

eg:
given problem , generate linear formula and answer using Seq2Seq model

"Problem": "the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?",
"linear_formula": "multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|multiply(#2,const_100)|divide(#3,#1)|",
"answer": 400

"""

class TextToMathDataset(Dataset):
    def __init__(self, file_path, data_prefix = "train"):
        self.file_path = file_path
        #data given in json file
        print("\n")
        pth = os.path.join(self.file_path, f"{data_prefix}.json")
        
        print(pth)
        self.data  = json.load(open(pth))
        print("Dataset Length =", len(self.data))

        with open(os.path.join(file_path, "encoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.encoder_vocab = vocab
        
        with open(os.path.join(file_path, "decoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.decoder_vocab = vocab

        with open(os.path.join(file_path, "encoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "encoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)

        self.en_word2idx = word2idx
        self.en_idx2word = idx2word

        with open(os.path.join(file_path, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.de_word2idx = word2idx
        self.de_idx2word = idx2word

        print("Encoder Vocab Size = {}, Decoder Vocab Size = {}".format(len(self.en_word2idx), len(self.de_word2idx)))

    def __len__(self):  
        return len(self.data)
    

    def __getitem__(self, index):
        problem = self.data[index]["Problem"]
        problem = ["<sos>"] + tokenize_problem(problem) + ["<eos>"]

        linear_formula = self.data[index]["linear_formula"]
        formula_split = ["<sos>"] +tokenize_formula(linear_formula) + ["<eos>"]
        
        answer = {"problem" : problem , "linear_formula" : formula_split, "answer" : self.data[index]["answer"]}
        return answer
    

def collate(batch):
    max_len_problem = max([len(sample['problem']) for sample in batch])
    max_len_formula = max([len(sample['linear_formula']) for sample in batch])
    
    problem_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_problem = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    
    formula_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_formula = torch.zeros((len(batch), max_len_formula), dtype=torch.long)
    
    answers = torch.zeros(len(batch), dtype=torch.long)
    for i, sample in enumerate(batch):
        problem = sample['problem']
        formula = sample['linear_formula']
        answer = sample['answer']
        
        problem_lens[i] = len(problem)
        padded_problem[i, :len(problem)] = torch.LongTensor(problem)
        
        formula_lens[i] = len(formula)
        padded_formula[i, :len(formula)] = torch.LongTensor(formula)
        
        answers[i] = answer

    ret = {'problem': padded_problem, 'problem_lens': problem_lens, 'linear_formula': padded_formula, 'formula_lens': formula_lens, 'answer': answers}

    return ret



class Text2MathBertDataset(Dataset):
    def __init__(self, file_path, data_prefix = "train"):
        self.file_path = file_path
        pth = os.path.join(self.file_path, f"{data_prefix}.json")
        
        print(pth)
        self.data  = json.load(open(pth))
        print("Dataset Length =", len(self.data))
        
        with open(os.path.join(file_path, "decoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.decoder_vocab = vocab
        
        with open(os.path.join(file_path, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.de_word2idx = word2idx
        self.de_idx2word = idx2word

        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")

        print("Encoder Vocab Size = , Decoder Vocab Size = {}".format( len(self.de_word2idx)))
        
    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, index):
        problem = self.data[index]["Problem"]
        problem = self.en_tokenizer.encode(problem)
        
        linear_formula = self.data[index]["linear_formula"]
        formula_split = ["<sos>"] +tokenize_formula(linear_formula) + ["<eos>"]
        formula_split = [self.de_word2idx[q] if q in self.de_word2idx else self.de_word2idx["<unk>"] for q in formula_split]
        aa = self.data[index]["answer"]

        answer = {"Problem" : problem , "linear_formula" : formula_split, "answer" : aa }
            
        return answer

#batch size=32
def collate_bert(batch):
    
    max_len_problem = max([len(sample['Problem']) for sample in batch])
    max_len_formula = max([len(sample['linear_formula']) for sample in batch])
    
    problem_lens = torch.zeros(len(batch), dtype=torch.long) #
    padded_problem = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    problem_attn_mask = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    
    formula_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_formula = torch.zeros((len(batch), max_len_formula), dtype=torch.long)

    answers = torch.zeros(len(batch), dtype=torch.long)
    for idx in range(len(batch)):
        
        problem = batch[idx]['Problem']
        linear_formula = batch[idx]['linear_formula']
        
        prob_len = len(problem)
        lf_len = len(linear_formula)
        problem_lens[idx] = prob_len
        formula_lens[idx] = lf_len

        
        padded_problem[idx, :prob_len] = torch.LongTensor(problem)
        problem_attn_mask[idx, :prob_len] = torch.ones((1, prob_len), dtype=torch.long) #

        padded_formula[idx, :lf_len] = torch.LongTensor(linear_formula)

        answers[idx] = batch[idx]['answer']
        
    ret = {'problem': padded_problem, 'problem_lens': problem_lens, 'problem_attn_mask': problem_attn_mask, 'linear_formula': padded_formula, 'formula_lens': formula_lens, 'answer' : answers}

    return ret
