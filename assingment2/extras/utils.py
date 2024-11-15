import torch
import torch.nn as nn
import os
import numpy as np
from nltk.tokenize import word_tokenize
import json
import re
import pickle




def tokenize_problem(problem):
    return word_tokenize(problem)

def tokenize_formula(formula):
    all_formulas = formula.split("|")
    formula_split = []
    for s in all_formulas:
        s_brk = re.match(r"([a-zA-Z]+)\(([^,]+),([^)]+)\)", s)
        if s_brk:
            formula_split.extend([s_brk.group(1),"(", s_brk.group(2),",", s_brk.group(3), ")","|"])
    return formula_split


def build_vocab(pth):
    # make file encoder.vocab dn decoder.vocab 
    # data is the directory which contain train.json
    
    encoder_vocab = set()
    decoder_vocab = set()
    encoder_word2idx = {}
    encoder_idx2word = {}
    decoder_word2idx = {}
    decoder_idx2word = {}
    pth2 = os.path.join(pth, "train.json")
    data = json.load(open(pth2))
    for d in data:
        problem = d["Problem"]
        linear_formula = d["linear_formula"]
        problem = tokenize_problem(problem)
        linear_formula = tokenize_formula(linear_formula)
        for p in problem:
            encoder_vocab.add(p)
        for l in linear_formula:
            decoder_vocab.add(l)

    pth2 = os.path.join(pth, "dev.json")
    data = json.load(open(pth2))
    for d in data:
        problem = d["Problem"]
        linear_formula = d["linear_formula"]
        problem = tokenize_problem(problem)
        linear_formula = tokenize_formula(linear_formula)
        for p in problem:
            encoder_vocab.add(p)
        for l in linear_formula:
            decoder_vocab.add(l)

    pth2 = os.path.join(pth, "test.json")
    data = json.load(open(pth2))
    for d in data:
        problem = d["Problem"]
        linear_formula = d["linear_formula"]
        problem = tokenize_problem(problem)
        linear_formula = tokenize_formula(linear_formula)
        for p in problem:
            encoder_vocab.add(p)
        for l in linear_formula:
            decoder_vocab.add(l)

    #add special tokens
    encoder_vocab.add("<sos>")
    encoder_vocab.add("<eos>")
    encoder_vocab.add("<unk>")
    encoder_vocab.add("<pad>")

    decoder_vocab.add("<sos>")
    decoder_vocab.add("<eos>")
    decoder_vocab.add("<unk>")
    decoder_vocab.add("<pad>")
    encoder_vocab = list(encoder_vocab)
    decoder_vocab = list(decoder_vocab)
    encoder_vocab.sort()
    decoder_vocab.sort()

    for i, word in enumerate(encoder_vocab):
        encoder_word2idx[word] = i
        encoder_idx2word[i] = word
    for i, word in enumerate(decoder_vocab):
        decoder_word2idx[word] = i
        decoder_idx2word[i] = word

    

    with open(os.path.join(pth, "encoder.vocab"), "x") as file:
        for word in encoder_vocab:
            file.write(word + "\n")
    with open(os.path.join(pth, "decoder.vocab"), "x") as file:
        for word in decoder_vocab:
            file.write(word + "\n")

    # make pickle file for word2idx and idx2word
    with open(os.path.join(pth, "encoder_word2idx.pickle"), "xb") as file:
        pickle.dump(encoder_word2idx, file)
    with open(os.path.join(pth, "encoder_idx2word.pickle"), "xb") as file:
        pickle.dump(encoder_idx2word, file)
    with open(os.path.join(pth, "decoder_word2idx.pickle"), "xb") as file:
        pickle.dump(decoder_word2idx, file)
    with open(os.path.join(pth, "decoder_idx2word.pickle"), "xb") as file:
        pickle.dump(decoder_idx2word, file)

    print("Encoder Vocab Size = {}, Decoder Vocab Size = {}".format(len(encoder_vocab), len(decoder_vocab)) )

def load_checkpoint(args, chkpt = "best"):

    if chkpt == "best":
        model_name = os.path.join(args.checkpoint_dir, "best_loss_checkpoint_{}.pth".format(args.model_type))
        status_file = os.path.join(args.checkpoint_dir, "best_loss_chkpt_status_{}.json".format(args.model_type))
    else:
        model_name = os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type))
        status_file = os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type))

    assert os.path.isfile(model_name), f"Model path/name invalid: {model_name}"
    
    net = torch.load(model_name)
    with open(status_file, "r") as file:
        model_dict = json.load(file)
    print(f"\n|--------- Model Load Success. Trained Epoch: {str(model_dict['epoch'])}")

    return net


