import json
from collections import Counter, defaultdict
import pickle
import pandas as pd
import subprocess
import argparse
import os
import re
import json
import pickle
import shutil
from time import time
import numpy as np
from collections import defaultdict
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe
import torch.optim as optim
import torch.nn as nn

from text2Math_S2S import *
from text2Math_S2SAtten import *
from text2Math_S2SBert import *
from text2Math_S2SBert import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Text2Math Inference")

    # path to data files.
    parser.add_argument("--test_data_file", type=str, help="Test data file path.")

    parser.add_argument("--model_file", type=str, help="Model File Path.")

    # model type
    parser.add_argument("--model_type", type=str, default="lstm_lstm", help="Select the model you want to run from [lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned].")

    parser.add_argument("--beam_size", type=int, default=10, help="Beam size to be used during beam search decoding.")

    return parser


def load_test_checkpoint(model, model_file):
    """
    Load the model from the model file
    """

    # Load the model
    model_path = os.path.relpath(model_file)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    print("Model loaded from the file: ", model_file)



if __name__ == "__main__":
    parser = get_parser()
    given_args = parser.parse_args()


    '''
    Example: 

    python3 infer.py -model_file models/lstm_lstm_attn.pth --model_type lstm_lstm_attn --test_data_file math_questions_test.json


    '''
    model_type = given_args.model_type
    model_path = given_args.model_file
    test_file  = given_args.test_data_file
    print("Model Type: ", model_type)
    print("Model Path: ", model_path)
    print("Test File: ", test_file)

    # Load the model
    if(model_type == "lstm_lstm"):
        run_test_S2S(model_path, test_file )
    elif(model_type == "lstm_lstm_attn"):
        run_test_S2SAtten(model_path, test_file)
    elif(model_type == "bert_lstm_attn_frozen"):
        run_test_BertFrozen(given_args)
    elif(model_type == "bert_lstm_attn_tuned"):
        run_test_BertTuned(given_args)
    else:
        print("Invalid model type. Please select from [lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned].")
        exit(0)