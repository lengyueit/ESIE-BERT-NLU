import os

# data
data_pkl_file_path = os.path.join("..", "data", "data.pkl")
data_vocab_dic_pkl_file_path = os.path.join("..", "data", "vocab.pkl")
# data multi-lingual
data_pkl_file_path_multi = os.path.join("..", "data", "data.pkl")
data_vocab_dic_pkl_file_path_multi = os.path.join("..", "data", "vocab.pkl")
UNK = "<UNK>"
PAD = "<PAD>"

# train
batch_size = 2
lr = 0.0001
epoch = 50
is_CRF = False
is_for_slot = False
optim_list = ['sgd', 'adam', 'adamW']
optim = "adam"
beta = 0.1

dataset_list = ['atis', 'snips', 'multi-en', 'multi-ph']
dataset = 'atis'
model_name_list = ['bert-base', 'bert-large', 'bert-multilingual', 'gpt1-base', 'gpt2-base', 'gpt2-mid']
model_name = 'gpt2-base'

# model para
max_size = 50

bert_hidden_state_size = 768 if 'base' or 'multilingual' in model_name else 1024

import torch

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device =  "cpu"
