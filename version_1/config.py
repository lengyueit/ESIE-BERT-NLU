import os

# data
data_pkl_file_path = os.path.join("..", "data", "data.pkl")
data_vocab_dic_pkl_file_path = os.path.join("..", "data", "vocab.pkl")
UNK = "<UNK>"
PAD = "<PAD>"

# train
batch_size = 5
lr = 0.0001
epoch = 30
is_CRF = False
is_for_slot = False
optim = "adam"

dataset_list = ['atis', 'snips']
dataset = 'atis'
model_name_list = ['bert-base', 'bert-large', 'gpt1', 't5-base', 't5-large']
model_name = 'gpt-base'

# model para
hidden_num = 128
word_embedding = 64
max_size = 50

bert_hidden_state_size = 768 if 'base' in model_name else 1024

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
