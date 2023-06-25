import os

# data
data_pkl_file_path = os.path.join("..", "data", "data.pkl")
data_vocab_dic_pkl_file_path = os.path.join("..", "data", "vocab.pkl")
UNK = "<UNK>"
PAD = "<PAD>"

# train
batch_size = 16
lr = 0.00001
epoch = 10
dataset_type_id = 0  # 0 atis; 1 snips
is_CRF = False
is_for_slot = False
model_name = ['bert-base','bert-large  ']

# model para
hidden_num = 128
word_embedding = 64
max_size = 50

bert_hidden_state_size = 768

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
