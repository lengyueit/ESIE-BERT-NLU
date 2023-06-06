batch_size = 30
lr = 0.00005
epoch = 10

# model para
hidden_num = 128
word_embedding = 64
max_size = 50

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
