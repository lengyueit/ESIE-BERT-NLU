import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, logging
from torchcrf import CRF

logging.set_verbosity_error()


# Bi LSTM
class MyLstm(nn.Module):
    def __init__(self, word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, pad_index):
        super(MyLstm, self).__init__()

        self.word_size = word_size
        self.word_embedding = word_embedding
        self.hidden_num = hidden_num
        self.intent_label_size = intent_label_size
        self.slot_label_size = slot_label_size

        # layer
        self.embedding = nn.Embedding(word_size, word_embedding, padding_idx=pad_index)
        self.bi_lstm = nn.LSTM(word_embedding, hidden_num, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_id = nn.Linear(hidden_num * 2, intent_label_size)
        self.linear_slot = nn.Linear(hidden_num * 2, slot_label_size)

        # loss fun
        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, data_len, _):
        embedding = self.embedding(xs)
        pack = nn.utils.rnn.pack_padded_sequence(embedding, data_len, batch_first=True, enforce_sorted=False)
        res, (h_t, _) = self.bi_lstm(pack)
        res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)

        h_t = h_t.transpose(0, 1)
        h_t = h_t.reshape(h_t.shape[0], -1)

        # intent d
        res_id = self.linear_id(h_t)

        # slot f
        res_sf = self.linear_slot(res)

        return res_id, res_sf


# Bert joint 对wordpiece不做处理，补长label
class MyBert(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBert, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        # self.bert = BertModel.from_pretrained("../bert-base-uncased/")
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, xs_len, masks, token_start_idxs):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # # slot f, wordpiece 只取首词
        # bert_res_first_wordpiece = []
        # for one_batch_res_all, one_idxs in zip(res_all, token_start_idxs):
        #     bert_res_first_wordpiece.append(torch.index_select(one_batch_res_all, dim=0, index=one_idxs))
        #
        # res_sf = self.linear_slot(torch.tensor([item.cpu().detach().numpy() for item in bert_res_first_wordpiece], device=self.device))
        res_sf = self.linear_slot(res_all)

        return res_id, res_sf


# Bert joint 取wordpiece的第一个，label不变
class MyBertFirstWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertFirstWordPiece, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, masks, token_start_idxs, _):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # # slot f, wordpiece 只取首词
        bert_res_first_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)
        for i, one_batch in enumerate(res_all):
            cur_index_tensor = torch.index_select(one_batch, dim=0, index=token_start_idxs[i])
            bert_res_first_wordpiece[i] = cur_index_tensor

        res_sf = self.linear_slot(bert_res_first_wordpiece)

        return res_id, res_sf


# Bert joint MeanWordPiece  对wordpiece取均值
class MyBertMainWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertMainWordPiece, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, masks, token_start_idxs, subword_lengths):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # slot f, wordpiece 取均值
        bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

        for i, one_batch in enumerate(res_all):

            for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                if lens == 1:
                    mean_word = one_batch[start, :]
                else:
                    target_index = torch.tensor([start, start + lens], device=self.device)
                    cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                    mean_word = torch.mean(cur_index_tensor, dim=0)
                # mean_word = one_batch[start, :]

                bert_res_main_wordpiece[i][index] = mean_word

        res_sf = self.linear_slot(bert_res_main_wordpiece)
        return res_id, res_sf


# 注意力机制根据 hidden来计算 后续embedding各得多少分 点乘，不需要反向传播
class AttnContext(nn.Module):
    def __init__(self):
        super(AttnContext, self).__init__()

    def forward(self, first_embedding, all_embedding):
        # all_embedding:[wordPiece_len, hidden_size]
        # hidden:[1, hidden_size]

        # 点乘操作
        attn_weight = torch.sum(first_embedding * all_embedding, dim=1)  # [wordPiece_len]

        attn_weight = torch.functional.F.softmax(attn_weight).unsqueeze(0)  # [1, wordPiece_len]

        # 类似于注意力向量
        attn_vector = attn_weight.mm(all_embedding)  # [1, hidden_size]

        return attn_vector


# 注意力机制 BP反向传播 将first_embedding 一起linear
class AttentionV1(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionV1, self).__init__()
        self.attn = nn.Linear((hidden_dim * 2), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def concat_score(self, first_embedding, all_embedding):
        # all_embedding:[wordPiece_len, hidden_size]
        # first_embedding:[1, hidden_size]

        word_piece_len = all_embedding.shape[0]
        first_embedding = first_embedding.repeat(word_piece_len, 1)  # [wordPiece_len, hidden_size]
        energy = torch.tanh(
            self.attn(torch.cat((first_embedding, all_embedding), dim=1)))  # [wordPiece_len, hidden_dim]
        attention = self.v(energy).squeeze(1)  # 前向转播为一个值，相比直接将789进行相加 [wordPiece_len]
        return attention  # [wordPiece_len]

    def forward(self, first_embedding, all_embedding):
        # all_embedding:[wordPiece_len, hidden_size]
        # first_embedding:[1, hidden_size]

        attn_energies = self.concat_score(first_embedding, all_embedding)

        attn_weight = torch.functional.F.softmax(attn_energies).unsqueeze(0)  # # softmax归一化，[1, wordPiece_len]

        # 注意力向量
        attn_vector = attn_weight.mm(all_embedding)  # [1, hidden_size]

        return attn_vector + first_embedding


# 注意力机制 BP反向传播 只用wordPiece 做linear
class AttentionV2(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionV2, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def concat_score(self, all_embedding):
        # all_embedding:[wordPiece_len, hidden_size]
        # first_embedding:[1, hidden_size]

        energy = torch.tanh(self.attn(all_embedding))  # [wordPiece_len, hidden_dim]
        attention = self.v(energy).squeeze(1)  # 前向转播为一个值 而不是直接将789进行简单相加 [wordPiece_len]
        return attention  # [wordPiece_len]

    def forward(self, first_embedding, all_embedding):
        # all_embedding:[wordPiece_len, hidden_size]
        # first_embedding:[1, hidden_size]

        attn_energies = self.concat_score(all_embedding)

        attn_weight = torch.functional.F.softmax(attn_energies).unsqueeze(0)  # # softmax归一化，[1, wordPiece_len]

        # 注意力向量
        attn_vector = attn_weight.mm(all_embedding)  # [1, hidden_size]

        return attn_vector + first_embedding


# Bert joint MyBertAttnWordPiece  对wordpiece做attention操作（点乘）
class MyBertAttnWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertAttnWordPiece, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")

        # attention
        self.wordpiece_attention = AttnContext()

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, masks, token_start_idxs, subword_lengths):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # slot f, wordpiece 做attention操作
        bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

        for i, one_batch in enumerate(res_all):
            for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                if lens == 1:
                    word = one_batch[start, :]
                else:
                    target_index = torch.tensor([start, start + lens], device=self.device)
                    cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                    # 对wordPiece首词做attn
                    attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
                    word = attention_word
                bert_res_main_wordpiece[i][index] = word

        res_sf = self.linear_slot(bert_res_main_wordpiece)
        return res_id, res_sf


# Bert joint MyBertAttnWordPiece  对wordpiece做attention操作 + CRF
class MyBertAttnWordPieceCRF(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertAttnWordPieceCRF, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")

        # attention
        self.wordpiece_attention = AttnContext()

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

        # crf
        self.crf = CRF(slot_label_size, batch_first=True)

    def forward(self, xs, masks, token_start_idxs, subword_lengths):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # slot f, wordpiece 做attention操作
        bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

        for i, one_batch in enumerate(res_all):
            for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                if lens == 1:
                    word = one_batch[start, :]
                else:
                    target_index = torch.tensor([start, start + lens], device=self.device)
                    cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                    # 对wordPiece首词做attn
                    attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
                    word = attention_word
                bert_res_main_wordpiece[i][index] = word

        res_sf = self.linear_slot(bert_res_main_wordpiece)
        return res_id, res_sf

    # crf loss fn
    def loss_fn(self, inputs, target, mask_crf):
        return -self.crf.forward(inputs, target, mask_crf, reduction='mean')


# Bert joint MyBertAttnWordPiece  对wordpiece做attention操作（BP）
class MyBertAttnBPWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertAttnBPWordPiece, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")

        # attention
        self.wordpiece_attention = AttentionV2(768)

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, masks, token_start_idxs, subword_lengths, is_for_slot=False):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        # intent d
        res_id = self.linear_id(res)

        # slot f, wordpiece 做attention操作
        bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

        for i, one_batch in enumerate(res_all):
            for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                if lens == 1:
                    word = one_batch[start, :]
                else:
                    target_index = torch.tensor([start, start + lens], device=self.device)
                    cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                    # 对wordPiece首词做attn
                    attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
                    word = attention_word
                bert_res_main_wordpiece[i][index] = word

        res_sf = self.linear_slot(bert_res_main_wordpiece)
        return res_id, res_sf


# Bert joint MyBertAttnWordPiece  对wordpiece做attention操作（BP） + CRF
class MyBertAttnBPWordPieceCRF(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertAttnBPWordPieceCRF, self).__init__()

        # 定义网络层
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert + 冻结参数
        self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")

        # attention
        self.wordpiece_attention = AttentionV2(768)

        self.linear_id = nn.Linear(768, intent_label_size)
        self.linear_slot = nn.Linear(768, slot_label_size)

        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

        # crf
        self.crf = CRF(slot_label_size, batch_first=True)

    def forward(self, xs, masks, token_start_idxs, subword_lengths, is_for_slot=False):
        bert_res = self.bert(xs, attention_mask=masks)

        res_all = bert_res[0]
        res = bert_res[1]  # [CLS]

        if is_for_slot:
            # intent d
            # res_id = self.linear_id(res)

            # slot f, wordpiece 做attention操作
            bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

            for i, one_batch in enumerate(res_all):
                for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                    if lens == 1:
                        word = one_batch[start, :]
                    else:
                        target_index = torch.tensor([start, start + lens], device=self.device)
                        cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                        # 对wordPiece首词做attn
                        attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
                        word = attention_word
                    # bert_res_main_wordpiece[i][index] = word
                    bert_res_main_wordpiece[i][index] = word

            res_sf = self.linear_slot(bert_res_main_wordpiece)

            return "", res_sf
        else:
            # intent d
            res_id = self.linear_id(res)

            # slot f, wordpiece 做attention操作
            bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)

            for i, one_batch in enumerate(res_all):
                for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                    if lens == 1:
                        word = one_batch[start, :]
                    else:
                        target_index = torch.tensor([start, start + lens], device=self.device)
                        cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                        # 对wordPiece首词做attn
                        attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
                        word = attention_word
                    # bert_res_main_wordpiece[i][index] = word
                    bert_res_main_wordpiece[i][index] = word + res[i]  # 将整个CLS的表征+每一个槽位

            res_sf = self.linear_slot(bert_res_main_wordpiece)

            return res_id, res_sf

    # crf loss fn
    def loss_fn(self, inputs, target, mask_crf):
        return -self.crf.forward(inputs, target, mask_crf, reduction='mean')


# bi-RNN
class MyRNN(nn.Module):
    def __init__(self, word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, pad_index):
        super(MyRNN, self).__init__()

        self.word_size = word_size
        self.word_embedding = word_embedding
        self.hidden_num = hidden_num
        self.intent_label_size = intent_label_size
        self.slot_label_size = slot_label_size

        # layer
        self.embedding = nn.Embedding(word_size, word_embedding, padding_idx=pad_index)
        self.rnn = nn.RNN(word_embedding, hidden_num, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_id = nn.Linear(hidden_num * 2, intent_label_size)
        self.linear_slot = nn.Linear(hidden_num * 2, slot_label_size)

        # loss fun
        self.cross_loss_slot = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, data_len, _):
        embedding = self.embedding(xs)
        pack = nn.utils.rnn.pack_padded_sequence(embedding, data_len, batch_first=True, enforce_sorted=False)
        res, h_t = self.rnn(pack)
        res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)

        h_t = h_t.transpose(0, 1)
        h_t = h_t.reshape(h_t.shape[0], -1)

        # intent d
        res_id = self.linear_id(h_t)

        # slot f
        res_sf = self.linear_slot(res)

        return res_id, res_sf


# bi-GRU
class MyGRU(nn.Module):
    def __init__(self, word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, pad_index):
        super(MyGRU, self).__init__()

        self.word_size = word_size
        self.word_embedding = word_embedding
        self.hidden_num = hidden_num
        self.intent_label_size = intent_label_size
        self.slot_label_size = slot_label_size

        # layer
        self.embedding = nn.Embedding(word_size, word_embedding, padding_idx=pad_index)
        self.gru = nn.GRU(word_embedding, hidden_num, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_id = nn.Linear(hidden_num * 2, intent_label_size)
        self.linear_slot = nn.Linear(hidden_num * 2, slot_label_size)

        # loss fun
        self.cross_loss_slot = nn.CrossEntropyLoss()
        self.cross_loss_intent = nn.CrossEntropyLoss()

    def forward(self, xs, data_len, _):
        embedding = self.embedding(xs)
        pack = nn.utils.rnn.pack_padded_sequence(embedding, data_len, batch_first=True, enforce_sorted=False)
        res, h_t = self.gru(pack)
        res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)

        h_t = h_t.transpose(0, 1)
        h_t = h_t.reshape(h_t.shape[0], -1)

        # intent d
        res_id = self.linear_id(h_t)

        # slot f
        res_sf = self.linear_slot(res)

        return res_id, res_sf


if __name__ == '__main__':
    model = MyBertAttnBPWordPiece(30, 30)
    print(model)
