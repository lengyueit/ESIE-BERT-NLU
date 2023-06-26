import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformers import BertModel, AutoModel, logging, GPT2LMHeadModel, OpenAIGPTModel
from torchcrf import CRF

logging.set_verbosity_error()


# Bert joint 取wordpiece的第一个，label不变
class MyBertFirstWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertFirstWordPiece, self).__init__()

        # 定义网络层
        self.device = config.device

        # pretrain_model
        if config.model_name == "bert-base":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")
        elif config.model_name == "bert-large":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/bert/bert-large-uncased/")
        elif config.model_name == "gpt-base":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/gpt/gpt2-base/")
        elif config.model_name == "gpt-large":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/gpt/gpt2-large/")

        self.linear_id = nn.Linear(config.bert_hidden_state_size, intent_label_size)
        self.linear_slot = nn.Linear(config.bert_hidden_state_size, slot_label_size)

        self.cross_loss_fn = nn.CrossEntropyLoss()

    if 'bert' in config.model_name:
        def forward(self, xs, masks, token_start_idxs, _, __):
            batch_size, _ = xs.shape

            pretrain_model_res = self.pretrain_model(xs, attention_mask=masks)

            res_all = pretrain_model_res[0]
            res = pretrain_model_res[1]  # [CLS]

            # intent d
            res_id = self.linear_id(res)

            # slot f, wordpiece 只取首词
            bert_res_first_wordpiece = torch.zeros((batch_size, config.max_size + 1, config.bert_hidden_state_size),
                                                   device=self.device)
            for i, one_batch in enumerate(res_all):
                cur_index_tensor = torch.index_select(one_batch, dim=0, index=token_start_idxs[i])
                bert_res_first_wordpiece[i] = cur_index_tensor

            res_sf = self.linear_slot(bert_res_first_wordpiece)

            return res_id, res_sf
    elif 'gpt' in config.model_name:
        # todo gpt 架构
        def forward(self, xs, masks, token_start_idxs, _, __):
            batch_size, _ = xs.shape

            pretrain_model_res = self.pretrain_model(xs, attention_mask=masks)

            last_hidden_layer_all = pretrain_model_res[0]
            last_hidden_layer_last_word = last_hidden_layer_all[:, -1, :] # [CLS]

            # intent d
            res_id = self.linear_id(last_hidden_layer_last_word)

            # slot f, wordpiece 只取首词
            bert_res_first_wordpiece = torch.zeros((batch_size, config.max_size + 1, config.bert_hidden_state_size),
                                                   device=self.device)
            for i, one_batch in enumerate(last_hidden_layer_all):
                cur_index_tensor = torch.index_select(one_batch, dim=0, index=token_start_idxs[i])
                bert_res_first_wordpiece[i] = cur_index_tensor

            res_sf = self.linear_slot(bert_res_first_wordpiece)

            return res_id, res_sf


class IAA_Attn(nn.Module):
    """
    intention attention
    """

    def __init__(self, hidden_dim):
        super(IAA_Attn, self).__init__()
        self.dh = hidden_dim
        self.Wint = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, CLS_embedding, all_embedding):
        """
        :param all_embedding: [batch, max_len ,hidden_state_size]
        :return: h_intent + CLS_embedding
        """
        all_embedding = all_embedding[:, 1:, :]  # 丢掉CLS
        CLS_embedding = CLS_embedding.unsqueeze(1)

        Wint_hi = self.relu(self.Wint(all_embedding)).transpose(1, 2)
        Wh = CLS_embedding @ Wint_hi

        int_attn_score = F.softmax(Wh / torch.sqrt(torch.tensor(self.dh)), dim=2)  # softmax归一化

        h_intent = int_attn_score.bmm(all_embedding)

        return h_intent + CLS_embedding


class SAA_Attn(nn.Module):
    """
    slot attention
    """

    def __init__(self, hidden_dim):
        super(SAA_Attn, self).__init__()
        self.Wsq = nn.Linear(hidden_dim, hidden_dim)
        self.Wsk = nn.Linear(hidden_dim, hidden_dim)
        self.Wsv = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, first_embedding, all_embedding):
        # first_embedding:[1, hidden_size]
        # all_embedding:[wordPiece_len, hidden_size]

        # self-attn
        Q = self.relu(self.Wsq(all_embedding))
        K = self.relu(self.Wsk(all_embedding))
        V = self.relu(self.Wsv(all_embedding))
        attn_weight = Q @ K.transpose(0, 1)  # [wordPiece_len, wordPiece_len]

        attn_score = F.softmax(attn_weight, dim=1)  # softmax归一化，[wordPiece_len, wordPiece_len]

        # 注意力向量
        attn_vector = attn_score.mm(V)  # [wordPiece_len, hidden_size]

        # 返回首词的SAA feature
        return attn_vector[0] + first_embedding


# Bert joint EISE
class MyBertAttnBPWordPiece(nn.Module):
    def __init__(self, intent_label_size, slot_label_size):
        super(MyBertAttnBPWordPiece, self).__init__()

        # pretrain_model
        if config.model_name == "bert-base":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")
        elif config.model_name == "bert-large":
            self.pretrain_model = AutoModel.from_pretrained("../pretrain-model/bert/bert-large-uncased/")

        # attention
        self.wordpiece_attention = SAA_Attn(config.bert_hidden_state_size)
        self.intent_attention = IAA_Attn(config.bert_hidden_state_size)

        self.linear_id = nn.Linear(config.bert_hidden_state_size, intent_label_size)
        self.linear_slot = nn.Linear(config.bert_hidden_state_size, slot_label_size)

        self.cross_loss_fn = nn.CrossEntropyLoss()

    def forward(self, xs, masks, token_start_idxs, subword_lengths, is_for_slot=None):
        batch_size, _ = xs.shape

        pretrain_model_res = self.pretrain_model(xs, attention_mask=masks)

        res_all = pretrain_model_res[0]
        res = pretrain_model_res[1]  # [CLS]

        # slot SAA, wordpiece 做attention操作
        bert_res_main_wordpiece = torch.zeros((batch_size, config.max_size + 1, config.bert_hidden_state_size),
                                              device=config.device)
        # each batch 串行操作
        for i, one_batch in enumerate(res_all):
            for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):

                if lens == 1:
                    word = one_batch[start, :]
                else:
                    # 选出指定word的 subword 对应的tensor
                    target_index = torch.range(start, start + lens, device=config.device, dtype=torch.int)
                    cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)

                    # 对wordPiece首词做attn
                    attention_word = self.wordpiece_attention(cur_index_tensor[0], cur_index_tensor)
                    word = attention_word
                bert_res_main_wordpiece[i][index] = word

        # intent IAA
        h_intent = self.intent_attention(res, bert_res_main_wordpiece)

        # slot jointly intent feature
        slot_joint_intent_wordpiece = h_intent.repeat(1, bert_res_main_wordpiece.shape[1], 1) + bert_res_main_wordpiece

        # predict id and slot
        res_id = self.linear_id(h_intent.squeeze(1))
        res_sf = self.linear_slot(slot_joint_intent_wordpiece)

        return res_id, res_sf


# Bert joint EISE + CRF
# class MyBertAttnBPWordPieceCRF(nn.Module):
#     def __init__(self, intent_label_size, slot_label_size):
#         super(MyBertAttnBPWordPieceCRF, self).__init__()
#
#         # 定义网络层
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         # bert + 冻结参数
#         self.bert = BertModel.from_pretrained("../pretrain-model/bert/bert-base-uncased/")
#
#         # attention
#         self.wordpiece_attention = AttentionV2(768)
#
#         self.linear_id = nn.Linear(768, intent_label_size)
#         self.linear_slot = nn.Linear(768, slot_label_size)
#
#         self.cross_loss_slot = nn.CrossEntropyLoss()
#         self.cross_loss_intent = nn.CrossEntropyLoss()
#
#         # crf
#         self.crf = CRF(slot_label_size, batch_first=True)
#
#     def forward(self, xs, masks, token_start_idxs, subword_lengths, is_for_slot=False):
#         bert_res = self.bert(xs, attention_mask=masks)
#
#         res_all = bert_res[0]
#         res = bert_res[1]  # [CLS]
#
#         if is_for_slot:
#             # intent d
#             # res_id = self.linear_id(res)
#
#             # slot f, wordpiece 做attention操作
#             bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)
#
#             for i, one_batch in enumerate(res_all):
#                 for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):
#
#                     if lens == 1:
#                         word = one_batch[start, :]
#                     else:
#                         target_index = torch.tensor([start, start + lens], device=self.device)
#                         cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)
#
#                         # 对wordPiece首词做attn
#                         attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
#                         word = attention_word
#                     # bert_res_main_wordpiece[i][index] = word
#                     bert_res_main_wordpiece[i][index] = word
#
#             res_sf = self.linear_slot(bert_res_main_wordpiece)
#
#             return "", res_sf
#         else:
#             # intent d
#             res_id = self.linear_id(res)
#
#             # slot f, wordpiece 做attention操作
#             bert_res_main_wordpiece = torch.zeros((xs.shape[0], 51, 768), device=self.device)
#
#             for i, one_batch in enumerate(res_all):
#                 for index, (start, lens) in enumerate(zip(token_start_idxs[i], subword_lengths[i])):
#
#                     if lens == 1:
#                         word = one_batch[start, :]
#                     else:
#                         target_index = torch.tensor([start, start + lens], device=self.device)
#                         cur_index_tensor = torch.index_select(one_batch, dim=0, index=target_index)
#
#                         # 对wordPiece首词做attn
#                         attention_word = self.wordpiece_attention(one_batch[start, :], cur_index_tensor)
#                         word = attention_word
#                     # bert_res_main_wordpiece[i][index] = word
#                     bert_res_main_wordpiece[i][index] = word + res[i]  # 将整个CLS的表征+每一个槽位
#
#             res_sf = self.linear_slot(bert_res_main_wordpiece)
#
#             return res_id, res_sf
#
#     # crf loss fn
#     def loss_fn(self, inputs, target, mask_crf):
#         return -self.crf.forward(inputs, target, mask_crf, reduction='mean')


if __name__ == '__main__':
    model = MyBertAttnBPWordPiece(30, 30)
    print(model)
