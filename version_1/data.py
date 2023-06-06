from torch.utils.data import Dataset
import torch
# from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np

from transformers import BertTokenizer


class MyDataset(Dataset):

    def __init__(self, datas, label_intent, label_slot, word_2_id, label_intent_2_id, label_slot_2_id, max_size,
                 bert=False):
        self.datas = datas  # 数据集
        self.label_intent = label_intent  # label_intent label集
        self.label_slot = label_slot  # label_slot label集

        self.word_2_id = word_2_id  # 词表
        self.label_intent_2_id = label_intent_2_id  # intent label表
        self.label_slot_2_id = label_slot_2_id  # slot label表
        self.max_size = max_size  # 单句最大长度

        self.bert = bert  # 是否使用pre-model
        self.tokenizer = BertTokenizer.from_pretrained("../bert/bert-base-uncased/bert-base-uncased-vocab.txt")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # bert 特殊字符
        self.PAD, self.CLS, self.SEP = '[PAD]', '[CLS]', '[SEP]'

    def __getitem__(self, index):
        sentence = self.datas[index]
        label_intent = self.label_intent[index]
        label_slot = self.label_slot[index]

        # sentence_len = len([i.strip() for i in sentence.split(" ") if i.strip() != ''])

        # y label
        y_intent_index = self.label_intent_2_id[label_intent]

        y_slot = label_slot.split(" ")[:self.max_size]
        y_slot_index = [self.label_slot_2_id.get(word, self.label_slot_2_id.get('<UNK>')) for word in
                        y_slot]

        cur_len = len(y_slot_index)

        # 获取x、y
        if self.bert:
            sentence_cut = [i.strip() for i in sentence.split(" ") if i.strip() != '']  # 严格按空格切分
            sentences = []

            subwords = list(map(self.tokenizer.tokenize, sentence_cut[:self.max_size]))  # 按词进行wordpiece，最大长度50
            subword_lengths = list(map(len, subwords))  # 计算每个词wordpiece的长度
            # subword_lengths = subword_lengths + (self.max_size - cur_len) * [1]

            subwords = ['[CLS]'] + self.tokenizer.tokenize(sentence)  # 原句wordpiece ➕ CLS
            token_start_idxs = 1 + np.cumsum(
                [0] + subword_lengths[:-1])  # 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响
            xs_index = self.tokenizer.convert_tokens_to_ids(subwords)

            # for i in token_start_idxs.tolist():
            #     sentences.append(xs_index_bert[i])

            # 将标签补充至wordpiece一致
            y_slot_fill = []
            for one_slot, one_len in zip(y_slot, subword_lengths):
                if one_len == 1:
                    y_slot_fill.append(one_slot)
                    continue
                if one_slot == 'O':
                    for i in range(one_len):
                        y_slot_fill.append(one_slot)
                else:
                    y_slot_fill.append(one_slot)
                    for i in range(one_len - 1):
                        y_slot_fill.append("I{}".format(one_slot[1:]))

            y_slot_index = [self.label_slot_2_id.get('<PAD>')] + [
                self.label_slot_2_id.get(word, self.label_slot_2_id.get('<UNK>')) for word in
                y_slot_fill]

        else:
            xs_index = [self.word_2_id.get(word, self.word_2_id.get('<UNK>')) for word in sentence.split(" ")]

        return xs_index, y_intent_index, y_slot_index, cur_len, token_start_idxs.tolist()

    def __len__(self):
        return len(self.datas)

    # batch 数据补齐
    def batch_data_process(self, batch_datas):
        xs = []
        ys_intent = []
        ys_slot = []
        xs_len = []  # 原句长度
        xs_wordpiece_len = []  # 分词后长度 - 原句长度
        masks = []  # attention mask
        token_start_idxs = []  # wordpiece 首词下标

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.bert:
            for x, y_intent, y_slot, cur_len, token_start_idx in batch_datas:
                xs.append(x)
                ys_slot.append(y_slot)
                ys_intent.append(y_intent)
                xs_len.append(cur_len)  # 原句长度
                xs_wordpiece_len.append(len(x) - cur_len)  # 句子的分词后长度- 原句长度
                masks.append([1] * len(x))
                token_start_idxs.append([0] + token_start_idx)  # 加上CLS的下标

            cur_max_len = max(xs_wordpiece_len)

            # 将数据补齐
            xs = [i + self.tokenizer.convert_tokens_to_ids([self.PAD]) * (self.max_size + 1 - len(i)) for
                  i in xs]
            ys_slot = [i + [self.label_slot_2_id["<PAD>"]] * (self.max_size + 1 - len(i)) for i in
                       ys_slot]
            masks = [i + [0] * (self.max_size + 1 - len(i)) for i in masks]

            masks = torch.tensor(masks, dtype=torch.long, device=device)

        else:
            for x, y_intent, y_slot, cur_len in batch_datas:
                xs.append(x)
                ys_slot.append(y_slot)
                ys_intent.append(y_intent)
                xs_len.append(cur_len)

            current_max_size = max(xs_len)  # 当前batch的最长size
            # 将数据补齐
            xs = [i + [self.word_2_id["<PAD>"]] * (current_max_size - len(i)) for i in xs]
            ys_slot = [i + [self.label_slot_2_id["<PAD>"]] * (current_max_size - len(i)) for i in ys_slot]

        xs = torch.tensor(xs, dtype=torch.long, device=device)
        ys_intent = torch.tensor(ys_intent, dtype=torch.long, device=device)
        ys_slot = torch.tensor(ys_slot, dtype=torch.long, device=device)
        xs_len = torch.tensor(xs_len, dtype=torch.long, device=device)
        # token_start_idxs = torch.tensor(token_start_idxs, dtype=torch.long, device=device)

        return xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs


class MyDatasetWordPiece(Dataset):
    def __init__(self, datas, label_intent, label_slot, word_2_id, label_intent_2_id, label_slot_2_id, max_size,
                 bert=False):
        self.datas = datas  # 数据集
        self.label_intent = label_intent  # label_intent label集
        self.label_slot = label_slot  # label_slot label集

        self.word_2_id = word_2_id  # 词表
        self.label_intent_2_id = label_intent_2_id  # intent label表
        self.label_slot_2_id = label_slot_2_id  # slot label表
        self.max_size = max_size  # 单句最大长度

        self.tokenizer = BertTokenizer.from_pretrained("../bert/bert-base-uncased/bert-base-uncased-vocab.txt")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # bert 特殊字符
        self.PAD, self.CLS, self.SEP = '[PAD]', '[CLS]', '[SEP]'

    def __getitem__(self, index):
        sentence = self.datas[index]
        label_intent = self.label_intent[index]
        label_slot = self.label_slot[index]


        # y label
        y_intent_index = self.label_intent_2_id.get(label_intent, self.label_slot_2_id.get('<UNK>'))

        y_slot = label_slot.split(" ")[:self.max_size]
        y_slot_index = [self.label_slot_2_id.get(word, self.label_slot_2_id.get('<UNK>')) for word in
                        y_slot]

        cur_len = len(y_slot_index)

        # 获取x、y
        sentence_cut = [i.strip() for i in sentence.split(" ") if i.strip() != '']  # 严格按空格切分
        sentences = []

        subwords = list(map(self.tokenizer.tokenize, sentence_cut[:self.max_size]))  # 按词进行wordpiece，最大长度50
        subword_lengths = list(map(len, subwords))  # 计算每个词wordpiece的长度
        subword_lengths = subword_lengths + (self.max_size - cur_len) * [1]

        subwords = ['[CLS]'] + self.tokenizer.tokenize(sentence)  # 原句wordpiece ➕ CLS
        token_start_idxs = 1 + np.cumsum(
            [0] + subword_lengths[:-1])  # 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响
        xs_index = self.tokenizer.convert_tokens_to_ids(subwords)

        y_slot_index = [self.label_slot_2_id.get('<PAD>')] + y_slot_index

        return xs_index, y_intent_index, y_slot_index, cur_len, token_start_idxs.tolist(), subword_lengths

    def __len__(self):
        return len(self.datas)

    # batch 数据补齐
    def batch_data_process(self, batch_datas):
        xs = []
        ys_intent = []
        ys_slot = []
        xs_len = []  # 原句长度
        xs_wordpiece_len = []  # 分词后长度 - 原句长度
        masks = []  # attention mask
        masks_crf = []  # crf mask 数据为true，PAD为False
        token_start_idxs = []  # wordpiece 首词下标
        subword_lengths = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for x, y_intent, y_slot, cur_len, token_start_idx, subword_length in batch_datas:
            xs.append(x)
            ys_slot.append(y_slot)
            ys_intent.append(y_intent)
            xs_len.append(cur_len)  # 原句长度
            xs_wordpiece_len.append(len(x) - cur_len)  # 句子的分词后长度- 原句长度
            masks.append([1] * len(x))
            masks_crf.append([1] * cur_len)

            token_start_idxs.append([0] + token_start_idx)  # 加上CLS的下标
            subword_lengths.append([1] + subword_length)

        cur_max_len = max(xs_wordpiece_len)

        # 将数据补齐
        xs = [i + self.tokenizer.convert_tokens_to_ids([self.PAD]) * (self.max_size + cur_max_len - len(i)) for
              i in xs]
        ys_slot = [i + [self.label_slot_2_id["<PAD>"]] * (self.max_size + 1 - len(i)) for i in
                   ys_slot]
        masks = [i + [0] * (self.max_size + cur_max_len - len(i)) for i in masks]
        masks_crf = [i + [0] * (self.max_size - len(i)) for i in masks_crf]

        masks = torch.tensor(masks, dtype=torch.long, device=device)
        masks_crf = torch.tensor(masks_crf, dtype=torch.bool, device=device)
        xs = torch.tensor(xs, dtype=torch.long, device=device)
        ys_intent = torch.tensor(ys_intent, dtype=torch.long, device=device)
        ys_slot = torch.tensor(ys_slot, dtype=torch.long, device=device)
        xs_len = torch.tensor(xs_len, dtype=torch.long, device=device)
        token_start_idxs = torch.tensor(token_start_idxs, dtype=torch.long, device=device)

        return xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf
