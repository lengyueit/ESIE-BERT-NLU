import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from model import *
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics import classification_report as seq_classification_report, accuracy_score as seq_accuracy_score, \
    f1_score as seq_f1_score, precision_score as seq_precision_score, recall_score as seq_recall_score
import pickle
from tqdm import tqdm
import warnings
import config
from transformers import AutoTokenizer
import numpy as np

warnings.filterwarnings("ignore")


class MyDatasetWordPiece(Dataset):
    def __init__(self, datas, label_intent, label_slot, word_2_id, label_intent_2_id, label_slot_2_id, max_size):
        self.datas = datas  # 数据集
        self.label_intent = label_intent  # label_intent label集
        self.label_slot = label_slot  # label_slot label集

        self.word_2_id = word_2_id  # 词表 discard，利用BERT的词表即可
        self.label_intent_2_id = label_intent_2_id  # intent label表
        self.label_slot_2_id = label_slot_2_id  # slot label表
        self.max_size = max_size  # 单句最大长度

        if "bert" in config.model_name and "multilingual" not in config.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("..", "pretrain-model", "bert", "bert-base-uncased"))
        elif "multilingual" in config.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("..", "pretrain-model", "bert", "bert-base-multilingual-uncased"))
        elif "gpt1" in config.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("..", "pretrain-model", "gpt", "gpt1"))
        elif "gpt2" in config.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("..", "pretrain-model", "gpt", "gpt2-base"))

        # bert 特殊字符
        self.PAD, self.CLS, self.SEP = '[PAD]', '[CLS]', '[SEP]'

    def __getitem__(self, index):
        sentence = self.datas[index]
        label_intent = self.label_intent[index]
        label_slot = self.label_slot[index]

        # y label
        y_intent_index = self.label_intent_2_id.get(label_intent, '<UNK>')

        y_slot = label_slot.split(" ")[:self.max_size]
        y_slot_index = [self.label_slot_2_id.get(word, '<UNK>') for word in
                        y_slot]

        cur_len = len(y_slot_index)

        # 获取x、y
        sentence_cut = [i.strip() for i in sentence.split(" ") if i.strip() != '']  # 严格按空格切分

        subwords = list(map(self.tokenizer.tokenize, sentence_cut[:self.max_size]))  # 按词进行wordpiece，最大长度50
        subword_lengths = list(map(len, subwords))  # 计算每个词wordpiece的长度
        subword_lengths = subword_lengths + (self.max_size - cur_len) * [1]

        subwords = ['[CLS]'] + self.tokenizer.tokenize(sentence)  # CLS + 原句wordpiece
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
        masks = []  # attention pad mask
        masks_crf = []  # crf mask; 数据为true，PAD为False
        token_start_idxs = []  # wordpiece 首词下标
        subword_lengths = []  # 记录 sub-words wordpiece 长度

        device = config.device

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

        cur_max_len = max(xs_wordpiece_len)  # 当前batch中 subwords 最长

        # 将数据补齐
        xs = [i + self.tokenizer.convert_tokens_to_ids([self.PAD]) * (self.max_size + cur_max_len - len(i)) for
              i in xs]
        masks = [i + [0] * (self.max_size + cur_max_len - len(i)) for i in masks]
        masks_crf = [i + [0] * (self.max_size - len(i)) for i in masks_crf]

        ys_slot = [i + [self.label_slot_2_id["<PAD>"]] * (self.max_size + 1 - len(i)) for i in
                   ys_slot]

        masks = torch.tensor(masks, dtype=torch.long, device=device)
        masks_crf = torch.tensor(masks_crf, dtype=torch.bool, device=device)
        xs = torch.tensor(xs, dtype=torch.long, device=device)
        ys_intent = torch.tensor(ys_intent, dtype=torch.long, device=device)
        ys_slot = torch.tensor(ys_slot, dtype=torch.long, device=device)
        xs_len = torch.tensor(xs_len, dtype=torch.long, device=device)
        token_start_idxs = torch.tensor(token_start_idxs, dtype=torch.long, device=device)

        return xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf


def train(model, train_dataloader, valid_dataloader, test_dataloader, device, batch_size, num_epoch, lr, optim='adam',
          is_CRF=False, is_for_slot=False):
    print('training on:', device)
    model.to(device)

    # 优化器选择
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)

    # 用来保存每个epoch的Loss和acc以便最后画图
    train_losses = []
    train_id_acc_list = []
    train_slot_f1_list = []
    eval_id_acc_list = []
    eval_slot_f1_list = []
    best_acc = 0.0

    # 训练
    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        model.train()

        for batch in tqdm(train_dataloader, desc='训练'):
            xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf = batch

            if len(masks) > 0:
                masks = masks.to(device)
                # ys_slot = ys_slot[:, 1:]

            res_id, res_sf = model(xs, masks, token_start_idxs, subword_lengths, is_for_slot)

            optimizer.zero_grad()

            # 计算 id loss ，只Train slot 任务时无需id loss
            if is_for_slot == False:
                loss_id = model.cross_loss_fn(res_id, ys_intent)  # [batch_size, sentence_len] <-> [batch_size]

            # 丢掉 [CLS]
            res_sf = res_sf[:, 1:, ]
            ys_slot = ys_slot[:, 1:]

            # 计算 sf loss
            if is_CRF:
                loss_sf = model.loss_fn(res_sf, ys_slot, masks_crf)
            else:
                res_sf_2D = res_sf.reshape(-1, res_sf.shape[-1])
                ys_slot_2D = ys_slot.reshape(-1)
                loss_sf = model.cross_loss_fn(res_sf_2D, ys_slot_2D)

            # 只记录slot loss
            if is_for_slot:
                loss = loss_sf
            else:
                loss = config.beta * loss_id + (1 - config.beta) * loss_sf

            loss.backward()
            optimizer.step()

            # 评估
            train_id_acc = 0
            train_slot_f1 = 0

            all_id_pre = []  # id pre
            all_id_tag = []  # id y
            sentences_acc = []  # overall id acc

            all_slot_pre = []  # slot pre
            all_slot_tag = []  # slot y
            sentences_f1 = []  # overall slot f1

            if is_for_slot == False:  # train jointly
                # intent d
                pre_id_list = torch.argmax(res_id, dim=-1).detach().cpu().numpy().tolist()
                ys_intent_list = ys_intent.cpu().numpy().tolist()

                # 计算 id acc
                all_id_pre += pre_id_list
                all_id_tag += ys_intent_list

                for one_intent, one_pre in zip(ys_intent_list, pre_id_list):
                    sentences_acc.append(one_intent == one_pre)

            # slot f1
            pre_sf_list = torch.argmax(res_sf, dim=-1).detach().cpu().numpy().tolist()
            ys_slot_list = ys_slot.cpu().numpy().tolist()
            xs_len = xs_len.cpu().numpy().tolist()

            # 计算sf f1
            if is_CRF:
                pre_sf_list = model.crf.decode(res_sf, masks_crf)

            # 评估slot f1 不要pad
            for one_pre, one_tag, one_len in zip(pre_sf_list, ys_slot_list, xs_len):
                # if is_CRF is False:
                #     one_pre = one_pre[1:one_len + 1]
                # one_tag = one_tag[1:one_len + 1]

                one_pre = [id_2_label_slot[i] for i in one_pre[:one_len]]
                one_tag = [id_2_label_slot[i] for i in one_tag[:one_len]]

                all_slot_pre.append(one_pre)
                all_slot_tag.append(one_tag)
                # 评估 sentences
                sentences_f1.append(seq_f1_score([one_pre], [one_tag]))
            # break

        # 评估 overall
        if is_for_slot == False:
            # joint
            sentences_overall = 0
            for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1):
                if one_intent and one_slot_f1 == 1.0:
                    sentences_overall += 1

            print("epoch: {}, Loss: {}, slot_f1: {}, id_acc: {}, overall: {}".format(epoch + 1, loss.item(),
                                                                                     seq_f1_score(all_slot_tag,
                                                                                                  all_slot_pre),
                                                                                     accuracy_score(all_id_tag,
                                                                                                    all_id_pre),
                                                                                     sentences_overall / len(
                                                                                         sentences_acc)))
        else:
            # only slot
            print("epoch: {}, Loss: {}, slot_f1: {}".format(epoch + 1, loss.item(),
                                                            seq_f1_score(all_slot_tag, all_slot_pre)))

        # break

        # train_id_acc_list.append(train_id_acc / len(train_dataloader))
        # train_slot_f1_list.append(train_slot_f1 / len(train_dataloader))
        # train_losses.append(loss.item())

        # # 验证步骤 每个epoch完后验证一次
        # model.eval()
        # all_pre = []
        # all_tag = []
        # with torch.no_grad():
        #     for batch in valid_dataloader:
        #         xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf = batch
        #
        #         if len(masks) > 0:
        #             masks = masks.to(device)
        #             # ys_slot = ys_slot[:, 1:]
        #
        #         pre_id, pre_sf = model(xs, masks, token_start_idxs, subword_lengths)
        #
        #         pre_id = torch.argmax(pre_id, dim=-1).detach().cpu().numpy().tolist()
        #         pre_sf_list = torch.argmax(pre_sf, dim=-1).detach().cpu().numpy().tolist()
        #         ys_slot_list = ys_slot.cpu().numpy().tolist()
        #         ys_intent_list = ys_intent.cpu().numpy().tolist()
        #         xs_len = xs_len.cpu().numpy().tolist()
        #
        #         # 评估 id
        #         # print(classification_report(ys_intent_list, pre_id))
        #         # print("整体验证集上的acc intent :{}".format(accuracy_score(ys_intent_list, pre_id)))
        #         # print("整体验证集上的pre_s intent :{}".format(precision_score(ys_intent_list, pre_id, average="macro")))
        #         # print("整体验证集上的recall_score intent :{}".format(recall_score(ys_intent_list, pre_id, average="macro")))
        #         # print("整体验证集上的f1 intent :{}".format(f1_score(ys_intent_list, pre_id, average="macro")))
        #
        #         sentences_acc = []
        #         for one_intent, one_pre in zip(ys_intent_list, pre_id):
        #             sentences_acc.append(one_intent == one_pre)
        #
        #         # 评估 slot
        #         sentences_f1 = []
        #         sentences_f1_sklearn = []
        #         all_pre = []
        #         all_tag = []
        #         all_pre_sklearn = []
        #         all_tag_sklearn = []
        #
        #         if is_CRF:
        #             pre_sf_list = model.crf.decode(pre_sf[:, 1:, ], masks_crf)
        #
        #         for one_pre, one_tag, one_len in zip(pre_sf_list, ys_slot_list, xs_len):
        #             if is_CRF is False:
        #                 one_pre = one_pre[1:one_len + 1]
        #
        #             one_tag = one_tag[1:one_len + 1]
        #
        #             # sklearn
        #             all_pre_sklearn += one_pre
        #             all_tag_sklearn += one_tag
        #
        #             # 评估 sklearn sentences
        #             sentences_f1_sklearn.append(f1_score(one_pre, one_tag, average='micro'))
        #
        #             # seqveal
        #             one_pre = [id_2_label_slot[i] for i in one_pre]
        #             one_tag = [id_2_label_slot[i] for i in one_tag]
        #
        #             all_pre.append(one_pre)
        #             all_tag.append(one_tag)
        #
        #             # 评估 sentences
        #             sentences_f1.append(
        #                 seq_f1_score([one_pre], [one_tag]))
        #
        #     """保存最佳模型"""
        #     # eval_acc += accuracy_score(all_tag, all_pre)
        #     # eval_losses = eval_loss / (len(valid_dataloader))
        #     # eval_acc = eval_acc / (len(valid_dataloader))
        #     # if eval_acc > best_acc:
        #     #     best_acc = eval_acc
        #     # torch.save(net.state_dict(), 'best_acc.pth')
        #     # eval_acces.append(eval_acc)
        #     # print("整体验证集上的Loss: {}".format(eval_losses))
        #     # print("整体验证集上的正确率: {}".format(eval_acc))
        #
        #     acc = seq_accuracy_score(all_tag, all_pre)
        #     f1 = seq_f1_score(all_tag, all_pre, average="micro")
        #     f1_sklearn = f1_score(all_tag_sklearn, all_pre_sklearn, average="micro")
        #     p_score = seq_precision_score(all_tag, all_pre, average="micro")
        #     re_score = seq_recall_score(all_tag, all_pre, average="micro")
        #     # print("整体验证集上的accuracy_score slot:{}".format(acc))
        #     # print("整体验证集上的f1_score slot:{}".format(f1))
        #     # print("整体验证集上的f1_sklearn_score slot:{}".format(f1_sklearn))
        #     # print("整体验证集上的precision_score slot:{}".format(p_score))
        #     # print("整体验证集上的recall_score slot:{}".format(re_score))
        #     # print(seq_classification_report(all_tag, all_pre))
        #
        #     # overall
        #     sentences_overall = 0
        #     sentences_overall_90 = 0
        #     for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1):
        #         if one_intent and one_slot_f1 == 1.0:
        #             sentences_overall += 1
        #
        #         if one_intent and one_slot_f1 >= 0.9:
        #             sentences_overall_90 += 1
        #
        #     sentences_overall_sklearn = 0
        #     sentences_overall_sklearn_90 = 0
        #     for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1_sklearn):
        #         if one_intent and one_slot_f1 == 1.0:
        #             sentences_overall_sklearn += 1
        #
        #         if one_intent and one_slot_f1 >= 0.9:
        #             sentences_overall_sklearn_90 += 1
        #
        #     # print("sentences overall :{}".format(sentences_overall / len(sentences_acc)))
        #     # print("sentences sentences_overall_90 :{}".format(sentences_overall_90 / len(sentences_acc)))
        #     # print("sentences overall_sklearn :{}".format(sentences_overall_sklearn / len(sentences_acc)))
        #     # print(
        #     #     "sentences sentences_overall_sklearn_90 :{}".format(sentences_overall_sklearn_90 / len(sentences_acc)))
        #
        #     print("验证集合slot_f1: {},sklearn_slot_f1: {}, id_acc: {}, overall: {}".format(f1, f1_sklearn,
        #                                                                                 accuracy_score(ys_intent_list,
        #                                                                                                pre_id),
        #                                                                                 sentences_overall / len(
        #                                                                                     sentences_acc)))

    # 测试步骤开始
    model.eval()
    eval_loss = 0
    eval_acc = 0
    all_pre = []
    all_tag = []
    with torch.no_grad():
        for batch in test_dataloader:
            xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf = batch

            if len(masks) > 0:
                masks = masks.to(device)
                # ys_slot = ys_slot[:, 1:]

            # forward
            pre_id, pre_sf = model(xs, masks, token_start_idxs, subword_lengths, is_for_slot)

            if is_for_slot == False:
                # train jointly
                pre_id = torch.argmax(pre_id, dim=-1).detach().cpu().numpy().tolist()
                ys_intent_list = ys_intent.cpu().numpy().tolist()

            # 评估 id
            if is_for_slot == False:
                print("---------------------------------ID classification ---------------------------------")
                print("整体验证集上的acc intent :{}".format(accuracy_score(ys_intent_list, pre_id)))
                print("整体验证集上的precision_score intent :{}".format(
                    precision_score(ys_intent_list, pre_id, average="macro")))
                print("整体验证集上的recall_score intent :{}".format(
                    recall_score(ys_intent_list, pre_id, average="macro")))
                print("整体验证集上的f1 intent :{}".format(f1_score(ys_intent_list, pre_id, average="macro")))

                print(classification_report(ys_intent_list, pre_id))
                print("---------------------------------ID classification ---------------------------------")

                sentences_acc = []
                for one_intent, one_pre in zip(ys_intent_list, pre_id):
                    sentences_acc.append(one_intent == one_pre)

            # 丢掉 [CLS]
            pre_sf = pre_sf[:, 1:, ]
            ys_slot = ys_slot[:, 1:]

            # slot f1
            pre_sf_list = torch.argmax(pre_sf, dim=-1).detach().cpu().numpy().tolist()
            ys_slot_list = ys_slot.cpu().numpy().tolist()
            xs_len = xs_len.cpu().numpy().tolist()

            # 评估 slot
            sentences_f1 = []  # slot f1
            sentences_f1_sklearn = []  # slot f1

            all_pre = []
            all_tag = []
            all_pre_sklearn = []
            all_tag_sklearn = []

            if is_CRF:
                pre_sf_list = model.crf.decode(pre_sf, masks_crf)

            for one_pre, one_tag, one_len in zip(pre_sf_list, ys_slot_list, xs_len):
                # if is_CRF is False:
                #     one_pre = one_pre[1:one_len + 1]

                # one_tag = one_tag[1:one_len + 1]

                # sklearn
                all_pre_sklearn += one_pre[:one_len]
                all_tag_sklearn += one_tag[:one_len]

                # 评估 sklearn sentences
                sentences_f1_sklearn.append(f1_score(one_pre, one_tag, average='micro'))

                # seqveal
                one_pre = [id_2_label_slot[i] for i in one_pre[:one_len]]
                one_tag = [id_2_label_slot[i] for i in one_tag[:one_len]]

                all_pre.append(one_pre)
                all_tag.append(one_tag)

                # 评估 sentences
                sentences_f1.append(
                    seq_f1_score([one_pre], [one_tag]))

        """保存最佳模型"""
        # eval_acc += accuracy_score(all_tag, all_pre)
        # eval_losses = eval_loss / (len(valid_dataloader))
        # eval_acc = eval_acc / (len(valid_dataloader))
        # if eval_acc > best_acc:
        #     best_acc = eval_acc
        # torch.save(net.state_dict(), 'best_acc.pth')
        # eval_acces.append(eval_acc)
        # print("整体验证集上的Loss: {}".format(eval_losses))
        # print("整体验证集上的正确率: {}".format(eval_acc))

        acc = seq_accuracy_score(all_tag, all_pre)
        f1 = seq_f1_score(all_tag, all_pre, average="micro")
        f1_sklearn = f1_score(all_tag_sklearn, all_pre_sklearn, average="micro")
        p_score = seq_precision_score(all_tag, all_pre, average="micro")
        re_score = seq_recall_score(all_tag, all_pre, average="micro")
        print("整体验证集上的accuracy_score slot:{}".format(acc))
        print("整体验证集上的f1_score slot:{}".format(f1))
        print("整体验证集上的f1_sklearn_score slot:{}".format(f1_sklearn))
        print("整体验证集上的precision_score slot:{}".format(p_score))
        print("整体验证集上的recall_score slot:{}".format(re_score))
        print(seq_classification_report(all_tag, all_pre))

        if is_for_slot == False:
            # overall
            sentences_overall = 0
            sentences_overall_90 = 0
            for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1):
                if one_intent and one_slot_f1 == 1.0:
                    sentences_overall += 1

                if one_intent and one_slot_f1 >= 0.95:
                    sentences_overall_90 += 1

            sentences_overall_sklearn = 0
            sentences_overall_sklearn_90 = 0
            for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1_sklearn):
                if one_intent and one_slot_f1 == 1.0:
                    sentences_overall_sklearn += 1

                if one_intent and one_slot_f1 >= 0.95:
                    sentences_overall_sklearn_90 += 1

            print("sentences overall :{}".format(sentences_overall / len(sentences_acc)))
            print("sentences sentences_overall_95 :{}".format(sentences_overall_90 / len(sentences_acc)))
            print("sentences overall_sklearn :{}".format(sentences_overall_sklearn / len(sentences_acc)))
            print(
                "sentences sentences_overall_sklearn_95 :{}".format(sentences_overall_sklearn_90 / len(sentences_acc)))

            print("测试集合slot_f1: {},sklearn_slot_f1+O: {}, id_acc: {}, overall: {}".format(f1, f1_sklearn,
                                                                                              accuracy_score(
                                                                                                  ys_intent_list,
                                                                                                  pre_id),
                                                                                              sentences_overall / len(
                                                                                                  sentences_acc)))
        else:
            print("测试集合slot_f1: {},sklearn_slot_f1+O: {}".format(f1, f1_sklearn))

    # return train_losses, train_acces, eval_acces


if __name__ == "__main__":
    # setting param
    batch_size = config.batch_size
    lr = config.lr
    epoch = config.epoch
    max_size = config.max_size
    device = config.device
    is_CRF = config.is_CRF
    is_for_slot = config.is_for_slot  # True is only train slot task, False is jointly train slot and intent tasks

    # 加载数据集 load dataset
    with open(config.data_pkl_file_path, "rb") as fp:
        all_data = pickle.load(fp)

    # 加载词表
    with open(config.data_vocab_dic_pkl_file_path, "rb") as fp:
        all_vocab_data = pickle.load(fp)

    print("current training {} dataset".format(config.dataset))
    if "multi" not in config.dataset:
        all_data = all_data[config.dataset]
        datas_train, label_intent_train, label_slot_train = all_data['train'][0], all_data['train'][1], all_data['train'][2]
        datas_valid, label_intent_valid, label_slot_valid = all_data['valid'][0], all_data['valid'][1], all_data['valid'][2]
        datas_test, label_intent_test, label_slot_test = all_data['test'][0], all_data['test'][1], all_data['test'][2]

        all_vocab_dic = all_vocab_data[config.dataset]

    else:
        cur_type, cur_lingual = config.dataset.split("-")
        all_data = all_data[cur_type][cur_lingual]
        datas_train, label_intent_train, label_slot_train = all_data['train'][0], all_data['train'][1], all_data['train'][2]
        datas_valid, label_intent_valid, label_slot_valid = all_data['valid'][0], all_data['valid'][1], all_data['valid'][2]
        datas_test, label_intent_test, label_slot_test = all_data['test'][0], all_data['test'][1], all_data['test'][2]

        all_vocab_dic = all_vocab_data[cur_type][cur_lingual]

    # 词表 及 label 相关 id对应信息
    word_2_id = all_vocab_dic['vocab_dic']
    id_2_word = all_vocab_dic['id_2_word']
    label_intent_2_id = all_vocab_dic['label_intent_2_id']
    id_2_label_intent = all_vocab_dic['id_2_label_intent']
    label_slot_2_id = all_vocab_dic['label_slot_2_id']
    id_2_label_slot = all_vocab_dic['id_2_label_slot']

    word_size = len(word_2_id)  # 词表len
    intent_label_size = len(label_intent_2_id)
    slot_label_size = len(label_slot_2_id)

    # data loader
    train_dataset = MyDatasetWordPiece(datas_train, label_intent_train, label_slot_train, word_2_id,
                                       label_intent_2_id, label_slot_2_id, max_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.batch_data_process)

    dev_dataset = MyDatasetWordPiece(datas_valid, label_intent_valid, label_slot_valid, word_2_id,
                                     label_intent_2_id, label_slot_2_id, max_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False,
                                collate_fn=dev_dataset.batch_data_process)

    test_dataset = MyDatasetWordPiece(datas_test, label_intent_test, label_slot_test, word_2_id,
                                      label_intent_2_id,
                                      label_slot_2_id,
                                      max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                                 collate_fn=test_dataset.batch_data_process)
    # test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False,
    #                              collate_fn=test_dataset.batch_data_process)

    # 模型定义
    model = MyBertFirstWordPiece(intent_label_size, slot_label_size)  # sub-words 只用第一个piece

    # model = MyBertAttnBPWordPiece(intent_label_size, slot_label_size)

    # model = MyBertAttnBPWordPieceCRF(intent_label_size, slot_label_size)

    print("choose pre-training model: {}".format(config.model_name))

    # trainer
    train(model=model, train_dataloader=train_dataloader, valid_dataloader=dev_dataloader,
          test_dataloader=test_dataloader, device=device,
          batch_size=batch_size, num_epoch=epoch, lr=lr, optim=config.optim,
          is_CRF=is_CRF, is_for_slot=is_for_slot)
