import torch
from model import *
from data import *
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics import classification_report as seq_classification_report, accuracy_score as seq_accuracy_score, \
    f1_score as seq_f1_score, precision_score as seq_precision_score, recall_score as seq_recall_score
import pickle
from tqdm import tqdm
import warnings
import config

warnings.filterwarnings("ignore")


# load data
def get_data(dataset, data_type):
    """
    获取 input intent_label slot_label
    :param dataset: ATIS or SNIPS
    :param data_type: train valid test
    :return:
    """
    data_file = "../data/data.pkl"
    with open(data_file, "rb") as f:
        datas = pickle.load(f)
    data = datas[dataset]

    return data[data_type][0], data[data_type][1], data[data_type][2]


# create vocab_dic
def get_dict(datas, label_intent, label_slot):
    word_2_id = {}
    label_intent_2_id = {}
    id_2_label_intent = []
    label_slot_2_id = {}
    id_2_label_slot = []

    for data in datas:
        for i in data.split(" "):
            word_2_id[i] = word_2_id.get(i, 0) + 1

    # 构建词表
    id_2_word = sorted([i for i, v in word_2_id.items() if v >= 0], key=lambda v: v,
                       reverse=True)  # 首先是根据频次筛选，然后sort一下降序，然后取词表最大

    vocab_dic = {word_count: idx for idx, word_count in enumerate(id_2_word)}  ##从词表字典中找到我们需要的那些就可以了
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})  ##然后更新两个字符，一个是unk字符，一个pad字符

    # 构建标签 intent
    id_2_label_intent = list(set(label_intent))
    id_2_label_intent = sorted(id_2_label_intent)
    label_intent_2_id = {v: i for i, v in enumerate(id_2_label_intent)}
    label_intent_2_id.update({UNK: len(label_slot_2_id)})
    id_2_label_intent.append(UNK)

    # 构建标签 slot
    for data in label_slot:
        for i in data.split(" "):
            label_slot_2_id[i] = word_2_id.get(i, 0) + 1

    id_2_label_slot = sorted([i for i, v in label_slot_2_id.items()], key=lambda v: v,
                             reverse=True)  # 首先是根据频次筛选，然后sort一下降序，然后取词表最大

    label_slot_2_id = {word_count: idx for idx, word_count in enumerate(id_2_label_slot)}
    label_slot_2_id.update({UNK: len(label_slot_2_id), PAD: len(label_slot_2_id) + 1})
    id_2_label_slot.append(UNK)
    id_2_label_slot.append(PAD)
    return vocab_dic, id_2_word, label_intent_2_id, id_2_label_intent, label_slot_2_id, id_2_label_slot


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
        train_slot_f1 = 0
        train_id_acc = 0
        all_slot_pre = []
        all_slot_tag = []
        all_id_pre = []
        all_id_tag = []
        sentences_acc = []
        sentences_f1 = []

        for batch in tqdm(train_dataloader, desc='训练'):
            xs, ys_intent, ys_slot, xs_len, masks, token_start_idxs, subword_lengths, masks_crf = batch

            if len(masks) > 0:
                masks = masks.to(device)
                # ys_slot = ys_slot[:, 1:]

            res_id, res_sf = model(xs, masks, token_start_idxs, subword_lengths, is_for_slot)

            optimizer.zero_grad()

            # 计算 id loss
            if is_for_slot == False:
                loss_id = model.cross_loss_intent(res_id, ys_intent)

            # 计算 sf loss
            if is_CRF:
                loss_sf = model.loss_fn(res_sf[:, 1:, ], ys_slot[:, 1:], masks_crf)
            else:
                res_sf_2D = res_sf.reshape(-1, res_sf.shape[-1])
                ys_slot_2D = ys_slot.reshape(-1)
                loss_sf = model.cross_loss_slot(res_sf_2D, ys_slot_2D)

            if is_for_slot:
                loss = loss_sf
            else:
                loss = loss_id + loss_sf

            loss.backward()
            optimizer.step()

            if is_for_slot == False:
                res_id_list = torch.argmax(res_id, dim=-1).detach().cpu().numpy().tolist()
                pre_sf_list = torch.argmax(res_sf, dim=-1).detach().cpu().numpy().tolist()
                ys_slot_list = ys_slot.cpu().numpy().tolist()
                ys_intent_list = ys_intent.cpu().numpy().tolist()
                xs_len = xs_len.cpu().numpy().tolist()

                # 计算 id acc
                all_id_pre += res_id_list
                all_id_tag += ys_intent_list

                for one_intent, one_pre in zip(ys_intent_list, res_id_list):
                    sentences_acc.append(one_intent == one_pre)
            else:
                pre_sf_list = torch.argmax(res_sf, dim=-1).detach().cpu().numpy().tolist()
                ys_slot_list = ys_slot.cpu().numpy().tolist()
                xs_len = xs_len.cpu().numpy().tolist()

            # 计算sf f1
            if is_CRF:
                pre_sf_list = model.crf.decode(res_sf[:, 1:, ], masks_crf)

            for one_pre, one_tag, one_len in zip(pre_sf_list, ys_slot_list, xs_len):
                if is_CRF is False:
                    one_pre = one_pre[1:one_len + 1]
                one_tag = one_tag[1:one_len + 1]

                one_pre = [id_2_label_slot[i] for i in one_pre]
                one_tag = [id_2_label_slot[i] for i in one_tag]

                all_slot_pre.append(one_pre)
                all_slot_tag.append(one_tag)
                # 评估 sentences
                sentences_f1.append(seq_f1_score([one_pre], [one_tag]))

            # break

        # 评估 overall
        if is_for_slot == False:
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
            print("epoch: {}, Loss: {}, slot_f1: {}".format(epoch + 1, loss.item(),
                                                            seq_f1_score(all_slot_tag, all_slot_pre)))

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

            pre_id, pre_sf = model(xs, masks, token_start_idxs, subword_lengths, is_for_slot)

            if is_for_slot == False:
                pre_id = torch.argmax(pre_id, dim=-1).detach().cpu().numpy().tolist()
                pre_sf_list = torch.argmax(pre_sf, dim=-1).detach().cpu().numpy().tolist()
                ys_slot_list = ys_slot.cpu().numpy().tolist()
                ys_intent_list = ys_intent.cpu().numpy().tolist()
            else:
                pre_sf_list = torch.argmax(pre_sf, dim=-1).detach().cpu().numpy().tolist()
                ys_slot_list = ys_slot.cpu().numpy().tolist()

            xs_len = xs_len.cpu().numpy().tolist()

            if is_for_slot == False:

                # 评估 id
                print(classification_report(ys_intent_list, pre_id))
                print("整体验证集上的acc intent :{}".format(accuracy_score(ys_intent_list, pre_id)))
                print("整体验证集上的pre_s intent :{}".format(precision_score(ys_intent_list, pre_id, average="macro")))
                print("整体验证集上的recall_score intent :{}".format(recall_score(ys_intent_list, pre_id, average="macro")))
                print("整体验证集上的f1 intent :{}".format(f1_score(ys_intent_list, pre_id, average="macro")))

                sentences_acc = []
                for one_intent, one_pre in zip(ys_intent_list, pre_id):
                    sentences_acc.append(one_intent == one_pre)

            # 评估 slot
            sentences_f1 = []
            sentences_f1_sklearn = []
            all_pre = []
            all_tag = []
            all_pre_sklearn = []
            all_tag_sklearn = []

            if is_CRF:
                pre_sf_list = model.crf.decode(pre_sf[:, 1:, ], masks_crf)

            for one_pre, one_tag, one_len in zip(pre_sf_list, ys_slot_list, xs_len):
                if is_CRF is False:
                    one_pre = one_pre[1:one_len + 1]

                one_tag = one_tag[1:one_len + 1]

                # sklearn
                all_pre_sklearn += one_pre
                all_tag_sklearn += one_tag

                # 评估 sklearn sentences
                sentences_f1_sklearn.append(f1_score(one_pre, one_tag, average='micro'))

                # seqveal
                one_pre = [id_2_label_slot[i] for i in one_pre]
                one_tag = [id_2_label_slot[i] for i in one_tag]

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

                if one_intent and one_slot_f1 >= 0.9:
                    sentences_overall_90 += 1

            sentences_overall_sklearn = 0
            sentences_overall_sklearn_90 = 0
            for one_intent, one_slot_f1 in zip(sentences_acc, sentences_f1_sklearn):
                if one_intent and one_slot_f1 == 1.0:
                    sentences_overall_sklearn += 1

                if one_intent and one_slot_f1 >= 0.9:
                    sentences_overall_sklearn_90 += 1

            print("sentences overall :{}".format(sentences_overall / len(sentences_acc)))
            print("sentences sentences_overall_90 :{}".format(sentences_overall_90 / len(sentences_acc)))
            print("sentences overall_sklearn :{}".format(sentences_overall_sklearn / len(sentences_acc)))
            print(
                "sentences sentences_overall_sklearn_90 :{}".format(sentences_overall_sklearn_90 / len(sentences_acc)))

            print("测试集合slot_f1: {},sklearn_slot_f1+O: {}, id_acc: {}, overall: {}".format(f1, f1_sklearn,
                                                                                          accuracy_score(ys_intent_list,
                                                                                                         pre_id),
                                                                                          sentences_overall / len(
                                                                                              sentences_acc)))
        else:
            print("测试集合slot_f1: {},sklearn_slot_f1+O: {}".format(f1, f1_sklearn))

    # return train_losses, train_acces, eval_acces


if __name__ == "__main__":
    batch_size = config.batch_size
    lr = config.lr
    epoch = config.epoch
    hidden_num = config.hidden_num
    word_embedding = config.word_embedding
    max_size = config.max_size
    device = config.device

    atis_or_snip = True  # True = atis , False = snip
    is_for_slot = True  # True is only train slot , False is jointly train

    # 加载数据
    if atis_or_snip:
        datas_train, label_intent_train, label_slot_train = get_data(dataset="atis", type="train")
        datas_valid, label_intent_valid, label_slot_valid = get_data(dataset="atis", type="valid")
        datas_test, label_intent_test, label_slot_test = get_data(dataset="atis", type="test")
    else:
        datas_train, label_intent_train, label_slot_train = get_data(dataset="snips", type="train")
        datas_valid, label_intent_valid, label_slot_valid = get_data(dataset="snips", type="valid")
        datas_test, label_intent_test, label_slot_test = get_data(dataset="snips", type="test")

    # 构建词表
    word_2_id, id_2_word, label_intent_2_id, id_2_label_intent, \
    label_slot_2_id, id_2_label_slot = get_dict(datas_train, label_intent_train, label_slot_train)

    word_size = len(word_2_id)  # 词表len
    intent_label_size = len(label_intent_2_id)
    slot_label_size = len(label_slot_2_id)

    # data loader
    train_dataset = MyDatasetWordPiece(datas_train, label_intent_train, label_slot_train, word_2_id,
                                       label_intent_2_id, label_slot_2_id, max_size,
                                       bert=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.batch_data_process)

    dev_dataset = MyDatasetWordPiece(datas_valid, label_intent_valid, label_slot_valid, word_2_id,
                                     label_intent_2_id, label_slot_2_id, max_size,
                                     bert=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False,
                                collate_fn=dev_dataset.batch_data_process)

    test_dataset = MyDatasetWordPiece(datas_test, label_intent_test, label_slot_test, word_2_id,
                                      label_intent_2_id,
                                      label_slot_2_id,
                                      max_size, bert=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
                                 collate_fn=test_dataset.batch_data_process)

    # model
    # model = MyLstm(word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, word_2_id["<PAD>"])
    # model = MyRNN(word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, word_2_id["<PAD>"])
    # model = MyGRU(word_size, word_embedding, hidden_num, intent_label_size, slot_label_size, word_2_id["<PAD>"])
    # model = MyBert(intent_label_size, slot_label_size)
    # model = MyBertMainWordPiece(intent_label_size, slot_label_size) # sub-words 加起来取平均
    # model = MyBertFirstWordPiece(intent_label_size, slot_label_size) # sub-words 只用第一个piece
    # model = MyBertAttnWordPiece(intent_label_size, slot_label_size)
    # model = MyBertAttnWordPieceCRF(intent_label_size, slot_label_size)
    # model = MyBertAttnBPWordPiece(intent_label_size, slot_label_size)
    model = MyBertAttnBPWordPieceCRF(intent_label_size, slot_label_size)
    train(model=model, train_dataloader=train_dataloader, valid_dataloader=dev_dataloader,
          test_dataloader=test_dataloader, device=device,
          batch_size=batch_size, num_epoch=epoch, lr=lr, optim='adamW', is_CRF=True, is_for_slot=is_for_slot)
