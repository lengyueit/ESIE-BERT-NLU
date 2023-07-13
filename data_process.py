import os
import pickle
import config

"""dataset process"""


def get_data(file, data_type):
    """
    获取数据
    :param file: root path
    :param data_type: train valid test
    :return: datas, label_intent, label_slot
    """
    data_file = file + "/" + data_type + "/seq.in"
    label_intent_file = file + "/" + data_type + "/label"
    label_slot_file = file + "/" + data_type + "/seq.out"

    with open(data_file, "r", encoding="utf8") as f:
        datas = []
        data = f.read()
        data = data.split("\n")
        for i in data:
            if i == "":
                continue
            datas.append(i.strip())

    with open(label_intent_file, "r", encoding="utf8") as f:
        label_intent = []
        data = f.read()
        data = data.split("\n")
        for i in data:
            if i == "":
                continue
            label_intent.append(i.strip())

    with open(label_slot_file, "r", encoding="utf8") as f:
        label_slot = []
        data = f.read()
        data = data.split("\n")
        for i in data:
            if i == "":
                continue
            label_slot.append(i.strip())

    return datas, label_intent, label_slot


def data_save_pkl(save_file):
    """
    保存数据到pkl
    :param save_file: file path
    :return:  none
    """
    data = {}
    data['multi'] = {}

    # 加载数据
    datas_train_atis, label_intent_train_atis, label_slot_train_atis = get_data(file="../data/atis", data_type="train")
    datas_valid_atis, label_intent_valid_atis, label_slot_valid_atis = get_data(file="../data/atis", data_type="valid")
    datas_test_atis, label_intent_test_atis, label_slot_test_atis = get_data(file="../data/atis", data_type="test")
    data['atis'] = {"train": [datas_train_atis, label_intent_train_atis, label_slot_train_atis],
                    "valid": [datas_valid_atis, label_intent_valid_atis, label_slot_valid_atis],
                    "test": [datas_test_atis, label_intent_test_atis, label_slot_test_atis]}

    datas_train_snips, label_intent_train_snips, label_slot_train_snips = get_data(file="../data/snips",
                                                                                   data_type="train")
    datas_valid_snips, label_intent_valid_snips, label_slot_valid_snips = get_data(file="../data/snips",
                                                                                   data_type="valid")
    datas_test_snips, label_intent_test_snips, label_slot_test_snips = get_data(file="../data/snips", data_type="test")
    data['snips'] = {"train": [datas_train_snips, label_intent_train_snips, label_slot_train_snips],
                     "valid": [datas_valid_snips, label_intent_valid_snips, label_slot_valid_snips],
                     "test": [datas_test_snips, label_intent_test_snips, label_slot_test_snips]}

    # 多语言
    datas_train_atis_multi_en, label_intent_train_atis_multi_en, label_slot_train_atis_multi_en = get_data(
        file="../data/en-ph-atis/en",
        data_type="train")
    datas_valid_atis_multi_en, label_intent_valid_atis_multi_en, label_slot_valid_atis_multi_en = get_data(
        file="../data/en-ph-atis/en", data_type="dev")
    datas_test_atis_multi_en, label_intent_test_atis_multi_en, label_slot_test_atis_multi_en = get_data(
        file="../data/en-ph-atis/en", data_type="test")

    data['multi']['en'] = {
        "train": [datas_train_atis_multi_en, label_intent_train_atis_multi_en, label_slot_train_atis_multi_en],
        "valid": [datas_valid_atis_multi_en, label_intent_valid_atis_multi_en, label_slot_valid_atis_multi_en],
        "test": [datas_test_atis_multi_en, label_intent_test_atis_multi_en, label_slot_test_atis_multi_en]}

    datas_train_atis_multi_ph, label_intent_train_atis_multi_ph, label_slot_train_atis_multi_ph = get_data(
        file="../data/en-ph-atis/ph",
        data_type="train")
    datas_valid_atis_multi_ph, label_intent_valid_atis_multi_ph, label_slot_valid_atis_multi_ph = get_data(
        file="../data/en-ph-atis/ph",
        data_type="dev")
    datas_test_atis_multi_ph, label_intent_test_atis_multi_ph, label_slot_test_atis_multi_ph = get_data(
        file="../data/en-ph-atis/ph",
        data_type="test")

    data['multi']['ph'] = {
        "train": [datas_train_atis_multi_ph, label_intent_train_atis_multi_ph, label_slot_train_atis_multi_ph],
        "valid": [datas_valid_atis_multi_ph, label_intent_valid_atis_multi_ph, label_slot_valid_atis_multi_ph],
        "test": [datas_test_atis_multi_ph, label_intent_test_atis_multi_ph, label_slot_test_atis_multi_ph]}

    # 写入文件
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)


def get_dict(all_data):
    """
    获取词表
    :param all_data: 数据
    :return: vocab_dic, id_2_word, label_intent_2_id, id_2_label_intent, label_slot_2_id, id_2_label_slot
    """
    dict_result = {}
    dict_result["multi"] = {}

    data_type = ['atis', 'snips', 'multi-en', 'multi-ph']
    for cur_type in data_type:
        word_2_id = {}
        label_intent_2_id = {}
        id_2_label_intent = []
        label_slot_2_id = {}
        id_2_label_slot = set()

        # data   label_intent   label_slot
        if "multi" not in cur_type:
            datas_train, label_intent_train, label_slot_train = all_data[cur_type]['train']
            datas_valid, label_intent_valid, label_slot_valid = all_data[cur_type]['valid']
            datas_test, label_intent_test, label_slot_test = all_data[cur_type]['test']
        else:
            cur_type, cur_lingual = cur_type.split("-")
            datas_train, label_intent_train, label_slot_train = all_data[cur_type][cur_lingual]['train']
            datas_valid, label_intent_valid, label_slot_valid = all_data[cur_type][cur_lingual]['valid']
            datas_test, label_intent_test, label_slot_test = all_data[cur_type][cur_lingual]['test']

        label_intent = set(label_intent_train + label_intent_valid + label_intent_test)
        label_slot = label_slot_train + label_slot_valid + label_slot_test

        # 词频统计
        for data in datas_train:
            for i in data.split(" "):
                word_2_id[i] = word_2_id.get(i, 0) + 1
        for data in datas_valid:
            for i in data.split(" "):
                word_2_id[i] = word_2_id.get(i, 0) + 1
        for data in datas_test:
            for i in data.split(" "):
                word_2_id[i] = word_2_id.get(i, 0) + 1

        # 构建词表
        id_2_word = sorted([i for i, v in word_2_id.items() if v >= 1], key=lambda v: v,
                           reverse=True)  # 首先是根据频次筛选，然后sort一下降序，然后取词表最大

        vocab_dic = {word_count: idx for idx, word_count in enumerate(id_2_word)}  ##从词表字典中找到我们需要的那些就可以了
        vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1})  ##然后更新两个字符，一个是unk字符，一个pad字符

        # 构建标签 intent
        id_2_label_intent = sorted(list(label_intent))
        label_intent_2_id = {v: i for i, v in enumerate(id_2_label_intent)}

        # 构建标签 slot
        for data in label_slot:
            for i in data.split(" "):
                id_2_label_slot.add(i)

        id_2_label_slot = sorted(list(id_2_label_slot))

        label_slot_2_id = {word_count: idx for idx, word_count in enumerate(id_2_label_slot)}
        # label_slot_2_id.update({config.UNK: len(label_slot_2_id), config.PAD: len(label_slot_2_id) + 1})
        # id_2_label_slot.append(config.UNK)
        # id_2_label_slot.append(config.PAD)
        label_slot_2_id.update({config.PAD: len(label_slot_2_id)})
        id_2_label_slot.append(config.PAD)

        # dict_result[cur_type] = vocab_dic, id_2_word, label_intent_2_id, id_2_label_intent, label_slot_2_id, id_2_label_slot
        if "multi" not in cur_type:
            dict_result[cur_type] = {
                "vocab_dic": vocab_dic,
                "id_2_word": id_2_word,
                "label_intent_2_id": label_intent_2_id,
                "id_2_label_intent": id_2_label_intent,
                "label_slot_2_id": label_slot_2_id,
                "id_2_label_slot": id_2_label_slot,
            }
        else:
            dict_result[cur_type][cur_lingual] = {
                "vocab_dic": vocab_dic,
                "id_2_word": id_2_word,
                "label_intent_2_id": label_intent_2_id,
                "id_2_label_intent": id_2_label_intent,
                "label_slot_2_id": label_slot_2_id,
                "id_2_label_slot": id_2_label_slot,
            }
    return dict_result


def data_vocab_save_pkl(save_file):
    """
    将词表保存进pkl
    :param save_file:
    :return:
    """
    # 构建词表 将所有train test valid 词追加进入
    with open(data_pkl_file_path, "rb") as fp:
        all_data = pickle.load(fp)

    dict_result = get_dict(all_data)
    # 写入文件
    with open(save_file, 'wb') as f:
        pickle.dump(dict_result, f)


if __name__ == "__main__":
    # # 加载数据
    data_pkl_file_path = config.data_pkl_file_path
    data_save_pkl(data_pkl_file_path)

    # 加载词表
    data_vocab_dic_pkl_file_path = config.data_vocab_dic_pkl_file_path
    data_vocab_save_pkl(data_vocab_dic_pkl_file_path)

    # 加载数据 多语言
    # data_pkl_file_path = config.data_pkl_file_path_multi
    # data_save_pkl(data_pkl_file_path)
    #
    # # 加载词表
    # data_vocab_dic_pkl_file_path = config.data_vocab_dic_pkl_file_path_multi
    # data_vocab_save_pkl(data_vocab_dic_pkl_file_path)
