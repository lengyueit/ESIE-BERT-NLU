import pickle

"""数据集处理"""

# 读取数据
def get_data(file, type):
    data_file = file + "/" + type + "/seq.in"
    label_intent_file = file + "/" + type + "/label"
    label_slot_file = file + "/" + type + "/seq.out"

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


if __name__ == "__main__":
    save_file = "../data/data.pkl"
    data = {}

    # 加载数据
    datas_train_atis, label_intent_train_atis, label_slot_train_atis = get_data(file="../data/atis", type="train")
    datas_valid_atis, label_intent_valid_atis, label_slot_valid_atis = get_data(file="../data/atis", type="valid")
    datas_test_atis, label_intent_test_atis, label_slot_test_atis = get_data(file="../data/atis", type="test")
    data['atis'] = {"train": [datas_train_atis, label_intent_train_atis, label_slot_train_atis],
                    "valid": [datas_valid_atis, label_intent_valid_atis, label_slot_valid_atis],
                    "test": [datas_test_atis, label_intent_test_atis, label_slot_test_atis]}

    datas_train_snips, label_intent_train_snips, label_slot_train_snips = get_data(file="../data/snips", type="train")
    datas_valid_snips, label_intent_valid_snips, label_slot_valid_snips = get_data(file="../data/snips", type="valid")
    datas_test_snips, label_intent_test_snips, label_slot_test_snips = get_data(file="../data/snips", type="test")
    data['snips'] = {"train": [datas_train_snips, label_intent_train_snips, label_slot_train_snips],
                     "valid": [datas_valid_snips, label_intent_valid_snips, label_slot_valid_snips],
                     "test": [datas_test_snips, label_intent_test_snips, label_slot_test_snips]}

    # 写入文件
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
