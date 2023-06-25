from transformers import BertTokenizer

import pickle
from tqdm import tqdm


# 读取数据
def get_data(dataset, type):
    data_file = "./data.pkl"
    with open(data_file, "rb") as f:
        datas = pickle.load(f)
    data = datas[dataset]

    return data[type][0], data[type][1], data[type][2]


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("../pretrain-model/bert/bert-base-uncased/vocab.txt")

    atis_or_snip = True  # True = atis , False = snip

    wordpiece_count = 0

    if atis_or_snip:
        datas_train, label_intent_train, label_slot_train = get_data(dataset="atis", type="train")
        datas_valid, label_intent_valid, label_slot_valid = get_data(dataset="atis", type="valid")
        datas_test, label_intent_test, label_slot_test = get_data(dataset="atis", type="test")
    else:
        datas_train, label_intent_train, label_slot_train = get_data(dataset="snips", type="train")
        datas_valid, label_intent_valid, label_slot_valid = get_data(dataset="snips", type="valid")
        datas_test, label_intent_test, label_slot_test = get_data(dataset="snips", type="test")

    for i in tqdm(datas_train):
        for j in i.split(" "):
            cur_tokens = tokenizer.tokenize(j)
            if len(cur_tokens) > 1:
                wordpiece_count += 1

    print(wordpiece_count)
