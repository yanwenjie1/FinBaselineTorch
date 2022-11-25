# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/7
@Time    : 8:45
@File    : process.py
@Function: XX
@Other: XX
"""
import os
import sys
import json
import logging
import pickle
from transformers import BertTokenizer
import random

now_workspace = os.getcwd()
pre_now = os.path.dirname(now_workspace)
pre_pre_now = os.path.dirname(pre_now)
sys.path.append(pre_pre_now)

import config
for item in sys.path:
    print('sys.path:' + item)

class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels


def load_data(filename):
    results = []
    labels_all = []
    with open(filename, encoding='utf-8') as f:
        contents = f.read()
    for content in contents.split('\n\n'):
        words = []
        labels = []
        if not content:
            continue
        for word_label in content.split('\n'):
            word, label = word_label.split('\t')
            words.append(word)
            labels.append(label)
            if label not in labels_all:
                labels_all.append(label)
        if len(words) <= args.max_seq_len - 2:
            results.append((words, labels))
        else:
            print("Too long !")
            print(len(words))
            partOfWords = []
            partOfLabels = []
            partOfResults = []
            for i in range(len(words)):
                partOfWords.append(words[i])
                partOfLabels.append(labels[i])
                if words[i] == '。' and labels[i] == 'O':
                    if len(partOfWords) <= args.max_seq_len - 2:
                        partOfResults.append((partOfWords[:], partOfLabels[:]))
                    partOfWords = []
                    partOfLabels = []
                elif i == len(words) - 1 and len(partOfWords) != 0:
                    if len(partOfWords) <= args.max_seq_len - 2:
                        partOfResults.append((partOfWords[:], partOfLabels[:]))
            # 把partOfResults拼接为尽量长的数据
            WsAll = []
            LsAll = []
            for (Ws, Ls) in partOfResults:
                if len(WsAll) + len(Ws) <= args.max_seq_len - 2:
                    WsAll.extend(Ws)
                    LsAll.extend(Ls)
                elif len(WsAll) <= args.max_seq_len - 2:
                    results.append((WsAll[:], LsAll[:]))
                    WsAll = Ws[:]
                    LsAll = Ls[:]
                else:
                    WsAll = Ws[:]
                    LsAll = Ls[:]
            if 0 < len(WsAll) <= args.max_seq_len - 2:
                results.append((WsAll[:], LsAll[:]))
    # [PAD] [CLS] [SEP]
    labels_all.sort()
    labels_all.insert(0, 'SEP')
    labels_all.insert(0, 'CLS')
    labels_all.insert(0, 'PAD')  # PAD和0对齐 与 O区分
    print('labels_all：', labels_all)
    print('labels_all_len：', len(labels_all))
    # if not os.path.exists(args.data_dir):
    #     os.makedirs(args.data_dir)
    with open(os.path.join(args.data_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(labels_all, ensure_ascii=False))
    return results, labels_all


def convert_examples_to_features(examples, labels_all, tokenizer: BertTokenizer):
    label_to_id = {v: k for k, v in enumerate(labels_all)}
    features = []
    for (words, labels) in examples:
        tokens = words[:]
        assert 0 < len(tokens) <= args.max_seq_len - 2, f'{len(tokens)}'
        label_ids = [label_to_id[i] for i in labels]
        label_ids.insert(0, label_to_id['CLS'])
        label_ids.append(label_to_id['SEP'])
        if len(label_ids) < args.max_seq_len:
            pad_length = args.max_seq_len - len(label_ids)
            label_ids = label_ids + [0] * pad_length
        assert len(label_ids) == args.max_seq_len, f'{len(label_ids)}'
        #  空格会变成 100 UNK 后面计划重写这个函数 让 空格和 99对应 即 unused99
        word_ids = tokenizer.encode_plus(text=tokens,
                                         max_length=args.max_seq_len,
                                         padding="max_length",
                                         truncation='longest_first',
                                         return_token_type_ids=True,
                                         return_attention_mask=True)
        token_ids = word_ids['input_ids']
        attention_masks = word_ids['attention_mask']
        token_type_ids = word_ids['token_type_ids']
        # 重写多麻烦！ 这里改下不就好了
        for i, token in enumerate(tokens):
            if token == ' ':
                token_ids[i + 1] = 99
        feature = BertFeature(
            # bert inputs
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            labels=label_ids,
        )
        features.append(feature)
    return features


if __name__ == '__main__':
    args = config.Args().get_parser()
    current_workspace = os.getcwd()
    args.data_dir = current_workspace
    print('os.getcwd():' + current_workspace)
    args.max_seq_len = 512
    my_examples, my_labels = load_data(os.path.join(args.data_dir, 'train_data_20221124.txt'))
    my_tokenizer = BertTokenizer(os.path.join('../../' + args.bert_dir, 'vocab.txt'))
    random.shuffle(my_examples)  # 打乱数据集
    print('样本量 ', len(my_examples))
    train_data = my_examples[int(len(my_examples) / 8):]
    dev_data = my_examples[:int(len(my_examples) / 8) + 1]

    # train_data.sort(key=lambda i: len(i[0]), reverse=True)  # 让训练集中长度相近的在一起
    train_data = convert_examples_to_features(train_data, my_labels, my_tokenizer)
    dev_data = convert_examples_to_features(dev_data, my_labels, my_tokenizer)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('train_data')), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('dev_data')), 'wb') as f:
        pickle.dump(dev_data, f)

