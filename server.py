# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/9
@Time    : 9:52
@File    : server.py
@Function: XX
@Other: XX
"""
import copy
import json
import os
import torch
import time
import socket
import numpy as np
from flask import Flask, request
from gevent import pywsgi
from transformers import BertTokenizer
from utils.bert_ner_model import BertNerModel
from utils.utils import load_model_and_parallel, get_entity


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def test_torch():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    # ip获取
    addrs = socket.getaddrinfo(socket.gethostname(), None)
    myip = [item[4][0] for item in addrs if ':' not in item[4][0]][0]
    return myip


def encode(text):
    tokens = [i for i in text]
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=args.max_seq_len,
                                        padding='max_length',
                                        truncation='longest_first',
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    for i, token in enumerate(tokens):
        if token == ' ':
            encode_dict['input_ids'][i + 1] = 99
    # torch.from_numpy() 数组转张量 且二者共享内存 对张量进行修改比如重新赋值 那么原始数组也会相应发生改变
    token_ids = torch.from_numpy(np.array(encode_dict['input_ids'], dtype=np.int64)).unsqueeze(0).to(device)
    attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0).to(device)
    token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'], dtype=np.int64)).unsqueeze(0).to(device)
    return token_ids, attention_masks, token_type_ids


def decode(token_ids, attention_masks, token_type_ids, if_label=False):
    logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
    if args.use_crf == 'True':
        output = logits
    else:
        output = logits.detach().cpu().numpy()
        output = np.argmax(output, axis=2)
    if if_label:
        return [id2label[i] for i in output[0]][1:-1]
    pred_entities = get_entity([id2label[i] for i in output[0]][1:-1])  # 1:-1 剔除 CLS 和 SEP
    return pred_entities


def encode_batch(texts):
    """

    :param texts: list of str
    :return:
    """
    assert type(texts) == type([1, 2])
    # 一个比较简单的实现是 按照list直接喂给 tokenizer 然后补空格
    # 本函数的实现是 每个都独立生成 然后拼一起
    token_ids = "空变量"
    attention_masks = torch.empty(1, 2)
    token_type_ids = torch.empty(1, 2)
    for text in texts:
        token_id, attention_mask, token_type_id = encode(text)
        if token_ids == "空变量":
            token_ids = token_id
            attention_masks = attention_mask
            token_type_ids = token_type_id
        else:
            token_ids = torch.cat((token_ids, token_id), dim=0)
            attention_masks = torch.cat((attention_masks, attention_mask), dim=0)
            token_type_ids = torch.cat((token_type_ids, token_type_id), dim=0)
    return token_ids, attention_masks, token_type_ids


def decode_batch(token_ids, attention_masks, token_type_ids, if_label=False):
    """

    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :param if_label: 是否按照label输出
    :return:
    """
    logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
    if args.use_crf == 'True':
        output = logits
    else:
        output = logits.detach().cpu().numpy()
        output = np.argmax(output, axis=2)
    # 批量输入的时候 就不能只取1了
    results = []
    for y_pre in output:
        results.append(get_entity([id2label[i] for i in y_pre][1:-1]))
    return results


test_torch()
model_name = 'albert_base_bilstm_crf_adver_seed1024_2022-11-29_fxjg'  # 24这个可以！ 27也可以 44亦可  64 ok
data_path = './data/fxjg'

# model_name = 'bert_base_bilstm_crf_adver_seed1024_2022-10-20_ktgg'
# data_path = './data/ktgg'


args_path = './checkpoints/{}/args.json'.format(model_name)
model_path = './checkpoints/{}/model_best.pt'.format(model_name)
port = 8092

with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
model = BertNerModel(args)
model, device = load_model_and_parallel(model, args.gpu_ids, model_path)
model = model.to(device)
model.eval()
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
app = Flask(__name__)


# manager = Manager(app)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msg = request.get_data()
        msg = msg.decode('utf-8')
        # print(msg)
        token_ids, attention_masks, token_type_ids = encode(msg)
        # 1: {'label': B-AAA, 'content': '我'}
        # 2: ['我爱中国', 'label', start, end]
        # 3: ['B','I','O']
        return_type = 3
        if return_type == 1:
            entities = decode(token_ids, attention_masks, token_type_ids, True)  # False时返回实体
            # print(entities)
            entities = [{'label': j, 'content': msg[i]} for i, j in enumerate(entities)]
        elif return_type == 2:
            entities = decode(token_ids, attention_masks, token_type_ids, False)  # False时返回实体
            entities = [[msg[item[1]:item[2]], item[0], item[1], item[2]] for item in entities]
        elif return_type == 3:
            entities = decode(token_ids, attention_masks, token_type_ids, True)  # False时返回实体
        else:
            entities = []
        res = json.dumps(entities, ensure_ascii=False)
        return res
    except Exception as e:
        return str(e)


@app.route('/prediction_batch', methods=['POST'])
def prediction_batch():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        msgs = msgs.decode('utf-8')
        # print(msg)
        msgs = json.loads(msgs)
        results = []
        count = 10  # 控制小batch推理
        for index in range(len(msgs) // count + 1):

            msg = msgs[index * count: index * count + count]
            if len(msg) == 0:
                continue
            token_ids, attention_masks, token_type_ids = encode_batch(msg)
            entities = decode_batch(token_ids, attention_masks, token_type_ids, False)  # False时返回实体
            for i, items in enumerate(entities):
                results.append([[msg[i][item[1]:item[2]], item[0], item[1], item[2]] for item in items])
        res = json.dumps(results, ensure_ascii=False)
        torch.cuda.empty_cache()
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, threaded=False, debug=True)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port))
    server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
