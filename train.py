# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/8
@Time    : 9:52
@File    : train.py
@Function: XX
@Other: XX
"""
import datetime
import os
import logging
import numpy as np
import torch
from utils.utils import set_seed, set_logger, save_json
from utils.bert_ner_model import BertForNer
from utils.bert_classify_model import BertForClassify
import config
import json
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import BertTokenizer
import pickle


args = config.Args().get_parser()
logger = logging.getLogger(__name__)


class NerDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index], 'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index], 'labels': self.labels[index]}

        return data


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


if __name__ == '__main__':
    set_seed(args.seed)
    if args.use_lstm == 'True':
        if args.use_crf == 'True':
            args.model_name = '{}_bilstm_crf'.format(args.model_name)
        else:
            args.model_name = '{}_bilstm'.format(args.model_name)
    else:
        if args.use_crf == 'True':
            args.model_name = '{}_crf'.format(args.model_name)
        else:
            pass
    if args.use_advert_train == 'True':
        args.model_name = args.model_name + '_adver'
    args.model_name = args.model_name + '_seed' + str(args.seed) + '_' + str(datetime.date.today())

    set_logger(os.path.join(args.log_dir, '{}_{}.log'.format(args.model_name, args.data_name)))

    if args.data_name == "ktgg":
        args.max_seq_len = 256
        args.train_batch_size = args.eval_batch_size = 32
        # args.crf_lr = 2e-5
        args.num_layers = 1
        args.train_epochs = 50
        data_path = args.data_dir
        with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)
        logger.info(args)
        with open(os.path.join(data_path, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(data_path, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

        save_json('./checkpoints/{}_{}/'.format(args.model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, id2label)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model_best.pt'.format(args.model_name, args.data_name)
        bertForNer.test(model_path)

        raw_text = '海林市人民法院民事审判第一庭定于2016年6月14日8时40分0秒，在本院第三审判庭开庭审理原告尹丽波与被告人张小磊，劳务合同纠纷' \
                   '一案。 特此公告。2016年6月12日。发布时间：2016年6月13日8时39分48秒。海林市人民法院民事审判第一庭定于2016年6月14日8' \
                   '时40分0秒，在本院第三审判庭开庭审理原告尹丽波与被告人张小磊，劳务合同纠纷一案。 特此公告。2016年6月12日。'
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)
    if args.data_name == "yjyc":
        args.max_seq_len = 512
        args.train_epochs = 100
        args.num_layers = 2
        args.train_batch_size = args.eval_batch_size = 12
        data_path = args.data_dir
        with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)
        logger.info(args)
        with open(os.path.join(data_path, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(data_path, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

        save_json('./checkpoints/{}_{}/'.format(args.model_name, args.data_name), vars(args), 'args')
        bertForNer = BertForNer(args, train_loader, dev_loader, id2label)
        bertForNer.train()

        model_path = './checkpoints/{}_{}/model_best.pt'.format(args.model_name, args.data_name)
        model_path = './checkpoints/albert_base_bilstm_crf_adver_seed1024_2022-10-23_yjyc/model_best.pt'
        bertForNer.test(model_path)

        raw_text = '重要内容提示：1、西安陕鼓动力股份有限公司(以下简称“公司”)预计2021年度实现营业收入0至1,058,670万元，' \
                   '比上年同期增加229,177万元至252,177万元，同比增长28.42%至31.27%。2、预计2021年度归属于上市公司股东的净利润为83,229' \
                   '万元至88,754万元，与上年同期相比增加14,743万元至20,268万元，同比增长21.53%至29.59%。3、预计2021年度归属于上市公司股东' \
                   '的扣除非经常性损益的净利润为64,925万元至70,450万元，与上年同期相比增加12,716万元至18,241万元，同比增长24.35%至34.94%。'
        logger.info(raw_text)
        bertForNer.predict(raw_text, model_path)
    if args.data_name == "hyfl":
        if_classify = False
        args.max_seq_len = 256
        args.train_batch_size = args.eval_batch_size = 36
        # args.crf_lr = 2e-5
        args.num_layers = 1
        args.train_epochs = 10
        data_path = args.data_dir
        with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)
        logger.info(args)
        with open(os.path.join(data_path, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(data_path, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

        save_json('./checkpoints/{}_{}/'.format(args.model_name, args.data_name), vars(args), 'args')

        model_path = './checkpoints/{}_{}/model_best.pt'.format(args.model_name, args.data_name)
        raw_text = '徐州万旺豪新材料有限公司@许可项目：包装装潢印刷品印刷（依法须经批准的项目，经相关部门批准后方可开展经营活' \
                   '动，具体经营项目以审批结果为准）一般项目：新材料技术推广服务；新材料技术研发；技术服务、技术开发、技术咨询' \
                   '、技术交流、技术转让、技术推广；技术推广服务；科技推广和应用服务；塑料制品销售；塑料制品制造；装卸搬运；运' \
                   '输货物打包服务；包装服务；有色金属压延加工；化工产品销售（不含许可类化工产品）；涂料销售（不含危险化学品）' \
                   '（除依法须经批准的项目外，凭营业执照依法自主开展经营活动）'
        if if_classify:
            bertForClassify = BertForClassify(args, train_loader, dev_loader, label_list)
            bertForClassify.train()
            bertForClassify.test(model_path)
            logger.info(raw_text)
            bertForClassify.predict(raw_text, model_path)
        else:
            bertForNer = BertForNer(args, train_loader, dev_loader, id2label)
            bertForNer.train()
            bertForNer.test(model_path)
            logger.info(raw_text)
            bertForNer.predict(raw_text, model_path)
    if args.data_name == "esg":
        args.max_seq_len = 64
        args.train_batch_size = args.eval_batch_size = 256
        # args.crf_lr = 2e-5
        args.num_layers = 1
        args.train_epochs = 20
        data_path = args.data_dir
        with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)
        logger.info(args)
        with open(os.path.join(data_path, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(data_path, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

        save_json('./checkpoints/{}_{}/'.format(args.model_name, args.data_name), vars(args), 'args')

        model_path = './checkpoints/{}_{}/model_best.pt'.format(args.model_name, args.data_name)
        raw_text = '能源使用@天然氣消耗量(千立方米)@2650.300000000千立方米'
        bertForClassify = BertForClassify(args, train_loader, dev_loader, label_list)
        bertForClassify.train()
        bertForClassify.test(model_path)
        logger.info(raw_text)
        bertForClassify.predict(raw_text, model_path)


