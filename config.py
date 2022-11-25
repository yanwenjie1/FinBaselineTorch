# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/5
@Time    : 13:05
@File    : config.py
@Function: XX
@Other: XX
"""
import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()  # 命令行命令下 解析配置信息
        return parser

    @staticmethod
    def initialize(parser):  # 初始化配置信息
        # args for path
        # chinese-bert-wwm-ext  chinese-albert-base-cluecorpussmall
        # chinese-roberta-wwm-ext-large chinese-roberta-wwm-ext
        # chinese-electra-180g-base-discriminator
        # chinese-electra-180g-small-discriminator
        # chinese-nezha-base
        parser.add_argument('--bert_dir', default='../model/chinese-albert-base-cluecorpussmall/',
                            help='pre train model dir for uer')
        parser.add_argument('--model_name', type=str, default='albert_base',
                            help='模型名字')
        parser.add_argument('--data_dir', default='./data/fxjg/',
                            help='data dir for uer')
        parser.add_argument('--data_name', type=str, default='fxjg',
                            help='数据集名字')

        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--log_dir', default='./logs/',
                            help='log dir for uer')
        # other args
        parser.add_argument('--num_tags', default=10, type=int,
                            help='number of tags')
        parser.add_argument('--seed', type=int, default=1024, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=512, type=int)

        parser.add_argument('--eval_batch_size', default=16, type=int)

        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=100, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.5, type=float,
                            help='the drop out probability of pre train model ')

        # 2e-5
        parser.add_argument('--lr', default=2e-5, type=float,
                            help='bert学习率')
        # 2e-3
        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='bi-lstm和多层感知机学习率')
        # 2e-3
        parser.add_argument('--crf_lr', default=2e-3, type=float,
                            help='条件随机场学习率')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=16, type=int)
        parser.add_argument('--use_lstm', type=str, default='True',
                            help='是否使用BiLstm')
        parser.add_argument('--lstm_hidden', default=198, type=int,
                            help='lstm隐藏层节点数')
        parser.add_argument('--num_layers', default=2, type=int,
                            help='lstm层数大小 形成堆叠lstm')
        parser.add_argument('--dropout', default=0.3, type=float,
                            help='对多层lstm的输出dropout')
        parser.add_argument('--use_crf', type=str, default='True',
                            help='是否使用Crf')
        parser.add_argument('--use_advert_train', type=str, default='True',
                            help='是否使用对抗训练')
        parser.add_argument('--advert_train_epsilon', type=float, default=1.0,
                            help='PGD超参')
        parser.add_argument('--advert_train_alpha', type=float, default=0.3,
                            help='PGD超参')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
