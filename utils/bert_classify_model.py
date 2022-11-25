# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/22
@Time    : 13:28
@File    : bert_classify_model.py
@Function: XX
@Other: XX
"""
import torch
import torch.nn as nn
import numpy as np
import logging
import pynvml
import os
import config
from tqdm import tqdm
from transformers import BertTokenizer
from utils.bert_base_model import BaseModel
from utils.utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model, get_entity
from utils.adversarial_training import PGD
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)

class BertClassifyModel(BaseModel):
    def __init__(self, args):
        super(BertClassifyModel, self).__init__(bert_dir=args.bert_dir,
                                                dropout_prob=args.dropout_prob,
                                                model_name=args.model_name)
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device  # 指示当前设备
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(out_dims, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert_module(input_ids=token_ids,  # vocab 对应的id
                                        attention_mask=attention_masks,  # pad mask 情况
                                        token_type_ids=token_type_ids  # CLS *** SEP *** SEP 区分第一个和第二个句子
                                        )
        # 输出是namedtuple或字典对象  可以通过属性或序号访问模型输出结果
        # 输入的维度是：input_ids:[batch_size, tokens] (tokens=max_len)
        # outputs[0]是last_hidden_state, 是基于token表示的， 对于实体命名、问答非常有用、实际包括四个维度[layers, batches, tokens, features]
        # 不获取全部的12层输出的条件下 是 [batches, tokens, features] outputs[1]是 [batches, features]
        # outputs[1]是整个输入的合并表达， 形状为[1, representation_size], 提取整篇文章的表达， 不是基于token级别的
        # outputs一共四个属性、last_hidden_state, pooler_output, hidden_states, attentions
        # 增加 hidden_states 和 attentions 才会有输出产生
        # pooler_output的输出是由 hidden_states获取了cls标签后进行了dense 和 Tanh后的输出 dense层是768*768的全连接, 就为了调整输出
        # 所以bert的model并不是简单的组合返回。一般来说，需要使用bert做句子级别的任务，可以使用pooled_output结果做baseline， 进一步的微调可以使用last_hidden_state的结果
        # 分类任务的时候 再乘 [features, num_tags]的线性层 实现 one_hot的输出

        # 常规
        # seq_out = bert_outputs[1]  # [batchsize, features] 有空的时候这里要看看   bert_outputs['pooler_output']
        # # seq_out1 = bert_outputs[1]  # bert_outputs['pooler_output']
        # seq_out = self.dropout(seq_out)
        # seq_out = self.classifier(seq_out)  # [batchsize, num_tags]

        # 平均池化
        seq_out = bert_outputs[0].mean(1)
        seq_out = self.dropout(seq_out)
        seq_out = self.classifier(seq_out)  # [batchsize, num_tags]

        return seq_out


class BertForClassify:
    def __init__(self, args, train_loader, dev_loader, labels):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.criterion = nn.CrossEntropyLoss()
        model = BertClassifyModel(self.args)
        self.model, self.device = load_model_and_parallel(model, args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train == 'True':
            pgd = PGD(self.model,
                      emb_name='word_embeddings.',
                      epsilon=self.args.advert_train_epsilon,
                      alpha=self.args.advert_train_alpha)
            K = 3
        for epoch in range(self.args.train_epochs):  # 训练epoch数 默认50
            bar = tqdm(self.train_loader)
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss = self.criterion(train_outputs, batch_data['labels'])
                bar.set_postfix(loss='%.4f' % loss.detach().item())
                loss.backward()  # 反向传播 计算当前梯度
                if self.args.use_advert_train == 'True':
                    pgd.backup_grad()
                    # 对抗训练
                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
                        if t != K - 1:
                            self.model.zero_grad()
                        else:
                            pgd.restore_grad()
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss = self.criterion(train_outputs_adv, batch_data['labels'])
                        bar.set_postfix(loss='%.4f' % loss.detach().item())
                        loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数
                # 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_norm 所谓的梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
            dev_loss, f1 = self.dev()
            logger.info('[eval] epoch:{} loss:{:.6f} f1_score={:.6f} best_f1_score={:.6f}'
                        .format(epoch, dev_loss, f1, best_f1))
            if f1 > best_f1:
                save_model(self.args, self.model, self.args.model_name + '_' + self.args.data_name,
                           str(epoch) + '_{:.4f}'.format(f1))
                best_f1 = f1
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logger.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            dev_outputs = []
            dev_targets = []
            for dev_data in self.dev_loader:
                for key in dev_data.keys():
                    dev_data[key] = dev_data[key].to(self.device)
                outputs = self.model(dev_data['token_ids'],
                                     dev_data['attention_masks'],
                                     dev_data['token_type_ids'])
                loss = self.criterion(outputs, dev_data['labels'])
                tot_dev_loss += loss.item()
                # flatten() 降维 默认降为1维
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(dev_data['labels'].cpu().detach().numpy().tolist())
            accuracy = accuracy_score(dev_targets, dev_outputs)
            # 二分类 用binary 可以用pos_label指定某一类的f1
            # macro 先算每个类别的f1 再算平均 对错误的分布比较敏感
            # micro 先算总体的TP FN FP 再算f1
            # 可以这样理解 有一类比较少 但是全错了 会严重影响macro 而不会太影响micro
            micro_f1 = f1_score(dev_targets, dev_outputs, average='micro')

        return tot_dev_loss, micro_f1

    def test(self, model_path):
        model = BertClassifyModel(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            test_outputs = []
            test_targets = []
            for dev_data in tqdm(self.dev_loader):
                for key in dev_data.keys():
                    dev_data[key] = dev_data[key].to(self.device)
                outputs = model(dev_data['token_ids'],
                                dev_data['attention_masks'],
                                dev_data['token_type_ids'])
                loss = self.criterion(outputs, dev_data['labels'])
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(dev_data['labels'].cpu().detach().numpy().tolist())
            micro_f1 = f1_score(test_targets, test_outputs, average='micro')
            logger.info('[test] loss:{:.6f} f1_score={:.6f}'
                        .format(total_loss, micro_f1))
            # report = classification_report(test_targets, test_outputs, target_names=self.labels)
            report = classification_report(test_targets, test_outputs)
            logger.info(report)

    def predict(self, raw_text, model_path):
        model = BertClassifyModel(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=self.args.max_seq_len,
                                                truncation='longest_first',
                                                padding="max_length",
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')  # pytorch
            for i, token in enumerate(tokens):
                if token == ' ':
                    encode_dict['input_ids'][i + 1] = 99
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'], dtype=np.int64))
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8))
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'], dtype=np.int64))
            outputs = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [self.labels[i] for i in outputs]
                logger.info(outputs)
            else:
                return logger.info('不好意思，没有识别出来。')


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.bert_dir = '../' + args.bert_dir
    args.num_tags = 33
    model = BertClassifyModel(args)
    for name, weight in model.named_parameters():
        print(name)