# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2022/9/8
@Time    : 8:23
@File    : bert_ner_model.py
@Function: XX
@Other: XX
"""
import torch
import torch.nn as nn
import logging
import os
import pynvml
import numpy as np
from utils.bert_base_model import BaseModel
from torchcrf import CRF
from utils.utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model, get_entity
from utils.adversarial_training import PGD
import config
from transformers import BertTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BertNerModel(BaseModel):
    def __init__(self, args, **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob,
                                           model_name=args.model_name)
        self.args = args
        self.num_layers = args.num_layers  # 这里num_layers是同一个time_step的结构堆叠 Lstm堆叠层数与time step无关
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device  # 指示当前设备
        out_dims = self.bert_config.hidden_size  # 预训练模型的输出维度

        if args.use_lstm == 'True':  # 如果使用lstm层
            self.lstm = nn.LSTM(out_dims, self.lstm_hidden, self.num_layers, bidirectional=True, batch_first=True,
                                dropout=args.dropout)
            self.linear = nn.Linear(self.lstm_hidden * 2, args.num_tags)  # lstm之后的线性层
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.linear]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        else:
            # 在kwargs里 找到对应的值 如果有就返回 并且从 kwargs中删除掉 如果没有这个 key 就返回256
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            out_dims = mid_linear_dims
            # self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(out_dims, args.num_tags)  # 线性层 分类用
            # self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.mid_linear, self.classifier]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        if args.use_crf == 'True':
            self.crf = CRF(args.num_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self, token_ids, attention_masks, token_type_ids, labels):
        bert_outputs = self.bert_module(input_ids=token_ids,  # vocab 对应的id
                                        attention_mask=attention_masks,  # pad mask 情况
                                        token_type_ids=token_type_ids  # CLS *** SEP *** SEP 区分第一个和第二个句子
                                        )
        # 输出是namedtuple或字典对象  可以通过属性或序号访问模型输出结果
        # outputs[0]是last_hidden_state, 是基于token表示的， 对于实体命名、问答非常有用、实际包括四个维度[layers, batches, tokens, features]
        # outputs[1]是整个输入的合并表达， 形状为[1, representation_size], 提取整篇文章的表达， 不是基于token级别的
        # outputs一共四个属性、last_hidden_state, pooler_output, hidden_states, attentions
        # 增加 hidden_states 和 attentions 才会有输出产生
        # pooler_output的输出是由 hidden_states获取了cls标签后进行了dense 和 Tanh后的输出
        # 所以bert的model并不是简单的组合返回。一般来说，需要使用bert做句子级别的任务，可以使用pooled_output结果做baseline， 进一步的微调可以使用last_hidden_state的结果

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768] 有空的时候这里要看看   bert_outputs['last_hidden_state']
        # seq_out1 = bert_outputs[1]  # bert_outputs['pooler_output']

        batch_size = seq_out.size(0)

        if self.args.use_lstm == 'True':
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)  # 这里有空的时候也要看看
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1)  # [batchsize, max_len, num_tags]
        else:
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
            # seq_out = self.dropout(seq_out)
            seq_out = self.classifier(seq_out)  # [24, 256, 53] 没有lstm时 默认值设置的256

        if self.args.use_crf == 'True':
            logits = self.crf.decode(seq_out, mask=attention_masks)
            # print(self.crf.transitions)
            if labels is None:
                return logits
            loss = -self.crf(seq_out, labels, mask=attention_masks, reduction='mean')
            outputs = (loss,) + (logits,)
            return outputs
        else:
            logits = seq_out
            if labels is None:
                return logits
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            outputs = (loss,) + (logits,)
            return outputs


class BertForNer:
    def __init__(self, args, train_loader, dev_loader, idx2tag):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.idx2tag = idx2tag  # id 转 label
        model = BertNerModel(args)
        self.model, self.device = load_model_and_parallel(model, args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        # eval_steps = len(self.train_loader) // 5  # 每多少个step打印损失及进行验证 每一个epoch中评估5次
        best_f1 = 0.0
        if self.args.use_advert_train == 'True':
            pgd = PGD(self.model,
                      emb_name='word_embeddings.',
                      epsilon=self.args.advert_train_epsilon,
                      alpha=self.args.advert_train_alpha)
            K = 3
        for epoch in range(self.args.train_epochs):  # 训练epoch数 默认50
            bar = tqdm(self.train_loader)
            losses = []
            for batch_data in bar:
                self.model.train()
                # model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
                # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                loss, logits = self.model(batch_data['token_ids'], batch_data['attention_masks'],
                                          batch_data['token_type_ids'], batch_data['labels'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                # loss.backward(loss.clone().detach())
                loss.backward()  # 反向传播 计算当前梯度
                if self.args.use_advert_train == 'True':
                    pgd.backup_grad()  # 如果梯度不为空 把原始梯度存到self.grad_backup
                    # 对抗训练
                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
                        if t != K - 1:
                            self.model.zero_grad()  # 把所有模型参数的梯度置为0
                        else:
                            pgd.restore_grad()  # 在对抗的最后一次恢复一开始保存的梯度 这时候的embedding参数层也加了3次扰动!
                        loss_adv, logits_adv = self.model(batch_data['token_ids'], batch_data['attention_masks'],
                                                          batch_data['token_type_ids'], batch_data['labels'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数层

                # 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
                # optimizer.zero_grad()的作用是清除所有优化器的torch.Tensor的梯度 当模型只用了一个优化器时 是等价的
                # print('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                global_step += 1  # 必要时可以用来控制学习率衰减
            dev_loss, precision, recall, f1_score = self.dev()
            if f1_score > best_f1:
                save_model(self.args, self.model, self.args.model_name + '_' + self.args.data_name,
                           str(epoch) + '_{:.4f}'.format(f1_score))
                best_f1 = f1_score
            logger.info('[eval] epoch:{} loss:{:.6f} precision={:.6f} recall={:.6f} f1_score={:.6f} best_f1_score={:.6f}'
                        .format(epoch, dev_loss, precision, recall, f1_score, best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logger.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小

    def dev(self):
        self.model.eval()  # 切换到测试模式 通知dropout层和batch_norm层在train和val模式间切换
        # 在val模式下，dropout层会让所有的激活单元都通过，而batch_norm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值
        # val模式不会影响各层的梯度计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播
        with torch.no_grad():
            # 主要是用于停止自动求导模块的工作，以起到加速和节省显存的作用
            # 具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batch_norm层的行为
            # 如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation的结果
            # with torch.no_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储gradient）
            # 从而可以更快计算，也可以跑更大的batch来测试
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in self.dev_loader:
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                # dev_loss 当前loss
                # dev_logits 预测的标签
                dev_loss, dev_logits = self.model(dev_batch_data['token_ids'],
                                                  dev_batch_data['attention_masks'],
                                                  dev_batch_data['token_type_ids'],
                                                  dev_batch_data['labels'])
                tot_dev_loss += dev_loss.item()
                if self.args.use_crf == 'True':
                    batch_output = dev_logits
                else:
                    batch_output = dev_logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)
                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity([self.idx2tag[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity([self.idx2tag[i] for i in y_true_list]))
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            # print('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            return tot_dev_loss, precision, recall, f1

    def test(self, model_path):
        model = BertNerModel(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        # 根据label确定有哪些实体类
        tags = [item[1] for item in self.idx2tag.items()]
        tags.remove('O')
        tags.remove('SEP')
        tags.remove('CLS')
        tags.remove('PAD')
        tags = [v[2:] for v in tags]
        entitys = list(set(tags))
        entitys.sort()
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, logits = model(dev_batch_data['token_ids'],
                                  dev_batch_data['attention_masks'],
                                  dev_batch_data['token_type_ids'],
                                  dev_batch_data['labels'])
                if self.args.use_crf == 'True':
                    batch_output = logits
                else:
                    batch_output = logits.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=2)

                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity([self.idx2tag[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity([self.idx2tag[i] for i in y_true_list]))
                    X_all += len(R & T)
                    Y_all += len(R)
                    Z_all += len(T)
                    for item in R & T:
                        X[entitys_to_ids[item[0]]] += 1
                    for item in R:
                        Y[entitys_to_ids[item[0]]] += 1
                    for item in T:
                        Z[entitys_to_ids[item[0]]] += 1
        len1 = max(max([len(i) for i in entitys]), 4)
        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all
        str_log = '\n{:<10}{:<15}{:<15}{:<15}\n'.format('实体' + chr(12288) * (len1 - len('实体')), 'precision', 'recall',
                                                        'f1-score')
        str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format('全部实体' + chr(12288) * (len1 - len('全部实体')), precision,
                                                                recall, f1)
        # logger.info('all_entity: precision:{:.6f}, recall:{:.6f}, f1-score:{:.6f}'
        #             .format(precision, recall, f1))
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        for entity in entitys:
            str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format(entity + chr(12288) * (len1 - len(entity)),
                                                                    precision[entitys_to_ids[entity]],
                                                                    recall[entitys_to_ids[entity]],
                                                                    f1[entitys_to_ids[entity]])
        logger.info(str_log)

    def predict(self, raw_text, model_path):
        model = BertNerModel(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            tokens = [i for i in raw_text]
            # 参数is_pretokenized表示是否已经预分词化，如果为True，则输入的sequence和pair都应该为一个列表
            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=self.args.max_seq_len,
                                                padding='max_length',
                                                truncation='longest_first',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            for i, token in enumerate(tokens):
                if token == ' ':
                    encode_dict['input_ids'][i + 1] = 99
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'], dtype=np.int64)).unsqueeze(0)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'], dtype=np.int64)).unsqueeze(0)
            logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
            if self.args.use_crf == 'True':
                output = logits
            else:
                output = logits.detach().cpu().numpy()
                output = np.argmax(output, axis=2)
            pred_entities = get_entity([self.idx2tag[i] for i in output[0]][1:-1])  # 1:-1 剔除 CLS 和 SEP
            pred_entities = [[raw_text[item[1]:item[2]], item[0], item[1], item[2]] for item in pred_entities]
            logger.info(pred_entities)


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.bert_dir = '../' + args.bert_dir
    args.num_tags = 33
    args.use_lstm = 'True'
    args.use_crf = 'True'
    args.num_layers = 2
    model = BertNerModel(args)
    for name, weight in model.named_parameters():
        print(name)
