# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/3 15:59
# software: PyCharm


import numpy as np
from competition_1.src.utils import DataProcess
import os
import torch.nn as nn
import torch


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=1, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dim_k = np.power(key_size, 0.5)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)
        self.linear = nn.Linear(sentence_num, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None, lina=False):
        """
        :param x:  [batch_size, max_seq_len, embedding_size]
        :param mask:
        :return:   [batch_size, embedding_size]
        """
        k = self.linear_k(x)
        q = self.linear_q(x)
        v = self.linear_v(x)
        # f = self.softmax(q.matmul(k.t()) / self.dim_k)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.dim_k
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        if lina:
            return self.tanh(self.linear(output.transpose(1, 2)).squeeze(-1))
        return output, attn


class Feature:

    def gen_prob(self, data):
        type_list = {"药物": 0, "症状": 1, "疾病": 2, "NoneType": 3, "检查科目": 4, "病毒": 5, "细菌": 6, "医学专科": 7}
        chars = []
        data_type = {}
        for d in data:
            if d[-1] not in data_type:
                data_type[d[-1]] = []
            data_type[d[-1]].append(d[0])

            for c in d[0]:
                if c not in chars:
                    chars.append(c)

        def tongji(c, v):
            num = 0
            for v1 in v:
                if c in v1:
                    num += 1
                # num += v1.count(c)
            return num

        data_type1 = {}
        for k, v in data_type.items():
            data_type1[k] = {}
            for c in chars:
                data_type1[k][c] = tongji(c, v)

        data_type1["all"] = {}
        for c in chars:
            data_type1["all"][c] = 0
            for k in type_list.keys():
                data_type1["all"][c] += data_type1[k][c]

        vectors = []
        for c in chars:
            v1 = [0.0]*len(type_list)

            for k, v in type_list.items():
                if data_type1[k][c] == 0:
                    v1[v] = 0.0
                else:
                    v1[v] = data_type1[k][c]/data_type1["all"][c]

            vectors.append(v1)
        chars_id = {c: i for i, c in enumerate(chars)}
        # vectors = np.array(vectors)
        return chars_id, vectors, len(type_list)


# from competition_1.src import train_file_path
train_file_path = "/home/hemei/xjie/bert_classification/ccks_7_1_competition_data/训练集"
data = DataProcess().read_train_data(os.path.join(train_file_path, "entity_type.txt"))
chars_id, vectors, vectors_size = Feature().gen_prob(data)


if __name__ == '__main__':

    from competition_1.src.utils import DataProcess
    from competition_1.src import train_file_path
    import os
    data = DataProcess().read_train_data(os.path.join(train_file_path, "entity_type.txt"))
    chars_id, vectors, vectors_size = Feature().gen_prob(data)

    tokens = [
        ['[CLS]', '低', '温', '性', '昏', '迷', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
        ['[CLS]', '阴', '道', '斜', '隔', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
        ['[CLS]', '苯', '唑', '西', '林', '钠', '胶', '囊', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
        ['[CLS]', '眶', '周', '瘀', '斑', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
         '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]

    tokens_vectors = []
    for token in tokens:
        t_v = []
        for t in token:
            if t in chars_id:
                t_v.append(vectors[chars_id[t]])
            else:
                t_v.append([0.0]*vectors_size)
        tokens_vectors.append(t_v)

    att = SelfAttention(key_size=8, hidden_size=8)

    tokens_vectors = torch.tensor(tokens_vectors)
    a = att(tokens_vectors)
    print(1)
