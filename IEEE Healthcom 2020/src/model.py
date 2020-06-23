# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/23 22:22
# software: PyCharm

import torch.nn as nn
import torch
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        y = self.softmax(x1)
        # y = self.sigmoid(x1)
        return y


class AttTensorNetwork(nn.Module):

    def __init__(self, args):
        super(AttTensorNetwork, self).__init__()

        self.hidden_size = args.hidden_size
        self.max_seq_len = args.max_seq_len
        self.embedding_size = args.embedding_size
        self.num_classes = 2

        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.dropout = 0.1

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)

        self.fc_cnn = FCLayer(self.num_filters * len(self.filter_sizes), self.num_classes)

        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input, label):
        out = input.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.fc_cnn(out)
        # loss = F.cross_entropy(out, label)
        if label is not None:
            if self.num_classes == 1:
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, label.float())
            else:
                # loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))
                loss = self.loss_fct_bce(logits, label)

            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits


