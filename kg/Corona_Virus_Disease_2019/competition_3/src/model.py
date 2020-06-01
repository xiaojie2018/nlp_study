# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/13 14:44
# software: PyCharm

import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import numpy as np


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=0, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dim_k = np.power(key_size, 0.5)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)
        self.linear = nn.Linear(sentence_num, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None, lina=True):
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


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=False):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        # y = self.softmax(x1)
        y = self.sigmoid(x1)
        return y


class KGBrtEmbedding(BertPreTrainedModel):

    def __init__(self, config, args):
        super(KGBrtEmbedding, self).__init__(config)
        self.args = args
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.hidden_size = config.hidden_size
        self.key_size = self.hidden_size
        self.sentence_num = 3
        self.label_num = 1

        self.fc = FCLayer(self.hidden_size, 1)
        self.att = SelfAttention(sentence_num=self.sentence_num, key_size=self.key_size, hidden_size=self.hidden_size)
        self.bilstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, bidirectional=True)  # [embeddings_size, hidden_dim, layer_num]

        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()

    @staticmethod
    def get_sep_vec(hidden_output, sep_masks):
        """
        :param hidden_output: [batch_size, max_sen_len, embedding_size]
        :param sep_masks: [batch_size, sentence_num, max_sen_len]
        :return:   shape: [batch_size, sentence_num, embedding_size]
        """
        seq_vec = torch.bmm(sep_masks.float(), hidden_output)
        return seq_vec

    @staticmethod
    def present_average(hidden_output, present_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = present_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (present_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, sep_masks, label):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # [SEP]
        # seq_vec1 = self.get_sep_vec(sequence_output, sep_masks)  # [batch_size, sentence_num, embedding_size]

        # attention
        # seq_vec = self.att(seq_vec1)

        # lstm_output, (hn, cn) = self.bilstm(seq_vec1.transpose(0, 1))  # [sentence_num, batch_size, 2*embedding_size]

        logits = self.fc(pooled_output)  # [batch_size, 1]

        if label is not None:
            if self.label_num == 1:
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, label.float())
            else:
                loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))

            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits

