# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/13 14:44
# software: PyCharm

import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=0, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(sentence_num, key_size, bias=False)
        self.linear_q = nn.Linear(sentence_num, key_size, bias=False)
        self.linear_v = nn.Linear(sentence_num, hidden_size, bias=False)
        self.dim_k = np.power(key_size, 0.5)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
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
        return output, attn


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.softmax(self.linear(x))


class BertDialogueIntentClassification(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertDialogueIntentClassification, self).__init__(config)
        self.args = args

        self.sentence_num = self.args.sentence_num
        self.key_size = 120

        self.hidden_size = config.hidden_size
        self.label_num = len(self.config.id2label)

        self.bert = BertModel(config=config)  # Load pretrained bert
        self.att = SelfAttention(sentence_num=self.sentence_num, key_size=self.key_size, hidden_size=self.hidden_size)
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1)  # [embeddings_size, hidden_dim, layer_num]

        self.fc = FCLayer(self.hidden_size*3, self.label_num)

        self.loss_fct_ms = nn.MSELoss()
        self.loss_fct_cros = nn.CrossEntropyLoss()

    @staticmethod
    def get_sep_vec(hidden_output, sep_masks):
        """
        :param hidden_output: [batch_size, max_sen_len, embedding_size]
        :param sep_masks: [batch_size, sentence_num, max_sen_len]
        :return:   shape: [batch_size, sentence_num, embedding_size]
        """
        seq_vec = torch.bmm(sep_masks, hidden_output)
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

    def forward(self, input_ids, attention_mask, token_type_ids, labels, sep_masks, present_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # Average
        present_h = self.present_average(sequence_output, present_mask)

        # [SEP]
        seq_vec = self.get_sep_vec(sequence_output, sep_masks)  # [batch_size, sentence_num, embedding_size]

        # attention
        seq_vec = self.att(seq_vec)

        # lstm
        out, _ = self.lstm(seq_vec.transpose(0, 1))
        out = out[-1::]
        out = out.transpose(0, 1)
        out = out.squeeze(1)

        # W(cls + lstm) + b
        out = torch.cat([pooled_output, out, present_h], dim=-1)
        logits = self.fc(out)

        if labels is not None:
            if self.num_labels == 1:
                # loss_fct = nn.MSELoss()
                # loss = loss_fct(logits.view(-1), labels.view(-1))
                loss = self.loss_fct_ms(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = nn.CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = self.loss_fct_cros(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + logits

        return outputs  # (loss), logits
