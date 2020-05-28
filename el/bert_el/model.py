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
    def __init__(self, input_dim, output_dim, dropout_rate=0.15, use_activation=False):
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
        # x2 = self.softmax(x1)
        #
        # y1 = self.linear1(x)
        # y2 = self.sigmo(y1)
        y3 = self.sigmoid(x1)
        return y3
        # return self.softmax(self.linear(x))


class FCLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5, use_activation=True):
        super(FCLayer1, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BertTwoSentenceSimilarityToEntityLink(BertPreTrainedModel):

    def __init__(self, config, args):
        super(BertTwoSentenceSimilarityToEntityLink, self).__init__(config)
        self.args = args

        self.NTN = args.ntn
        self.ntn_weight = torch.randn(config.hidden_size, config.hidden_size, requires_grad=True)

        self.sentence_num = self.args.sentence_num
        self.key_size = 768

        self.hidden_size = config.hidden_size
        self.label_num = 1

        self.bert = BertModel(config=config)  # Load pretrained bert
        self.att = SelfAttention(sentence_num=self.sentence_num, key_size=self.key_size, hidden_size=self.hidden_size)
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1)  # [embeddings_size, hidden_dim, layer_num]

        self.bilstm1 = nn.LSTM(self.hidden_size, self.hidden_size, 1, bidirectional=True)
        self.bilstm2 = nn.LSTM(self.hidden_size, self.hidden_size, 1, bidirectional=True)

        self.FC1 = FCLayer1(self.hidden_size * 2, self.hidden_size)

        self.FC2 = FCLayer1(self.hidden_size * 2, self.hidden_size)

        self.FC3 = FCLayer1(self.hidden_size * 2, self.hidden_size)

        self.FC4 = FCLayer1(self.hidden_size * 2, self.hidden_size)
        # self.cnn1 = nn.Conv1d()

        self.fc1 = FCLayer1(self.hidden_size*2, 300)
        self.fc = FCLayer(300, 1)

        self.loss_fct_ms = nn.MSELoss()
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

    def forward(self, input_ids1, attention_mask1, token_type_ids1, mention_masks,
                input_ids2, attention_mask2, token_type_ids2, sep_masks2,
                labels):
        """
        :param input_ids1:  [batch_size, max_sen_len1]
        :param attention_mask1:  [batch_size, max_sen_len1]
        :param token_type_ids1:  [batch_size, max_sen_len1]
        :param mention_masks: [batch_size, max_sen_len1]
        :param input_ids2: [batch_size, max_sen_len2]
        :param attention_mask2: [batch_size, max_sen_len2]
        :param token_type_ids2: [batch_size, max_sen_len2]
        :param sep_masks2: [batch_size, sentence_num2]
        :param labels: [batch_size, 1]
        :return:
        """
        outputs1 = self.bert(input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output1 = outputs1[0]  # [batch_size, max_sen_len1, embedding_size]
        pooled_output1 = outputs1[1]  # [CLS]  [batch_size, embedding_size]

        outputs2 = self.bert(input_ids2, attention_mask=attention_mask2, token_type_ids=token_type_ids2)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output2 = outputs2[0]  # [batch_size, max_sen_len2, embedding_size]
        pooled_output2 = outputs2[1]  # [CLS]  [batch_size, embedding_size]

        # BiLstm

        lstm_output1, (hn1, cn1) = self.bilstm1(sequence_output1.transpose(0, 1))  # [batch_size, max_sen_len1, 2*embedding_size]

        lstm_output2, (hn2, cn2) = self.bilstm2(sequence_output2.transpose(0, 1))  # [batch_size, max_sen_len1, 2*embedding_size]

        # Average mention
        mention_o = self.present_average(lstm_output1.transpose(0, 1), mention_masks)  # [batch_size, 2*embedding_size]

        mention_o = self.FC1(mention_o)  # [batch_size, embedding_size]

        out1 = torch.cat([pooled_output1, mention_o], dim=-1)

        out1 = self.FC2(out1)  # [batch_size, embedding_size]

        # [SEP]
        seq_vec = self.get_sep_vec(lstm_output2.transpose(0, 1), sep_masks2)  # [batch_size, sentence_num2, embedding_size]

        # attention
        seq_vec = self.FC4(seq_vec)
        seq_vec = self.att(seq_vec)  # [batch_size, embedding_size]

        out2 = torch.cat([pooled_output2, seq_vec], dim=-1)

        out2 = self.FC3(out2)  # [batch_size, embedding_size]

        # tensor network
        if self.NTN:
            out = out1.mm(self.ntn_weight).mm(out2.transpose(0, 1))  # [batch_size, batch_size]
            out = out.diag()
            logits = self.fc(out)  # [batch_size, 1]

        else:
            out = torch.cat([out1, out2], dim=-1)
            out = self.fc1(out)
            logits = self.fc(out)  # [batch_size, 1]

        if labels is not None:
            if self.label_num == 1:
                # loss_fct = nn.MSELoss()
                # loss = loss_fct(logits.view(-1), labels.view(-1))
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, labels.float())
            else:
                # loss_fct = nn.CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
                loss = self.loss_fct_cros(logits.view(-1, self.label_num), labels.view(-1))

            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits
