# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm


from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
import torch.nn as nn
from config import MODEL_CLASSES
import torch
import torch.nn.functional as F
import numpy as np


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        # y = self.softmax(x1)
        # y = self.sigmoid(x1)
        return x1


class FCLayer_softmax(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayer_softmax, self).__init__()
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


class FCLayer_sigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=True):
        super(FCLayer_sigmoid, self).__init__()
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


class BertPool(nn.Module):
    def __init__(self, config):
        super(BertPool, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=0, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
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


class FCLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer1, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class NTNLayer(nn.Module):
    def __init__(self, input_dim, output_k=2, dropout_rate=0., use_activation=True):
        super(NTNLayer, self).__init__()
        self.tanh = nn.Tanh()
        self.bilinear = nn.Bilinear(input_dim, input_dim, output_k, bias=False)
        self.linear = nn.Linear(2 * input_dim, output_k)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_activation = use_activation
        # self.linear1 = nn.Linear(output_k, 1, bias=False)
        # self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        o1 = self.bilinear(x1, x2)
        o2 = self.linear(torch.cat([x1, x2], dim=-1))
        o = self.tanh(o1 + o2)
        # o = self.linear1(o)
        # o = self.softmax(o)
        return o


class ClassificationModel(BertPreTrainedModel):

    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_classes

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)

        super(ClassificationModel, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        # attention
        # self.att = SelfAttention(sentence_num=34, key_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size)

        self.ntn_layer = NTNLayer(bert_config.hidden_size, int(bert_config.hidden_size/2), args.dropout_rate)

        self.label_classifier = FCLayer_softmax(bert_config.hidden_size + int(bert_config.hidden_size/2), self.label_num, args.dropout_rate)

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.entity_fc_layer1 = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)

        # loss
        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()
        self.loss_fct_mse = nn.MSELoss()
        self.loss_fct_bce1 = nn.BCELoss(reduction='none')

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, attention_mask_a, attention_mask_b, label=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator",
                                    "electra_small_discriminator", "electra_small_generator"]:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[0][:, 0, :]
        else:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # Average
        e1_h = self.entity_average(sequence_output, attention_mask_a)
        e2_h = self.entity_average(sequence_output, attention_mask_b)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer1(e2_h)

        o1 = self.ntn_layer(e1_h, e2_h)
        # Concat -> fc_layer
        # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        # concat_h = torch.cat([e1_h, e2_h], dim=-1)
        concat_h = torch.cat([pooled_output, o1], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if label is not None:
            if self.label_num == 1:
                loss = self.loss_fct_mse(logits.view(-1), label.view(-1))
            else:
                # loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))
                loss = self.loss_fct_bce(logits, label)

            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits, (hidden_states), (attentions)
