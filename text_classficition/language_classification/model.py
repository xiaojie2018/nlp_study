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


def _get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def torch_device_one():
    return torch.tensor(1.).to(_get_device())


def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


unsup_criterion = nn.KLDivLoss(reduction='none')


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


class BertPool(nn.Module):
    def __init__(self, config):
        super().__init__()
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


class ClassificationModel(BertPreTrainedModel):

    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_classes

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)

        if self.args.is_uda_model:
            self.unsup_ratio = self.args.unsup_ratio

        self.bert_config_output_hidden_states = False
        # bert_config.output_attentions = True
        # bert_config.output_hidden_states = True
        if bert_config.output_hidden_states:
            self.bert_config_output_hidden_states = True

        super(ClassificationModel, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.pooling = BertPool(bert_config)

        # attention
        self.att = SelfAttention(sentence_num=34, key_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size)

        self.fc = FCLayer(bert_config.hidden_size, self.label_num)
        self.fc2 = FCLayer(bert_config.hidden_size * 2, self.label_num)
        # self.fc1 = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc2 = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc3 = FCLayer(bert_config.hidden_size, self.label_num)
        self.fc3 = FCLayer1(self.args.max_seq_len, 1)

        # loss
        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()
        self.loss_fct_bce1 = nn.BCELoss(reduction='none')

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, type="test", global_step=1, t_total=1):

        if self.args.is_uda_model and type == "train":
            num_sample = label.shape[0]
            assert num_sample % (1 + 2 * self.unsup_ratio) == 0
            sup_batch_size = num_sample // (1 + 2 * self.unsup_ratio)
            unsup_batch_size = sup_batch_size * self.unsup_ratio
            
            input_ids0 = input_ids[:sup_batch_size]
            attention_mask0 = attention_mask[:sup_batch_size]
            token_type_ids0 = token_type_ids[:sup_batch_size]

            label0 = label[:sup_batch_size]

            ori_start = sup_batch_size
            ori_end = ori_start + unsup_batch_size
            aug_start = sup_batch_size + unsup_batch_size
            aug_end = aug_start + unsup_batch_size

            ori_input_ids = input_ids[ori_start: ori_end]
            ori_attention_mask = attention_mask[ori_start: ori_end]
            ori_token_type_ids = token_type_ids[ori_start: ori_end]

            aug_input_ids = input_ids[aug_start: aug_end]
            aug_attention_mask = attention_mask[aug_start: aug_end]
            aug_token_type_ids = token_type_ids[aug_start: aug_end]

            input_ids01 = torch.cat((input_ids0, aug_input_ids), dim=0)
            attention_mask01 = torch.cat((attention_mask0, aug_attention_mask), dim=0)
            token_type_ids01 = torch.cat((token_type_ids0, aug_token_type_ids), dim=0)

            outputs01 = self.bert(input_ids01, attention_mask=attention_mask01, token_type_ids=token_type_ids01)  # sequence_output, pooled_output, (hidden_states), (attentions)

            outputs = (outputs01[0][:sup_batch_size], outputs01[1][:sup_batch_size])
            aug_outputs = (outputs01[0][sup_batch_size:], outputs01[1][sup_batch_size:])

        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.bert_config_output_hidden_states:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]
            all_hidden_states = outputs[2]

            pooled_output_11 = self.pooling(all_hidden_states[11])
            pooled_output_10 = self.pooling(all_hidden_states[10])
            # pooled_output_9 = self.pooling(all_hidden_states[9])

            logits1 = self.fc(pooled_output_11)
            logits2 = self.fc(pooled_output_10)
            # logits3 = self.fc(pooled_output_9)

        else:

            if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator",
                                        "electra_small_discriminator", "electra_small_generator"]:
                sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
                pooled_output = outputs[0][:, 0, :]
            else:
                sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
                pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # attention
        if self.args.is_attention:

            out, _ = self.att(sequence_output)
            # pooled_output = outputs[0][:, 0, :]
            out = self.fc3(out.transpose(1, 2))

            # logits0 = self.fc(out)

            out = torch.cat([pooled_output, out.squeeze(-1)], dim=-1)
            logits0 = self.fc2(out)

        else:

            logits0 = self.fc(pooled_output)
            
            logits0 = F.log_softmax(logits0, dim=-1)

        if type == "train" and self.args.is_uda_model:

            sup_loss = self.loss_fct_bce1(logits0, label0)

            sup_label = label0

            if self.args.tsa:

                label_ids = torch.tensor([1] * sup_label.shape[0]).to(_get_device())

                # sup_loss = torch.tensor([torch.mean(sup_loss[i]) for i in range(sup_loss.shape[0])]).to(_get_device())

                tsa_thresh = get_tsa_thresh(self.args.tsa, global_step, t_total, start=1. / logits0.shape[-1], end=1)
                larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold

                loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
                sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
                sup_loss = torch.mean(sup_loss)
            else:
                sup_loss = torch.mean(sup_loss)

            with torch.no_grad():
                ori_outputs = self.bert(ori_input_ids, attention_mask=ori_attention_mask, token_type_ids=ori_token_type_ids)
                # aug_outputs = self.bert(aug_input_ids, attention_mask=aug_attention_mask, token_type_ids=aug_token_type_ids)

                if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator",
                                            "electra_base_generator",
                                            "electra_small_discriminator", "electra_small_generator"]:
                    ori_sequence_output = ori_outputs[0]  # [batch_size, max_sen_len, embedding_size]
                    ori_pooled_output = ori_outputs[0][:, 0, :]
                    aug_sequence_output = aug_outputs[0]  # [batch_size, max_sen_len, embedding_size]
                    aug_pooled_output = aug_outputs[0][:, 0, :]
                else:
                    ori_sequence_output = ori_outputs[0]  # [batch_size, max_sen_len, embedding_size]
                    ori_pooled_output = ori_outputs[1]  # [CLS]  [batch_size, embedding_size]
                    aug_sequence_output = aug_outputs[0]  # [batch_size, max_sen_len, embedding_size]
                    aug_pooled_output = aug_outputs[1]  # [CLS]  [batch_size, embedding_size]

                ori_log_probs = self.fc(ori_pooled_output)

                ori_log_probs = F.log_softmax(ori_log_probs, dim=-1)

                aug_log_probs = self.fc(aug_pooled_output)

            if self.args.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_log_probs, dim=-1)[0] > self.args.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(ori_log_probs), dtype=torch.float32)

            unsup_loss_mask = unsup_loss_mask.to(_get_device())

            uda_softmax_temp = self.args.uda_softmax_temp if self.args.uda_softmax_temp > 0 else 1.
            aug_log_probs = F.log_softmax(aug_log_probs / uda_softmax_temp, dim=-1)

            # unsup_loss0 = unsup_criterion(aug_log_probs, ori_log_probs)

            unsup_loss = torch.sum(unsup_criterion(aug_log_probs, ori_log_probs), dim=-1)

            # unsup_loss02 = torch.mean(unsup_loss)

            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                     torch_device_one())
            # unsup_loss01 = torch.mean(unsup_loss0)

            final_loss = sup_loss + self.args.uda_coeff * unsup_loss

            return (final_loss,) + (logits0,)

        logits = logits0

        if label is not None:
            if self.label_num == 1:
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, label)
            else:
                # loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))
                loss = self.loss_fct_bce(logits, label)

            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits
