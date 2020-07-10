# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm


from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
import torch.nn as nn
from config import MODEL_CLASSES
import torch
import torch.nn.functional as F


def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.fc = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc1 = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc2 = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc3 = FCLayer(bert_config.hidden_size, self.label_num)

        # loss
        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()
        self.loss_fct_bce1 = nn.BCELoss(reduction='none')

    def forward(self, input_ids, attention_mask, token_type_ids, label, type="test", global_step=1, t_total=1):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)

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

        logits0 = self.fc(pooled_output)

        if type == "train":
            num_sample = label.shape[0]
            assert num_sample % (1 + 2 * self.unsup_ratio) == 0
            sup_batch_size = num_sample // (1 + 2 * self.unsup_ratio)
            unsup_batch_size = sup_batch_size * self.unsup_ratio

            sup_log_probs = logits0[:sup_batch_size]
            sup_label = label[:sup_batch_size]

            label_ids = torch.tensor([1]*sup_label.shape[0]).to(_get_device())

            ori_start = sup_batch_size
            ori_end = ori_start + unsup_batch_size
            aug_start = sup_batch_size + unsup_batch_size
            aug_end = aug_start + unsup_batch_size

            ori_log_probs = logits0[ori_start: ori_end]
            aug_log_probs = logits0[aug_start: aug_end]

            sup_loss = self.loss_fct_bce1(sup_log_probs, sup_label)

            if self.args.tsa:
                sup_loss = torch.tensor([torch.mean(sup_loss[i]) for i in range(sup_loss.shape[0])]).to(_get_device())

                tsa_thresh = get_tsa_thresh(self.args.tsa, global_step, t_total, start=1. / logits0.shape[-1], end=1)
                larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold

                loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
                sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
                sup_loss = torch.mean(sup_loss)
            else:
                sup_loss = torch.mean(sup_loss)

            if self.args.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_log_probs, dim=-1)[0] > self.args.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(ori_log_probs), dtype=torch.float32)

            unsup_loss_mask = unsup_loss_mask.to(_get_device())

            uda_softmax_temp = self.args.uda_softmax_temp if self.args.uda_softmax_temp > 0 else 1.
            aug_log_probs = F.log_softmax(aug_log_probs / uda_softmax_temp, dim=-1)

            unsup_loss = torch.sum(unsup_criterion(aug_log_probs, ori_log_probs), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                     torch_device_one())
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
