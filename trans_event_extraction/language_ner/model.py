# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm
from crf import CRF
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from config import MODEL_CLASSES
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import LabelSmoothingCrossEntropy, FocalLoss


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        x = self.softmax(x)
        return x


class PoolerStartLogits1(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits1, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        x = self.softmax(x)
        return x


class PoolerEndLogits1(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits1, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(-1)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        # x = self.softmax(x)
        return x


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=False):
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


class FCLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.9, use_activation=False):
        super(FCLayer1, self).__init__()
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
        y = self.linear(x)
        # y = self.softmax(x1)
        # y = self.sigmoid(x1)
        return y


class LanguageSoftmaxForNer(BertPreTrainedModel):

    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageSoftmaxForNer, self).__init__(bert_config)

        self.device1 = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        # self.dropout = nn.Dropout(self.args.dropout_rate)
        # self.classifier = nn.Linear(bert_config.hidden_size, self.label_num)
        self.loss_type = self.args.loss_type

        assert self.loss_type in ["lsr", 'focal', 'ce', 'bce', 'bce_with_log']

        if self.loss_type in ["lsr", 'focal', 'ce']:
            self.classifier = FCLayer1(bert_config.hidden_size, self.label_num)
        else:
            self.classifier = FCLayer(bert_config.hidden_size, self.label_num)

        if self.loss_type == 'lsr':
            self.loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
        elif self.loss_type == 'focal':
            self.loss_fct = FocalLoss(ignore_index=0)
        elif self.loss_type == 'bce':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'bce_with_log':
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, is_test=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator",
        #                             "electra_small_discriminator", "electra_small_generator"]:
        #     sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        #     pooled_output = outputs[0][:, 0, :]
        # else:
        #     sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        #     pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label is not None:
            # assert self.loss_type in ['lsr', 'focal', 'ce']
            # if self.loss_type == 'lsr':
            #     loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss(ignore_index=0)
            # else:
            #     loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]

                if self.loss_fct in ['bce', 'bce_with_log']:
                    label = label.view(-1)[active_loss]
                    label = label.unsqueeze(1).to("cpu")
                    active_labels = torch.zeros(active_logits.shape[0], self.num_labels).scatter_(1, label, 1).to(self.device1)
                else:
                    active_labels = label.view(-1)[active_loss]

                # active_logits = logits.view(-1, self.label_num)
                # active_labels = label.view(-1)
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.label_num), label.view(-1))
            outputs = (loss,) + outputs
        else:
            return logits
        return outputs  # (loss), scores, (hidden_states), (attentions)


class LanguageCrfForNer(BertPreTrainedModel):
    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageCrfForNer, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        # self.dropout = nn.Dropout(self.args.dropout_rate)
        # self.classifier = nn.Linear(bert_config.hidden_size, self.label_num)
        self.classifier = FCLayer1(bert_config.hidden_size, self.label_num)
        self.crf = CRF(num_tags=self.label_num, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, is_test=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if label is not None:
            loss = self.crf(emissions=logits, tags=label, mask=attention_mask)

            if is_test:
                tags = self.crf.decode(logits, attention_mask)
                outputs = (tags,)
                outputs = (-1 * loss,) + outputs
            else:
                outputs = (-1 * loss,) + outputs
        if label is None and not is_test:
            tags = self.crf.decode(logits, attention_mask)
            return tags
        return outputs  # (loss), scores


class LanguageSpanForNer(BertPreTrainedModel):
    def __init__(self, model_dir, args):

        self.args = args
        self.num_labels = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageSpanForNer, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.soft_label = True
        # self.num_labels = config.num_labels
        self.loss_type = self.args.loss_type

        # self.bert = BertModel(config)

        self.device1 = "cuda" if torch.cuda.is_available() else "cpu"

        self.dropout = nn.Dropout(self.args.dropout_rate)

        assert self.loss_type in ["lsr", 'focal', 'ce', 'bce', 'bce_with_log']

        if self.loss_type in ["lsr", 'focal', 'ce']:

            self.start_fc = PoolerStartLogits1(bert_config.hidden_size, self.num_labels)
            if self.soft_label:
                self.end_fc = PoolerEndLogits1(bert_config.hidden_size + self.num_labels, self.num_labels)
            else:
                self.end_fc = PoolerEndLogits1(bert_config.hidden_size + 1, self.num_labels)
        else:
            self.start_fc = PoolerStartLogits(bert_config.hidden_size, self.num_labels)
            if self.soft_label:
                self.end_fc = PoolerEndLogits(bert_config.hidden_size + self.num_labels, self.num_labels)
            else:
                self.end_fc = PoolerEndLogits(bert_config.hidden_size + 1, self.num_labels)

        if self.loss_type == 'lsr':
            self.loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
        elif self.loss_type == 'focal':
            self.loss_fct = FocalLoss(ignore_index=0)
        elif self.loss_type == 'bce':
            self.loss_fct = nn.BCELoss()
        elif self.loss_type == 'bce_with_log':
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)

        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # assert self.loss_type in ['lsr', 'focal', 'ce']
            # if self.loss_type == 'lsr':
            #     loss_fct = LabelSmoothingCrossEntropy()
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss()
            # else:
            #     loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            if self.loss_type in ['bce', 'bce_with_log']:
                active_start_labels = active_start_labels.unsqueeze(1).to("cpu")
                active_start_labels = torch.zeros(active_start_logits.shape[0], self.num_labels).scatter_(1, active_start_labels, 1).to(self.device1)

                active_end_labels = active_end_labels.unsqueeze(1).to("cpu")
                active_end_labels = torch.zeros(active_end_logits.shape[0], self.num_labels).scatter_(1, active_end_labels, 1).to(self.device1)

            start_loss = self.loss_fct(active_start_logits, active_start_labels)
            end_loss = self.loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs

