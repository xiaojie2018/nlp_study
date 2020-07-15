# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm


from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from config import MODEL_CLASSES
from torch.nn import CrossEntropyLoss

from loss import LabelSmoothingCrossEntropy, FocalLoss


class LanguageSoftmaxForNer(BertPreTrainedModel):

    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        super(LanguageSoftmaxForNer, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.classifier = nn.Linear(bert_config.hidden_size, self.label_num)
        self.loss_type = self.args.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, label):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator",
        #                             "electra_small_discriminator", "electra_small_generator"]:
        #     sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        #     pooled_output = outputs[0][:, 0, :]
        # else:
        #     sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        #     pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                active_logits = logits.view(-1, self.label_num)
                active_labels = label.view(-1)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.label_num), label.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
