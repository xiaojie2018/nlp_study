# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm


from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from config import MODEL_CLASSES


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

        self.fc = FCLayer(bert_config.hidden_size, self.label_num)

        # loss
        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()

    def forward(self, input_ids, attention_mask, token_type_ids, label):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        if self.args.model_type in ["xlnet_base", "xlnet_mid", "electra_base_discriminator", "electra_base_generator", 
                                    "electra_small_discriminator", "electra_small_generator"]:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[0][:, 0, :]
        else:
            sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
            pooled_output = outputs[1]  # [CLS]  [batch_size, embedding_size]

        logits = self.fc(pooled_output)

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
