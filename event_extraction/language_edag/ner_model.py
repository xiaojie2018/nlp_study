# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 17:42
# software: PyCharm

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
import torch.nn.functional as F
from torch import nn

from config import MODEL_CLASSES


class LanguageForBasicNERModel(BertPreTrainedModel):

    def __init__(self, args):

        self.args = args
        self.num_entity_labels = args.num_entity_labels

        self.config_class, _, config_model = MODEL_CLASSES[args.pred_model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)

        super(LanguageForBasicNERModel, self).__init__(bert_config)

        self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)

        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, self.num_entity_labels)

    def forward(self, input_ids, input_masks, token_type_ids=None, label_ids=None, train_flag=True, decode_flag=True):

        batch_seq_enc, _ = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_masks)

        # [batch_size, seq_len, hidden_size]
        batch_seq_enc = self.dropout(batch_seq_enc)
        # [batch_size, seq_len, num_entity_labels]
        batch_seq_logits = self.classifier(batch_seq_enc)

        batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

        if train_flag:
            batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
            batch_label = label_ids.view(-1)
            # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
            ner_loss = F.nll_loss(batch_logp, batch_label, reduction='none')
            ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
        else:
            ner_loss = None

        if decode_flag:
            batch_seq_preds = batch_seq_logp.argmax(dim=-1)
        else:
            batch_seq_preds = None

        return batch_seq_enc, ner_loss, batch_seq_preds
