# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 17:42
# software: PyCharm

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
import torch.nn.functional as F
from torch import nn
import transformer
from config import MODEL_CLASSES
from crf import CRFLayer
import torch


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


class NERModel(nn.Module):
    def __init__(self, config):
        super(NERModel, self).__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = NERTokenEmbedding(
            config.vocab_size, config.hidden_size,
            max_sent_len=config.max_sent_len, dropout=config.dropout
        )
        # Multi-layer Transformer Layers to Incorporate Contextual Information
        self.token_encoder = transformer.make_transformer_encoder(
            config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        )
        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(config.hidden_size, self.config.num_entity_labels)

    def forward(self, input_ids, input_masks,
                label_ids=None, train_flag=True, decode_flag=True):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        input_emb = self.token_embedding(input_ids)
        input_masks = input_masks.unsqueeze(-2)  # to fit for the transformer code
        batch_seq_enc = self.token_encoder(input_emb, input_masks)

        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc, seq_token_label=label_ids, batch_first=True,
                train_flag=train_flag, decode_flag=decode_flag
            )
        else:
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


class NERTokenEmbedding(nn.Module):
    """Add token position information"""
    def __init__(self, vocab_size, hidden_size, max_sent_len=256, dropout=0.1):
        super(NERTokenEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_sent_len, hidden_size)

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_ids):
        batch_size, sent_len = batch_token_ids.size()
        device = batch_token_ids.device

        batch_pos_ids = torch.arange(
            sent_len, dtype=torch.long, device=device, requires_grad=False
        )
        batch_pos_ids = batch_pos_ids.unsqueeze(0).expand_as(batch_token_ids)

        batch_token_emb = self.token_embedding(batch_token_ids)
        batch_pos_emb = self.pos_embedding(batch_pos_ids)

        batch_token_emb = batch_token_emb + batch_pos_emb

        batch_token_out = self.layer_norm(batch_token_emb)
        batch_token_out = self.dropout(batch_token_out)

        return batch_token_out
