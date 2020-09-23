# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 23:04
# software: PyCharm


import torch
from torch import nn
import torch.nn.functional as F
import math
from collections import OrderedDict, namedtuple, defaultdict
import random
import transformer


class EventTable(nn.Module):

    def __init__(self, event_type, field_types, hidden_size):

        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.hidden_size = hidden_size
        self.num_fields = len(field_types)

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event

        self.field_cls_list = nn.ModuleList(
            # 0: NA, 1: trigger this field
            [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp


class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out


class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out


class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)


class MLP(nn.Module):
    """Implements Multi-layer Perception."""

    def __init__(self, input_size, output_size, mid_size=None, num_mid_layer=1, dropout=0.1):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer-1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(F.relu(mid_fc(x)))
        x = self.out_fc(x)
        return x

