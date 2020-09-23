# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 9:25
# software: PyCharm
import collections
import logging
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel, BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from config import MODEL_CLASSES
from event_type import common_fields, event_type_fields_list, event_type2event_class, BaseEvent
from helper import default_load_json, init_logger
import re
from collections import defaultdict, Counter
init_logger()
logger = logging.getLogger(__name__)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class Example(object):
    def __init__(self, annguid, detail_align_dict, only_inference=False):
        self.guid = annguid
        # [sent_text, ...]
        self.sentences = detail_align_dict['sentences']
        self.num_sentences = len(self.sentences)

        if only_inference:
            # set empty entity/event information
            self.only_inference = True
            self.ann_valid_mspans = []
            self.ann_mspan2dranges = {}
            self.ann_mspan2guess_field = {}
            self.recguid_eventname_eventdict_list = []
            self.num_events = 0
            self.sent_idx2srange_mspan_mtype_tuples = {}
            self.event_type2event_objs = {}
        else:
            # set event information accordingly
            self.only_inference = False

            # [span_text, ...]
            self.ann_valid_mspans = detail_align_dict['ann_valid_mspans']
            # span_text -> [drange_tuple, ...]
            self.ann_mspan2dranges = detail_align_dict['ann_mspan2dranges']
            # span_text -> guessed_field_name
            self.ann_mspan2guess_field = detail_align_dict['ann_mspan2guess_field']
            # [(recguid, event_name, event_dict), ...]
            self.recguid_eventname_eventdict_list = detail_align_dict['recguid_eventname_eventdict_list']
            self.num_events = len(self.recguid_eventname_eventdict_list)

            # for create ner examples
            # sentence_index -> [(sent_match_range, match_span, match_type), ...]
            self.sent_idx2srange_mspan_mtype_tuples = {}
            for sent_idx in range(self.num_sentences):
                self.sent_idx2srange_mspan_mtype_tuples[sent_idx] = []

            for mspan in self.ann_valid_mspans:
                for drange in self.ann_mspan2dranges[mspan]:
                    sent_idx, char_s, char_e = drange
                    sent_mrange = (char_s, char_e)

                    sent_text = self.sentences[sent_idx]
                    if sent_text[char_s: char_e] != mspan:
                        raise Exception('GUID: {} span range is not correct, span={}, range={}, sent={}'.format(
                            annguid, mspan, str(sent_mrange), sent_text
                        ))

                    guess_field = self.ann_mspan2guess_field[mspan]

                    self.sent_idx2srange_mspan_mtype_tuples[sent_idx].append(
                        (sent_mrange, mspan, guess_field)
                    )

            # for create event objects
            # the length of event_objs should >= 1
            self.event_type2event_objs = {}
            for mrecguid, event_name, event_dict in self.recguid_eventname_eventdict_list:
                event_class = event_type2event_class[event_name]
                event_obj = event_class()
                assert isinstance(event_obj, BaseEvent)
                event_obj.update_by_dict(event_dict, recguid=mrecguid)

                if event_obj.name in self.event_type2event_objs:
                    self.event_type2event_objs[event_obj.name].append(event_obj)
                else:
                    self.event_type2event_objs[event_name] = [event_obj]

    def __repr__(self):
        dee_str = 'Example (\n'
        dee_str += '  guid: {},\n'.format(repr(self.guid))

        if not self.only_inference:
            dee_str += '  span info: (\n'
            for span_idx, span in enumerate(self.ann_valid_mspans):
                gfield = self.ann_mspan2guess_field[span]
                dranges = self.ann_mspan2dranges[span]
                dee_str += '    {:2} {:20} {:30} {}\n'.format(span_idx, span, gfield, str(dranges))
            dee_str += '  ),\n'

            dee_str += '  event info: (\n'
            event_str_list = repr(self.event_type2event_objs).split('\n')
            for event_str in event_str_list:
                dee_str += '    {}\n'.format(event_str)
            dee_str += '  ),\n'

        dee_str += '  sentences: (\n'
        for sent_idx, sent in enumerate(self.sentences):
            dee_str += '    {:2} {}\n'.format(sent_idx, sent)
        dee_str += '  ),\n'

        dee_str += ')\n'

        return dee_str


class Feature(object):

    def __init__(self, guid, ex_idx, doc_token_id_mat, doc_token_mask_mat, doc_token_label_mat,
                 span_token_ids_list, span_dranges_list, event_type_labels, event_arg_idxs_objs_list,
                 valid_sent_num=None):
        self.guid = guid
        self.ex_idx = ex_idx  # example row index, used for backtracking
        self.valid_sent_num = valid_sent_num

        # directly set tensor for dee feature to save memory
        # self.doc_token_id_mat = doc_token_id_mat
        # self.doc_token_mask_mat = doc_token_mask_mat
        # self.doc_token_label_mat = doc_token_label_mat
        self.doc_token_ids = torch.tensor(doc_token_id_mat, dtype=torch.long)
        self.doc_token_masks = torch.tensor(doc_token_mask_mat, dtype=torch.uint8)  # uint8 for mask
        self.doc_token_labels = torch.tensor(doc_token_label_mat, dtype=torch.long)

        # sorted by the first drange tuple
        # [(token_id, ...), ...]
        # span_idx -> span_token_id tuple
        self.span_token_ids_list = span_token_ids_list
        # [[(sent_idx, char_s, char_e), ...], ...]
        # span_idx -> [drange tuple, ...]
        self.span_dranges_list = span_dranges_list

        # [event_type_label, ...]
        # length = the total number of events to be considered
        # event_type_label \in {0, 1}, 0: no 1: yes
        self.event_type_labels = event_type_labels
        # event_type is denoted by the index of event_type_labels
        # event_type_idx -> event_obj_idx -> event_arg_idx -> span_idx
        # if no event objects, event_type_idx -> None
        self.event_arg_idxs_objs_list = event_arg_idxs_objs_list

        # event_type_idx -> event_field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        self.event_idx2field_idx2pre_path2cur_span_idx_set = self.build_dag_info(self.event_arg_idxs_objs_list)

        # event_type_idx -> key_sent_idx_set, used for key-event sentence detection
        self.event_idx2key_sent_idx_set, self.doc_sent_labels = self.build_key_event_sent_info()

    def generate_dag_info_for(self, pred_span_token_tup_list, return_miss=False):
        token_tup2pred_span_idx = {
            token_tup: pred_span_idx for pred_span_idx, token_tup in enumerate(pred_span_token_tup_list)
        }
        gold_span_idx2pred_span_idx = {}
        # pred_span_idx2gold_span_idx = {}
        missed_span_idx_list = []  # in terms of self
        missed_sent_idx_list = []  # in terms of self
        for gold_span_idx, token_tup in enumerate(self.span_token_ids_list):
            if token_tup in token_tup2pred_span_idx:
                pred_span_idx = token_tup2pred_span_idx[token_tup]
                gold_span_idx2pred_span_idx[gold_span_idx] = pred_span_idx
                # pred_span_idx2gold_span_idx[pred_span_idx] = gold_span_idx
            else:
                missed_span_idx_list.append(gold_span_idx)
                for gold_drange in self.span_dranges_list[gold_span_idx]:
                    missed_sent_idx_list.append(gold_drange[0])
        missed_sent_idx_list = list(set(missed_sent_idx_list))

        pred_event_arg_idxs_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                pred_event_arg_idxs_objs_list.append(None)
            else:
                pred_event_arg_idxs_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    pred_event_arg_idxs = []
                    for gold_span_idx in event_arg_idxs:
                        if gold_span_idx in gold_span_idx2pred_span_idx:
                            pred_event_arg_idxs.append(
                                gold_span_idx2pred_span_idx[gold_span_idx]
                            )
                        else:
                            pred_event_arg_idxs.append(None)

                    pred_event_arg_idxs_objs.append(tuple(pred_event_arg_idxs))
                pred_event_arg_idxs_objs_list.append(pred_event_arg_idxs_objs)

        # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        pred_dag_info = self.build_dag_info(pred_event_arg_idxs_objs_list)

        if return_miss:
            return pred_dag_info, missed_span_idx_list, missed_sent_idx_list
        else:
            return pred_dag_info

    def get_event_args_objs_list(self):
        event_args_objs_list = []
        for event_arg_idxs_objs in self.event_arg_idxs_objs_list:
            if event_arg_idxs_objs is None:
                event_args_objs_list.append(None)
            else:
                event_args_objs = []
                for event_arg_idxs in event_arg_idxs_objs:
                    event_args = []
                    for arg_idx in event_arg_idxs:
                        if arg_idx is None:
                            token_tup = None
                        else:
                            token_tup = self.span_token_ids_list[arg_idx]
                        event_args.append(token_tup)
                    event_args_objs.append(event_args)
                event_args_objs_list.append(event_args_objs)

        return event_args_objs_list

    def build_key_event_sent_info(self):
        assert len(self.event_type_labels) == len(self.event_arg_idxs_objs_list)
        # event_idx -> key_event_sent_index_set
        event_idx2key_sent_idx_set = [set() for _ in self.event_type_labels]
        for key_sent_idx_set, event_label, event_arg_idxs_objs in zip(
                event_idx2key_sent_idx_set, self.event_type_labels, self.event_arg_idxs_objs_list
        ):
            if event_label == 0:
                assert event_arg_idxs_objs is None
            else:
                for event_arg_idxs_obj in event_arg_idxs_objs:
                    sent_idx_cands = []
                    for span_idx in event_arg_idxs_obj:
                        if span_idx is None:
                            continue
                        span_dranges = self.span_dranges_list[span_idx]
                        for sent_idx, _, _ in span_dranges:
                            sent_idx_cands.append(sent_idx)
                    if len(sent_idx_cands) == 0:
                        raise Exception('Event {} has no valid spans'.format(str(event_arg_idxs_obj)))
                    sent_idx_cnter = Counter(sent_idx_cands)
                    key_sent_idx = sent_idx_cnter.most_common()[0][0]
                    key_sent_idx_set.add(key_sent_idx)

        doc_sent_labels = []  # 1: key event sentence, 0: otherwise
        for sent_idx in range(self.valid_sent_num):  # masked sents will be truncated at the model part
            sent_labels = []
            for key_sent_idx_set in event_idx2key_sent_idx_set:  # this mapping is a list
                if sent_idx in key_sent_idx_set:
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)
            doc_sent_labels.append(sent_labels)

        return event_idx2key_sent_idx_set, doc_sent_labels

    @staticmethod
    def build_dag_info(event_arg_idxs_objs_list):
        # event_idx -> field_idx -> pre_path -> {span_idx, ...}
        # pre_path is tuple of span_idx
        event_idx2field_idx2pre_path2cur_span_idx_set = []
        for event_idx, event_arg_idxs_list in enumerate(event_arg_idxs_objs_list):
            if event_arg_idxs_list is None:
                event_idx2field_idx2pre_path2cur_span_idx_set.append(None)
            else:
                num_fields = len(event_arg_idxs_list[0])
                # field_idx -> pre_path -> {span_idx, ...}
                field_idx2pre_path2cur_span_idx_set = []
                for field_idx in range(num_fields):
                    pre_path2cur_span_idx_set = {}
                    for event_arg_idxs in event_arg_idxs_list:
                        pre_path = event_arg_idxs[:field_idx]
                        span_idx = event_arg_idxs[field_idx]
                        if pre_path not in pre_path2cur_span_idx_set:
                            pre_path2cur_span_idx_set[pre_path] = set()
                        pre_path2cur_span_idx_set[pre_path].add(span_idx)
                    field_idx2pre_path2cur_span_idx_set.append(pre_path2cur_span_idx_set)
                event_idx2field_idx2pre_path2cur_span_idx_set.append(field_idx2pre_path2cur_span_idx_set)

        return event_idx2field_idx2pre_path2cur_span_idx_set

    def is_multi_event(self):
        event_cnt = 0
        for event_objs in self.event_arg_idxs_objs_list:
            if event_objs is not None:
                event_cnt += len(event_objs)
                if event_cnt > 1:
                    return True

        return False


class EventExtractionDataPreprocess(object):

    def __init__(self, config):
        self.config = config

        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>", "[UNK]"]
        self.tokenizer = self.load_tokenizer(self.config)

        self.entity_label_list = self.get_entity_label_list()
        self.event_type_fields_pairs = self.get_event_type_fields_pairs()
        self.max_sent_len = self.config.max_sent_len
        self.max_sent_num = self.config.max_sent_num
        self.max_seq_len = self.config.max_seq_len
        self.entity_label2index = {entity_label: idx for idx, entity_label in enumerate(self.entity_label_list)}

        self.event_type2index = {}
        self.event_type_list = []
        self.event_fields_list = []
        for idx, (event_type, event_fields) in enumerate(self.event_type_fields_pairs):
            self.event_type2index[event_type] = idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)

    def load_tokenizer(self, args):
        if args.pred_model_type in ["albert", "roberta"]:
            class CNerTokenizer(BertTokenizer):
                def __init__(self, vocab_file, do_lower_case=False):
                    super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
                    self.vocab_file = str(vocab_file)
                    self.do_lower_case = do_lower_case
                    self.vocab = load_vocab(vocab_file)

                def tokenize(self, text):
                    _tokens = []
                    for c in text:
                        if self.do_lower_case:
                            c = c.lower()
                        if c in self.vocab:
                            _tokens.append(c)
                        else:
                            _tokens.append('[UNK]')
                    return _tokens
        else:
            class CNerTokenizer(MODEL_CLASSES[args.pred_model_type][1]):
                def __init__(self, vocab_file, do_lower_case=False):
                    super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
                    self.vocab_file = str(vocab_file)
                    self.do_lower_case = do_lower_case
                    self.vocab = load_vocab(vocab_file)

                def tokenize(self, text):
                    _tokens = []
                    for c in text:
                        if self.do_lower_case:
                            c = c.lower()
                        if c in self.vocab:
                            _tokens.append(c)
                        else:
                            _tokens.append('[UNK]')
                    return _tokens

        tokenizer = CNerTokenizer.from_pretrained(args.model_name_or_path)

        return tokenizer

    def get_data(self, file):

        return default_load_json(file)

    def rearrange_sent_info(self, detail_align_info):
        if 'ann_valid_dranges' not in detail_align_info:
            detail_align_info['ann_valid_dranges'] = []
        if 'ann_mspan2dranges' not in detail_align_info:
            detail_align_info['ann_mspan2dranges'] = {}

        detail_align_info = dict(detail_align_info)
        split_rgx = re.compile('[，：:；;）)]')

        raw_sents = detail_align_info['sentences']
        doc_text = ''.join(raw_sents)
        raw_dranges = detail_align_info['ann_valid_dranges']
        raw_sid2span_char_set = defaultdict(lambda: set())
        for raw_sid, char_s, char_e in raw_dranges:
            span_char_set = raw_sid2span_char_set[raw_sid]
            span_char_set.update(range(char_s, char_e))

        # try to split long sentences into short ones by comma, colon, semi-colon, bracket
        short_sents = []
        for raw_sid, sent in enumerate(raw_sents):
            span_char_set = raw_sid2span_char_set[raw_sid]
            if len(sent) > self.max_sent_len:
                cur_char_s = 0
                for mobj in split_rgx.finditer(sent):
                    m_char_s, m_char_e = mobj.span()
                    if m_char_s in span_char_set:
                        continue
                    short_sents.append(sent[cur_char_s:m_char_e])
                    cur_char_s = m_char_e
                short_sents.append(sent[cur_char_s:])
            else:
                short_sents.append(sent)

        # merge adjacent short sentences to compact ones that match max_sent_len
        comp_sents = ['']
        for sent in short_sents:
            prev_sent = comp_sents[-1]
            if len(prev_sent + sent) <= self.max_sent_len:
                comp_sents[-1] = prev_sent + sent
            else:
                comp_sents.append(sent)

        # get global sentence character base indexes
        raw_char_bases = [0]
        for sent in raw_sents:
            raw_char_bases.append(raw_char_bases[-1] + len(sent))
        comp_char_bases = [0]
        for sent in comp_sents:
            comp_char_bases.append(comp_char_bases[-1] + len(sent))

        assert raw_char_bases[-1] == comp_char_bases[-1] == len(doc_text)

        # calculate compact doc ranges
        raw_dranges.sort()
        raw_drange2comp_drange = {}
        prev_comp_sid = 0
        for raw_drange in raw_dranges:
            raw_drange = tuple(raw_drange)  # important when json dump change tuple to list
            raw_sid, raw_char_s, raw_char_e = raw_drange
            raw_char_base = raw_char_bases[raw_sid]
            doc_char_s = raw_char_base + raw_char_s
            doc_char_e = raw_char_base + raw_char_e
            assert doc_char_s >= comp_char_bases[prev_comp_sid]

            cur_comp_sid = prev_comp_sid
            for cur_comp_sid in range(prev_comp_sid, len(comp_sents)):
                if doc_char_e <= comp_char_bases[cur_comp_sid + 1]:
                    prev_comp_sid = cur_comp_sid
                    break
            comp_char_base = comp_char_bases[cur_comp_sid]
            assert comp_char_base <= doc_char_s < doc_char_e <= comp_char_bases[cur_comp_sid + 1]
            comp_char_s = doc_char_s - comp_char_base
            comp_char_e = doc_char_e - comp_char_base
            comp_drange = (cur_comp_sid, comp_char_s, comp_char_e)

            raw_drange2comp_drange[raw_drange] = comp_drange
            assert raw_sents[raw_drange[0]][raw_drange[1]:raw_drange[2]] == \
                   comp_sents[comp_drange[0]][comp_drange[1]:comp_drange[2]]

        # update detailed align info with rearranged sentences
        detail_align_info['sentences'] = comp_sents
        detail_align_info['ann_valid_dranges'] = [
            raw_drange2comp_drange[tuple(raw_drange)] for raw_drange in detail_align_info['ann_valid_dranges']
        ]
        ann_mspan2comp_dranges = {}
        for ann_mspan, mspan_raw_dranges in detail_align_info['ann_mspan2dranges'].items():
            comp_dranges = [
                raw_drange2comp_drange[tuple(raw_drange)] for raw_drange in mspan_raw_dranges
            ]
            ann_mspan2comp_dranges[ann_mspan] = comp_dranges
        detail_align_info['ann_mspan2dranges'] = ann_mspan2comp_dranges

        return detail_align_info

    def convert_to_example_func(self, data):
        examples = []
        for annguid, detail_align_info in data:
            if self.config.rearrange_sent_flag:
                detail_align_info = self.rearrange_sent_info(detail_align_info)
            examples.append(Example(annguid, detail_align_info, only_inference=False))
        return examples

    def get_ner_feature_func(self, sent_text, srange_mspan_mtype_tuples, sequence_a_segment_id=0,
                             sequence_b_segment_id=1, cls_token_segment_id=0, pad_token_segment_id=0,
                             mask_padding_with_zero=True, log_flag=False):
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id

        num_chars = len(sent_text)
        tokens = self.tokenizer.tokenize(sent_text)
        assert len(tokens) == len(sent_text)
        labels = ['O'] * len(sent_text)
        for (srange, mspan, mtype) in srange_mspan_mtype_tuples:
            if 'B-' + mtype not in self.entity_label_list:
                print(mtype)
                continue
            for i in range(srange[0], srange[1]):
                labels[i] = 'I-' + mtype
            labels[srange[0]] = 'B-' + mtype
        label_ids = [self.entity_label2index[l] for l in labels]
        basic_label_index = self.entity_label2index['O']

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # ADD [CLS] [sep]
        tokens = [cls_token] + tokens + [sep_token]
        labels = ["O"] + labels + ['O']
        label_ids = [basic_label_index] + label_ids + [basic_label_index]
        token_type_ids = [cls_token_segment_id] + token_type_ids + [sequence_a_segment_id]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_len - len(input_ids)

        if padding_length > 0:

            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            label_ids = label_ids + [basic_label_index] * padding_length

        elif padding_length < 0:
            input_ids = input_ids[:self.max_seq_len - 1] + [input_ids[-1]]
            attention_mask = attention_mask[:self.max_seq_len - 1] + [attention_mask[-1]]
            token_type_ids = token_type_ids[:self.max_seq_len - 1] + [token_type_ids[-1]]
            label_ids = label_ids[:self.max_seq_len - 1] + [label_ids[-1]]

        assert len(input_ids) == self.max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                             self.max_seq_len)
        assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), self.max_seq_len)
        assert len(token_type_ids) == self.max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), self.max_seq_len)

        if log_flag:
                logger.info("*** NER Example ***")
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s " % " ".join([str(x) for x in label_ids]))

        return tokens, input_ids, attention_mask, token_type_ids, label_ids, num_chars

    def convert_example_to_feature(self, idx, example, log_flag=False):
        annguid = example.guid
        assert isinstance(example, Example)

        # 1. prepare doc token level feature (ner)
        # size(num_sent_num, num_sent_len)
        doc_token_id_mat = []
        doc_token_mask_mat = []
        doc_token_label_mat = []
        doc_token_segment_mat = []

        for sent_idx, sent_text in enumerate(example.sentences):
            if sent_idx >= self.max_sent_num:
                break

            srange_mspan_mtype_tuples = example.sent_idx2srange_mspan_mtype_tuples.get(sent_idx, [])

            tokens, input_ids, attention_mask, token_type_ids, label_ids, num_chars = self.get_ner_feature_func(sent_text,
                                                                                                                srange_mspan_mtype_tuples,
                                                                                                                log_flag=log_flag)
            doc_token_id_mat.append(input_ids)
            doc_token_mask_mat.append(attention_mask)
            doc_token_segment_mat.append(token_type_ids)
            doc_token_label_mat.append(label_ids)

        assert len(doc_token_id_mat) == len(doc_token_mask_mat) == len(doc_token_label_mat) == len(doc_token_segment_mat) <= self.max_sent_num
        valid_sent_num = len(doc_token_id_mat)

        # 2. prepare sapn feature

        span_token_ids_list = []
        span_dranges_list = []
        mspan2span_idx = {}
        for mspan in example.ann_valid_mspans:
            if mspan in mspan2span_idx:
                continue
            char_base_s = 1
            char_max_end = self.max_seq_len - 1
            span_dranges = []
            for sent_idx, char_s, char_e in example.ann_mspan2dranges[mspan]:
                if char_base_s + char_e <= char_max_end and sent_idx < self.max_sent_num:
                    span_dranges.append((sent_idx, char_base_s+char_s, char_base_s+char_e))
            if len(span_dranges) == 0:
                continue

            span_tokens = self.tokenizer.tokenize(mspan)
            span_token_ids = tuple(self.tokenizer.convert_tokens_to_ids(span_tokens))

            mspan2span_idx[mspan] = len(span_token_ids_list)
            span_token_ids_list.append(span_token_ids)
            span_dranges_list.append(span_dranges)

        assert len(span_token_ids_list) == len(span_dranges_list) == len(mspan2span_idx)

        if len(span_token_ids_list) == 0:
            logging.warning("neglect example {}".format(idx))

            return None

        # 3. prepare doc level event feature

        event_type_labels = []
        event_arg_idxs_objs_list = []
        for event_idx, event_type in enumerate(self.event_type_list):
            event_fields = self.event_fields_list[event_idx]

            if event_type not in example.event_type2event_objs:
                event_type_labels.append(0)
                event_arg_idxs_objs_list.append(None)
            else:
                event_objs = example.event_type2event_objs[event_type]

                event_arg_idx_objs = []
                for event_obj in event_objs:
                    assert isinstance(event_obj, BaseEvent)

                    event_arg_idxs = []
                    any_valid_flag = False
                    for field in event_fields:
                        arg_span = event_obj.field2content[field]

                        if arg_span is None or arg_span not in mspan2span_idx:
                            arg_span_idx = None
                        else:
                            arg_span_idx = mspan2span_idx[arg_span]
                            any_valid_flag = True

                        event_arg_idxs.append(arg_span_idx)

                    if any_valid_flag:
                        event_arg_idx_objs.append(tuple(event_arg_idxs))

                if event_arg_idx_objs:
                    event_type_labels.append(1)
                    event_arg_idxs_objs_list.append(event_arg_idx_objs)
                else:
                    event_type_labels.append(0)
                    event_arg_idxs_objs_list.append(None)

        feature = Feature(annguid, idx, doc_token_id_mat, doc_token_mask_mat, doc_token_label_mat,
                          span_token_ids_list, span_dranges_list, event_type_labels, event_arg_idxs_objs_list,
                          valid_sent_num=valid_sent_num)

        return feature

    def convert_to_feature_func(self, examples, log_example_num=0):

        features = []
        remove_ex_cnt = 0
        for idx, example in enumerate(examples):
            if idx < log_example_num:
                feature = self.convert_example_to_feature(idx-remove_ex_cnt, example, log_flag=True)
            else:
                feature = self.convert_example_to_feature(idx-remove_ex_cnt, example, log_flag=False)

            if feature is None:
                remove_ex_cnt += 1
                continue

            features.append(feature)
        return features

    def convert_to_dataset_func(self, features):
        assert len(features) > 0 and isinstance(features[0], Feature)
        return features

    def _get_data(self, data):

        examples = self.convert_to_example_func(data)
        features = self.convert_to_feature_func(examples)
        dataset = self.convert_to_dataset_func(features)

        return examples, features, dataset

    def get_entity_label_list(self):
        entity_set = set()
        for field in common_fields:
            entity_set.add(field)
        for event_name, fields in event_type_fields_list:
            for field in fields:
                entity_set.add(field)
        entity_list = sorted(list(entity_set))
        entity_label_list = ['O']
        for entity in entity_list:
            entity_label_list.extend(["B-" + entity, "I-" + entity])
        return entity_label_list

    def get_event_type_fields_pairs(self):
        return list(event_type_fields_list)
