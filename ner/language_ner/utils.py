# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm


from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from config import MODEL_CLASSES
import logging
import copy
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset
import collections


logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) string. The intent label of the example.
    """

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputSpanExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text, label=[]):
        self.guid = guid
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputSpanFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids, end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

        
def change_ner_text_len(data, fuhao=['。'], max_seq_len=125):
    """
        data: ([w1, w2, ...], [l1, l2, ...])
    """
    result = []
    rc, rl = [], []
    for c, l in zip(data[0], data[1]):
        if c in fuhao:
            if l == 'O':
                rc.append(c)
                rl.append(l)
                result.append((rc, rl))
                rc, rl = [], []
            else:
                print(l)
                rc.append(c)
                rl.append(l)
        else:
            rc.append(c)
            rl.append(l)
    
    if len(rc) > 0:
        result.append((rc, rl))

    res = []
    qs = 7
    for i in range(len(result)):
        
        if len(result[i][0]) <= max_seq_len:
            res.append(result[i])
    
        else:
            caha0 = result[i]
            caha = copy.deepcopy(caha0)
            
            while True:
                i = max_seq_len
                if len(caha[0]) <= max_seq_len:
                    break
                
                while True:
                    if len(set(caha[1][i-qs: i+qs])) == 1:
                        res.append((caha[0][:i], caha[1][:i]))
                        x = caha[0][i:]
                        y = caha[1][i:]
                        caha = copy.deepcopy((x, y))
                        break
                    
                    else:
                        i -= 1
            
            if len(caha[0]) > 0:
                
                res.append((caha[0][:i], caha[1][:i]))
    
    return res


class NerDataPreprocess:

    def __init__(self, config):

        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>", "[UNK]"]
        self.tokenizer = self.load_tokenizer(self.config)

    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta"]:
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
            class CNerTokenizer(MODEL_CLASSES[args.model_type][1]):
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

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer,
                                     cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                     sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                     sequence_a_segment_id=0, mask_padding_with_zero=True, ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            tokens = tokenizer.tokenize(example.text)
            label_ids = [label_map[x] for x in example.label]
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [label_map['O']]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [label_map['O']]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [label_map['O']] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(label_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                          segment_ids=segment_ids, label_ids=label_ids))
        return features

    def convert_examples_to_features_span(self, examples, label_list, max_seq_length, tokenizer,
                                          cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                          sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                          sequence_a_segment_id=0, mask_padding_with_zero=True, ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label2id = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            textlist = example.text
            subjects = example.label
            tokens = tokenizer.tokenize(textlist)
            start_ids = [0] * len(tokens)
            end_ids = [0] * len(tokens)
            subjects_id = []
            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label], start, end))
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            if cls_token_at_end:
                tokens += [cls_token]
                start_ids += [0]
                end_ids += [0]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                start_ids = [0] + start_ids
                end_ids = [0] + end_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                start_ids = ([0] * padding_length) + start_ids
                end_ids = ([0] * padding_length) + end_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                start_ids += ([0] * padding_length)
                end_ids += ([0] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
                logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

            features.append(InputSpanFeature(input_ids=input_ids,
                                             input_mask=input_mask,
                                             segment_ids=segment_ids,
                                             start_ids=start_ids,
                                             end_ids=end_ids,
                                             subjects=subjects_id,
                                             input_len=input_len))
        return features

    def trans_label(self, d):
        text = list(d['text'])
        labels = ["O"]*len(text)
        for e in d['entities']:
            for i in range(e['start_pos'], e['end_pos']):
                labels[i] = "I-{}".format(e['entity_type'])
            labels[e['start_pos']] = "B-{}".format(e['entity_type'])

        return text, labels

    def trans_span_label(self, d):
        text = list(d['text'])
        labels = []
        for e in d['entities']:
            if len(e['word']) == e['end_pos'] - e['start_pos']:

                labels.append([e['entity_type'], e['start_pos'], e['end_pos']-1])
            else:
                labels.append([e['entity_type'], e['start_pos'], e['end_pos']])

        return text, labels

    def _get_data(self, data, label_list, label_id, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        examples = []
        for i, d in tqdm(enumerate(data)):
            guid = "%s-%s" % (set_type, i)

            text, labels = self.trans_label(d)
            
            if len(text) < self.config.max_seq_len:
                examples.append(InputExample(guid=guid, text=text, label=labels))
            else:
                res = change_ner_text_len((text, labels), max_seq_len=self.config.max_seq_len-3)
                for r in res:
                    examples.append(InputExample(guid=guid, text=r[0], label=r[1]))

#             examples.append(InputExample(guid=guid, text=text, label=labels))

        pad_token_label_id = self.config.ignore_index

        features = self.convert_examples_to_features(examples=examples,
                                                     tokenizer=self.tokenizer,
                                                     label_list=label_list,
                                                     max_seq_length=self.config.max_seq_len,
                                                     cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                                                     pad_on_left=bool(self.config.model_type in ['xlnet']),
                                                     cls_token=self.tokenizer.cls_token,
                                                     cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                                                     sep_token=self.tokenizer.sep_token,
                                                     # pad on the left for xlnet
                                                     pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                     pad_token_segment_id=4 if self.config.model_type in ['xlnet'] else 0,)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_lens)
        return dataset, examples

    def get_data(self, file):
        data = []
        labels = set()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = eval(line)
                data.append(line)
                for e in line['entities']:
                    labels.add(e['entity_type'])
        labels = sorted(list(labels))
        return data, labels

    def get_data_1(self, data):
        """
        :param data:
        :return: [{"text": , "entities": []}, {}, ], []
        """
        res = []
        labels = set()
        for x, y in zip(data[0], data[1]):
            res.append({
                "text": x,
                "entities": y
            })

            for e in y:
                labels.add(e['entity_type'])
        labels = sorted(list(labels))
        return res, labels

    def _get_span_data(self, data, label_list, label_id, set_type="train"):
        
        def chang(text, labels, max_seq_len):
            if len(text) < max_seq_len:
                return [(text, labels)]
            label = ["O"]*len(text)
            for l in labels:
                for i in range(l[1], l[2]+1):
                    label[i] = "I-{}".format(l[0])
                label[i] = "B-{}".format(l[0])
            
            res = change_ner_text_len((text, label), max_seq_len=max_seq_len)
            res1 = []
            for r in res:
                rs1 = jiexi(r[0], r[1])
                rs2 = []
                for k11 in rs1:
                    rs2.append([k11['entity_type'], k11['start_pos'], k11['end_pos']-1])
                res1.append((r[0], rs2))
            return res1
            
        if set_type == 'train':
            random.shuffle(data)

        examples = []
        for i, d in tqdm(enumerate(data)):
            guid = "%s-%s" % (set_type, i)

            text, labels = self.trans_span_label(d)
            
            res = chang(text, labels, self.config.max_seq_len-3)
            
            for r in res:
                examples.append(InputSpanExample(guid=guid, text=r[0], label=r[1]))                                               
                                                                          
#             examples.append(InputSpanExample(guid=guid, text=text, label=labels))

        pad_token_label_id = self.config.ignore_index

        features = self.convert_examples_to_features_span(examples=examples,
                                                          tokenizer=self.tokenizer,
                                                          label_list=label_list,
                                                          max_seq_length=self.config.max_seq_len,
                                                          cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                                                          pad_on_left=bool(self.config.model_type in ['xlnet']),
                                                          cls_token=self.tokenizer.cls_token,
                                                          cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                                                          sep_token=self.tokenizer.sep_token, # pad on the left for xlnet
                                                          pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                          pad_token_segment_id=4 if self.config.model_type in ['xlnet'] else 0,)

        if set_type in ['test', 'dev', 'predict']:
            return features, examples

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                                all_input_lens)

        return dataset, examples


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def collate_span_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:, :max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens


def jiexi(words0, tag1):
    res = []
    ws = ""
    start_pos = 0
    end_pos = 0
    start_pos_1 = 0
    end_pos_1 = 0
    types = ""
    sentence = ""
    for i in range(len(tag1)):
        if tag1[i].startswith('S-'):
            ws += words0[i]
            start_pos_1 = i
            end_pos_1 = i
            start_pos = len(sentence)
            sentence += words0[i]
            end_pos = len(sentence) - 1
            types = tag1[i][2:]
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""

        if tag1[i].startswith("B-"):
            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""
            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                start_pos = len(sentence)
                sentence += words0[i]
                end_pos = len(sentence) - 1
                types = tag1[i][2:]

        elif tag1[i].startswith("I-"):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos = len(sentence) - 1
                end_pos_1 = i

            elif len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                start_pos = len(sentence)
                sentence += words0[i]
                end_pos = len(sentence) - 1
                types = tag1[i][2:]

        elif tag1[i].startswith('E-'):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos = len(sentence) - 1
                end_pos_1 = i
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                start_pos = len(sentence)
                sentence += words0[i]
                end_pos = len(sentence) - 1
                types = tag1[i][2:]
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

        elif tag1[i] == 'O':

            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            sentence += words0[i]

        if i == len(tag1) - 1 and len(ws) > 0:
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""

    res1 = []
    for s in res:
        s1 = {}
        s1['word'] = s[0]
        s1['start_pos'] = s[1]
        s1['end_pos'] = s[2] + 1
        s1['entity_type'] = s[3]
        res1.append(s1)

    return res1


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]

    for i, s_1 in enumerate(start_pred):
        if s_1 == 0:
            continue

        for j, e_1 in enumerate(end_pred[i:]):
            if s_1 == e_1:
                S.append((s_1, i, i+j))
                break

    return S


def get_extract_item(start_logits, end_logits):
    S = []
    start_pred = [x.index(max(x)) for x in start_logits]
    end_pred = [x.index(max(x)) for x in end_logits]

    for i, s_1 in enumerate(start_pred):
        if s_1 == 0:
            continue

        for j, e_1 in enumerate(end_pred[i:]):
            if s_1 == e_1:
                S.append((s_1, i, i + j))
                break

    return S
