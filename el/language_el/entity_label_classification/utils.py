# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm
import itertools

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
from mertics import metrics_report
from text_augment import word_level_augment


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

    def __init__(self, guid, text, entity, text_b=None, label=None):
        self.guid = guid
        self.text = text
        self.type = type
        self.entity = entity
        self.text_b = text_b
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

    def __init__(self, input_ids, attention_mask, token_type_ids, e1_mask, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.e1_mask = e1_mask
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


class EntityClassificationDataPreprocess:

    def __init__(self, config):

        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = self.load_tokenizer(self.config)

    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta"]:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            return tokenizer
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def convert_examples_to_features(self, examples, max_seq_len, tokenizer,
                                     cls_token='[CLS]',
                                     sep_token='[SEP]',
                                     pad_token=0,
                                     pad_token_label_id=-100,
                                     cls_token_segment_id=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                print("Writing example %d of %d" % (ex_index, len(examples)))

            # Tokenize word by word

            text = example.text[:example.entity['start_pos']] + \
                   "<e1>" + example.text[example.entity['start_pos']: example.entity['end_pos']] + "</e1>" + \
                   example.text[example.entity['end_pos']:]

            tokens = tokenizer.tokenize(text)

            e11_p = tokens.index("<e1>")
            e12_p = tokens.index("</e1>")

            tokens[e11_p] = "$"
            tokens[e12_p] = "$"

            e11_p += 1
            e12_p += 1

            # tokens = tokenizer.tokenize(example.text)
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids += [sequence_a_segment_id]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p+1):
                e1_mask[i] = 1

            # 去掉  $ $

            input_ids = input_ids[:e11_p] + input_ids[e11_p+1: e12_p] + input_ids[e12_p+1:]
            token_type_ids = token_type_ids[:e11_p] + token_type_ids[e11_p + 1: e12_p] + token_type_ids[e12_p + 1:]
            attention_mask = attention_mask[:e11_p] + attention_mask[e11_p + 1: e12_p] + attention_mask[e12_p + 1:]
            e1_mask = e1_mask[:e11_p] + e1_mask[e11_p + 1: e12_p] + e1_mask[e12_p + 1:]

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)

            if padding_length > 0:

                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                e1_mask = e1_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            elif padding_length < 0:
                input_ids = input_ids[:max_seq_len-1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_seq_len-1] + [attention_mask[-1]]
                token_type_ids = token_type_ids[:max_seq_len-1] + [token_type_ids[-1]]
                e1_mask = e1_mask[:max_seq_len - 1] + [e1_mask[-1]]

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            assert len(e1_mask) == max_seq_len, "Error with entity mask length {} vs {}".format(
                len(e1_mask), max_seq_len)

            label_id = example.label

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s " % " ".join([str(x) for x in label_id]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              e1_mask=e1_mask,
                              labels=label_id
                              ))

        return features

    def _get_data(self, data, label_id, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        leng_label = len(label_id)
        examples = []
        for i, d in enumerate(data):
            label = d['label']
            text = d['text']
            guid = "%s-%s" % (set_type, i)
            label_list = [0.0] * leng_label

            label_list[label_id[label]] = 1.0

            examples.append(InputExample(guid=guid, text=text, entity=d['entity'], label=label_list))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len, self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_e1_mask, all_labels_ids)
        return dataset

    def tongji(self, data):
        text_len = []
        label_num = {}
        for d in data:
            text_len.append(len(d['text']))
            if d['label'] not in label_num:
                label_num[d['label']] = 1
            else:
                label_num[d['label']] += 1

        print("max_len", max(text_len))
        print("min_len", min(text_len))
        print("mean_len", sum(text_len)/len(text_len))
        print(label_num)

    def trans(self, data, max_seq_len):
        res = []
        for d in data:
            entity = d['entity']
            if len(d['text']) <= max_seq_len - 7:
                res.append(d)
            else:
                max_ss = max_seq_len//3

                nn = max_ss if entity['start_pos'] - max_ss > 0 else entity['start_pos']
                text = d['text'][entity['start_pos'] - nn: entity['start_pos'] - nn + max_seq_len-7]

                entity1 = {
                    "entity_type": entity['entity_type'],
                    "start_pos": nn,
                    "end_pos": nn + len(entity['word']),
                    "word": entity['word']
                }
                res.append({
                    "text": text,
                    "type": d['type'],
                    "entity": entity1,
                    "label": d['label']
                })

        return res

    def get_data(self, file, no_labels=[]):

        data = []
        label = set()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = eval(line)

                if line['label'] in no_labels:
                    continue

                word = line['mention']['mention']
                start_pos = int(line['mention']['offset'])
                end_pos = start_pos + len(word)
                if line['mention']['kb_id'].startswith("NIL_"):
                    assert "NIL_"+line['label'] == line['mention']['kb_id'], print(line)

                data.append({
                    "text": line['text'],
                    "entity": {
                        "word": word,
                        "start_pos": start_pos,
                        "end_pos": end_pos
                    },
                    "label": line['label']
                })
                label.add(line['label'])

        return data, sorted(list(label))


def compute_metrics(intent_preds_list, out_intent_label_list):
    metrics, report = metrics_report(out_intent_label_list, intent_preds_list)

    return metrics, report

