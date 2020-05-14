# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/12 15:49
# software: PyCharm


import logging
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer
import json
import copy
import random


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertTokenizer)
}


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
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label

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

    def __init__(self, input_ids, attention_mask, token_type_ids, labels, sep_masks, present_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.sep_masks = sep_masks
        self.present_mask = present_mask

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


def get_intent_acc(intent_preds, intent_labels):
    labels = {}
    preds = {}
    preds1 = {}
    for x, y in zip(intent_preds, intent_labels):
        if y not in labels:
            labels[y] = 1
        else:
            labels[y] += 1
        if x == y:
            if x not in preds:
                preds[x] = 1
            else:
                preds[x] += 1
        if x not in preds1:
            preds1[x] = 1
        else:
            preds1[x] += 1

    res = {}
    for k, v in labels.items():
        p = preds.get(k, 0) / preds.get(k, 1)
        r = preds.get(k, 0) / v
        f1 = 2*p*r/(p+r)
        res[k] = {
            "p": p,
            "r": r,
            "f1": f1
        }
    acc = (preds == labels).mean()
    res["total_acc"] = acc
    return res


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}

    intent_result = get_intent_acc(intent_preds, intent_labels)

    results.update(intent_result)

    return results

