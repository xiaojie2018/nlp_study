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

    def __init__(self, guid, text1, mask1, text2, label=None):
        self.guid = guid
        self.text1 = text1
        self.mask1 = mask1
        self.text2 = text2
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

    def __init__(self, input_ids1, attention_mask1, token_type_ids1, mention_masks,
                 input_ids2, attention_mask2, token_type_ids2, sep_masks2,
                 labels):
        self.input_ids1 = input_ids1
        self.attention_mask1 = attention_mask1
        self.token_type_ids1 = token_type_ids1
        self.mention_masks = mention_masks

        self.input_ids2 = input_ids2
        self.attention_mask2 = attention_mask2
        self.token_type_ids2 = token_type_ids2
        self.sep_masks2 = sep_masks2

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
        if float(0) == float(p+r):
            f1 = float(0)
        else:
            f1 = 2*p*r/(p+r)
        res[k] = {
            "p": p,
            "r": r,
            "f1": f1
        }
    acc = (preds == labels).mean()
    res["total_acc"] = acc
    return res


def get_all_metrics_two(label_list, prediction_list):
    min_unit = 1e-9
    round_bit = 9

    def get_metrics(data):
        # precision, recall, f_score, true_sum = precision_recall_fscore_support(index_y, index_pred, labels=None,
        #                                                                        pos_label=1, average=None,
        #                                                                        warn_for=(
        #                                                                        'precision', 'recall', 'f-score'),
        #                                                                        sample_weight=None)
        # return round(precision, round_bit), round(recall, round_bit), round(f1, round_bit)
        precision = data['tp'] / (data['tp'] + data['fp'] + min_unit)
        recall = data['tp'] / (data['tp'] + data['fn'] + min_unit)
        f1 = (2 * precision * recall) / (precision + recall + min_unit)
        return round(precision, round_bit), round(recall, round_bit), round(f1, round_bit)

    label_dict = {i: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for i in sorted(set(label_list))}
    wrong_indexs_all = []
    for index in range(len(label_list)):
        label = label_list[index]
        predict = prediction_list[index]
        if label == predict:
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['tp'] += 1
                else:
                    label_dict[k]['tn'] += 1
        else:
            wrong_indexs_all.append(index)
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['fn'] += 1
                elif k == predict:
                    label_dict[k]['fp'] += 1
                else:
                    label_dict[k]['tn'] += 1
    metrics = {}
    all_data = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    mean_precision, mean_recall, macro_f1 = 0, 0, 0
    for k, v in label_dict.items():
        all_data = {'tp': all_data['tp'] + v['tp'],
                    'fp': all_data['fp'] + v['fp'],
                    'tn': all_data['tn'] + v['tn'],
                    'fn': all_data['fp'] + v['fn'],
                    }
        precision, recall, f1 = get_metrics(v)
        metrics[k] = {'precision': precision,
                      'recall': recall,
                      'f1': f1}
        mean_precision += precision
        mean_recall += recall
        macro_f1 += f1

    mean_precision = round(mean_precision / len(metrics), round_bit)
    mean_recall = round(mean_recall / len(metrics), round_bit)
    macro_f1 = round(macro_f1 / len(metrics), round_bit)
    sum_precision, sum_recall, micro_f1 = get_metrics(all_data)
    metrics['mean'] = {'mean_precision': mean_precision,
                       'mean_recall': mean_recall,
                       'macro_f1': macro_f1}
    metrics['sum'] = {'sum_precision': sum_precision,
                      'sum_recall': sum_recall,
                      'micro_f1': micro_f1}
    return metrics, wrong_indexs_all


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}

    # intent_result = get_intent_acc(intent_preds, intent_labels)
    intent_result, wrong_indexs_all = get_all_metrics_two(intent_labels.tolist(), intent_preds.tolist())

    results.update(intent_result)

    return results


def get_train_sample(text, offset, mention, kb_dd):

    # mention_mask = [0]*len(text)
    # for i in range(offset, offset+len(mention)):
    #     mention_mask[i] = 1
    mention_mask = [offset, offset+len(mention)]

    data = []
    for kb_d in kb_dd:
        # data.append(text)
        # data.append(mention_mask)
        d = {}
        d["mention"] = mention
        d['type'] = kb_d['type']
        for d1 in kb_d['data']:
            d[d1['predicate']] = d1['object']

        data.append([text, mention_mask, d])
    return data


def compute_metrics_link(result):
    """
    :param result: [{ }, {}, {}, ]  {"text": , "mention": , "offset": , "text_id": , "kb_id": , "pre_kb_id": }
    :return:
    """
    tp = 0
    fp = 0
    total = 0
    for data in result:
        total += 1
        if data["kb_id"] == data["pre_kb_id"]:
            tp += 1
        else:
            fp += 1
    p = tp / (tp + fp)
    r = tp / total
    a = 2 * p * r
    b = p + r
    if b == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}
    f1 = a / b
    return {"p": p, "r": r, "f1": f1}

