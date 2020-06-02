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
from tqdm import tqdm
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)


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

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


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
    # pr = 0.5
    results = {}
    # intent_preds1 = [1 if intent_preds[i] > pr else 0 for i in range(intent_preds.shape[0])]
    intent_result, wrong_indexs_all = get_all_metrics_two(intent_labels, intent_preds)
    results.update(intent_result)
    return results


class DataProcess:
    def __init__(self, config):
        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = self.load_tokenizer(self.config)

    def load_tokenizer(self, args):
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def _get_data(self, data, label_id, set_type="train"):
        """
        :param data: [[text, label], [第二个样本], ...]
        :param set_type:
        :return:
        """

        examples = []
        if set_type == "train":
            random.shuffle(data)
        logger.info("----- {} data num: {} ------".format(set_type, len(data)))
        for i, d in tqdm(enumerate(data)):
            text = d[0]
            label = label_id[d[1]]

            guid = "%s-%s" % (set_type, i)

            examples.append(InputExample(guid=guid, text=text, label=label))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples,
                                                     self.config.max_seq_len,
                                                     self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
        return dataset

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
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                print("Writing example %d of %d" % (ex_index, len(examples)))

            # Tokenize word by word
            text = example.text
            tokens_text = tokenizer.tokenize(text)

            tokens = [cls_token] + tokens_text + [sep_token]
            token_type_ids = [cls_token_segment_id] + [sequence_a_segment_id]*(len(tokens_text)+1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            label = example.label

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %d" % example.label)

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label
                              ))
        return features

    def get_data(self, file):
        label = []
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data.append(line.replace('\n', '').split('\t'))

        data_len = []
        for d in data:
            label.append(d[-1])
            data_len.append(len(d[0]))

        label = sorted(list(set(label)))
        print("max_len: ", max(data_len))
        print("min_len: ", min(data_len))
        print("mean_:  ", sum(data_len)/len(data_len))

        return data, label

