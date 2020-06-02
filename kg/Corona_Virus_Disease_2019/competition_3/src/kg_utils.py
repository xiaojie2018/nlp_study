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
from competition_3.src import attrs_file, entities_file, link_prediction_file, relationships_file, schema_file, virus2sequence_file
from competition_3.src.utils import DataProcess
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

    def __init__(self, guid, head, relation, tail, label=None):
        self.guid = guid
        self.head = head
        self.relation = relation
        self.tail = tail
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

    def __init__(self, input_ids, attention_mask, token_type_ids, sep_masks, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.sep_masks = sep_masks
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
    pr = 0.5
    results = {}
    intent_preds1 = [str(1) if intent_preds[i] > pr else str(0) for i in range(intent_preds.shape[0])]
    intent_labels1 = [str(int(i)) for i in intent_labels.tolist()]
    intent_result, wrong_indexs_all = get_all_metrics_two(intent_labels1, intent_preds1)
    results.update(intent_result)
    return results


class KGDataProcess(DataProcess):
    def __init__(self, config):
        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = self.load_tokenizer(self.config)

        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)
        self.entity_type = {}
        for k, v in self.entities.items():
            if k == "all":
                continue
            for v1 in v:
                self.entity_type[v1] = k

    def load_tokenizer(self, args):
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def gen_data(self, file):
        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)

        self.entity_type = {}
        for k, v in self.entities.items():
            if k == "all":
                continue
            for v1 in v:
                self.entity_type[v1] = k

        import pickle
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # 假如特殊字符 <e1> </e1>
        # self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        # 假如属性

        # 假如类别
        res = []
        res_len = []
        for d in data:
            e1 = d[0]
            e2 = d[2]
            e11 = e1.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[0] + e1.split("_")[1] + self.ADDITIONAL_SPECIAL_TOKENS[1]
            e22 = e2.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[2] + e2.split("_")[1] + self.ADDITIONAL_SPECIAL_TOKENS[3]

            e11 += ", type: {}".format(self.entity_type[e1])
            e22 += ", type: {}".format(self.entity_type[e2])

            # if e1 in self.attrs:
            #     xe1 = self.attrs[e1][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])
            #
            #     e11 += ", " + ": ".join(xe10)
            # if e2 in self.attrs:
            #     xe1 = self.attrs[e2][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])

            res.append([e11.lower(), d[1], e22.lower(), d[-1]])
            res_len.append(len(e11.lower() + d[1] + e22.lower()))

        print("max_len: ", max(res_len))
        print("min_len: ", min(res_len))
        print("meas: ", sum(res_len)/len(res_len))

        return res

    def gen_predict_data(self, texts):
        # 假如特殊字符 <e1> </e1>
        # self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        all_relations = [["Drug", "effect", "Virus"], ["Protein", "interaction", "Protein"],
                         ["Virus", "produce", "Protein"], ["Protein", "binding", "Protein"]]

        all_entity_types = ["Virus", "Drug", "Protein"]

        data = []  # [[e1, r, e2, label], [], ...]
        data1 = []
        for text in texts:
            if text[0] == '?':
                r = text[1]
                e2 = text[2]
                t2 = self.entity_type[e2]
                t1 = []
                for x in all_entity_types:
                    if [x, r, t2] in all_relations:
                        t1.append(x)
                all_e1 = []
                for x in t1:
                    all_e1 += self.entities[x]
                for e1 in all_e1:
                    data.append([e1, r, e2, 0])
                    data1.append([e1, r, e2, 1])

            elif text[2] == '?':
                r = text[1]
                e1 = text[0]
                t1 = self.entity_type[e1]
                t2 = []
                for x in all_entity_types:
                    if [t1, r, x] in all_relations:
                        t2.append(x)
                all_e2 = []
                for x in t2:
                    all_e2 += self.entities[x]

                for e2 in all_e2:
                    data.append([e1, r, e2, 0])
                    data1.append([e1, r, e2, 2])

        # 假如属性

        # 假如类别
        res = []
        for d in data:
            e1 = d[0]
            e2 = d[2]
            e11 = e1.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[0] + e1.split("_")[1] + \
                  self.ADDITIONAL_SPECIAL_TOKENS[1]
            e22 = e2.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[2] + e2.split("_")[1] + \
                  self.ADDITIONAL_SPECIAL_TOKENS[3]

            e11 += ", type: {}".format(self.entity_type[e1])
            e22 += ", type: {}".format(self.entity_type[e2])

            # if e1 in self.attrs:
            #     xe1 = self.attrs[e1][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])
            #
            #     e11 += ", " + ": ".join(xe10)
            # if e2 in self.attrs:
            #     xe1 = self.attrs[e2][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])

            res.append([e11.lower(), d[1], e22.lower(), d[-1]])

        return data1, res

    def _get_data(self, data, set_type="train"):
        """
        :param data: [[e1, r, e2, 0|1], [第二个样本], ...]   [实体1， 关系， 实体2， label]
        :param set_type:
        :return:
        """

        examples = []
        if set_type == "train":
            random.shuffle(data)
        logger.info("----- {} data num: {} ------".format(set_type, len(data)))
        for i, d in tqdm(enumerate(data)):
            head_entity = d[0]
            relation = d[1]
            tail_entity = d[2]
            label = d[3]

            guid = "%s-%s" % (set_type, i)

            examples.append(InputExample(guid=guid, head=head_entity, relation=relation, tail=tail_entity, label=label))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples,
                                                     self.config.max_seq_len,
                                                     self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_sep_masks = torch.tensor([f.sep_masks for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_sep_masks, all_label_ids)
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
            head = example.head
            relation = example.relation
            tail = example.tail

            tokens_head = tokenizer.tokenize(head)
            tokens_relation = tokenizer.tokenize(relation)
            tokens_tail = tokenizer.tokenize(tail)

            tokens = [cls_token] + tokens_head + [sep_token] + tokens_relation + [sep_token] + tokens_tail + [sep_token]
            # token_type_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
            token_type_ids = [cls_token_segment_id] + \
                             [sequence_a_segment_id]*(len(tokens_head)+1) + \
                             [sequence_b_segment_id]*(len(tokens_relation)+1) + \
                             [sequence_a_segment_id]*(len(tokens_tail)+1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # sep_mask   :find the sep token
            sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

            sep_mask_ids = []
            for i, x in enumerate(input_ids):
                if x == sep_token_id:
                    sep_mask_ids.append(i)
            sep_masks = []
            for i in sep_mask_ids:
                sep_mask0 = [0] * len(input_ids)
                sep_mask0[i] = 1
                sep_masks.append(sep_mask0)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            sep_masks111 = []
            for x in sep_masks:
                x = x + ([0] * padding_length)
                sep_masks111.append(x)

            sep_masks = np.array(sep_masks111)

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
                              sep_masks=sep_masks,
                              label=label
                              ))
        return features
