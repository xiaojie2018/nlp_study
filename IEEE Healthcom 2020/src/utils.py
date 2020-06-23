# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/23 21:23
# software: PyCharm

import logging
import os
import pandas as pd
import torch
import numpy as np
import random
import json
from torch.utils.data import TensorDataset
import copy
import pickle
from mertics import metrics_report


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input, label):
        self.input = input
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_list, out_intent_label_list):
    assert len(intent_list) == len(out_intent_label_list)
    results = {}
    intent_result, report = metrics_report(out_intent_label_list, intent_list)
    results.update(intent_result)

    return results, report


class DataPreprocess:

    def get_train_data(self, file_path):

        if os.path.exists('./data/train.pkl'):
            with open('./data/train.pkl', 'rb') as f:
                result = pickle.load(f)
                random.shuffle(result)
                return result

        label_file = os.path.join(file_path, 'train_label.csv')

        d = pd.read_csv(label_file).values.tolist()
        label_name_2_id = {d1[0]: d1[1] for d1 in d}

        data = {}
        train_file_ = os.path.join(file_path, 'train_data')
        train_file_list = os.listdir(train_file_)

        for f in train_file_list:
            f1 = os.path.join(train_file_, f)
            f_list = os.listdir(f1)

            for f_name in f_list:
                f_name1 = os.path.join(f1, f_name)
                d = pd.read_csv(f_name1, header=None).values
                data[f_name] = d

        result = []
        for k, v in data.items():
            result.append((v, label_name_2_id[k]))

        f = open('./data/train.pkl', 'wb')
        pickle.dump(result, f)
        f.close()

        return result

    def get_test_data(self, file_path):
        if os.path.exists('./data/test.pkl'):
            with open('./data/test.pkl', 'rb') as f:
                result = pickle.load(f)
                return result

        data = []
        for i in range(1, 441):
            f = os.path.join(file_path, "{}.csv".format(str(i)))
            d = pd.read_csv(f, header=None).values
            data.append((d, 0))

        f = open('./data/test.pkl', 'wb')
        pickle.dump(data, f)
        f.close()

        return data

    def get_data(self, data):

        features = []
        for d in data:
            l = [0.0, 0.0]
            l[d[1]] = 1.0
            features.append(
                InputFeatures(input=d[0],
                              label=l
                              ))

        all_input_ids = torch.tensor([f.input for f in features], dtype=torch.float32)
        all_labels_ids = torch.tensor([f.label for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_labels_ids)

        return dataset


