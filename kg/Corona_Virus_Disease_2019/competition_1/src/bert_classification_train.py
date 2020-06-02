# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/12 10:57
# software: PyCharm


from competition_1.src.bert_classification_utils import init_logger, DataProcess
from competition_1.src.trainer import Trainer
from competition_1.src import train_file_path
import torch
from tqdm import tqdm
import logging
import random
from argparse import Namespace
from torch.utils.data import TensorDataset
import numpy as np
import json
import os
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
init_logger()

logger = logging.getLogger(__name__)


class BertClassificationTrainModelHandler(DataProcess):

    def __init__(self, config_params):

        config_params.update(config_params.get("hyper_param", {}))
        self.config = Namespace(**config_params)
        self.config.no_cuda = False
        self.config.model_name_or_path = "E:\\nlp_tools\\bert_models\\bert-base-chinese"
        # self.config.model_name_or_path = "/home/hemei/xjie/bert_models/bert-base-chinese"

        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        super(BertClassificationTrainModelHandler, self).__init__(self.config)
        # token
        self.tokenizer = self.load_tokenizer(self.config)
        ##########

        ##########
        self.model_save_path = self.config.model_dir

    def data_process(self):
        # 获取正负样本
        train_file = os.path.join(train_file_path, "entity_type.txt")
        data, labels = self.get_data(train_file)

        label_id = {l: ind for ind, l in enumerate(labels)}
        id_label = {v: k for k, v in label_id.items()}
        self.config.labels = labels
        self.config.label_id = label_id
        self.config.id_label = id_label
        # data = data[:100]

        random.shuffle(data)
        train_len = int(len(data) * 0.8)
        train_data_ = data[:train_len]
        test_data_ = data[train_len:]

        self.train_data = self._get_data(train_data_, label_id, "train")

        self.test_data = self._get_data(test_data_, label_id, "test")

        self.dev_data = self.test_data

    def init_workflow(self):
        pass

    def fit(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        self.config.model_save_path = self.model_save_path

        with codecs.open(os.path.join(self.model_save_path, 'params_config.json'), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data)
        self.trainer.train()

    def eval(self):
        self.trainer.load_model()
        eval_results = self.trainer.evaluate1("test")
        print(eval_results)


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 19,
        "hyper_param_strategy": "CUSTOMED",
        "hyper_param": {
            "ADDITIONAL_SPECIAL_TOKENS": [],
            "model_dir": "./model",
            "data_dir": "./data",
            "model_type": "bert",
            "seed": 1234,
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "max_seq_len": 35,
            "max_seq_len2": 256,
            "learning_rate": 5e-5,
            "num_train_epochs": 10,  # 5， 20
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "max_steps": -1,
            "warmup_steps": 0,
            "dropout_rate": 0.1,
            "logging_steps": 1000,
            "save_steps": 1000,
            "neg_num": 5,
            "no_cuda": False,
            "ignore_index": 0,
            "do_train": True,
            "do_eval": True,
            "sentence_num": 2,
            "is_heading": True,
            "ntn": False
        },
        "train_file_url": [],
        "job_name": "bert_classification_0"
    }

    bt = BertClassificationTrainModelHandler(config_params)
    bt.data_process()
    bt.fit()
    bt.eval()
