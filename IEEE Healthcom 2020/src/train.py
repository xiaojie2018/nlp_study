# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/23 21:23
# software: PyCharm


from argparse import Namespace
from utils import init_logger, DataPreprocess
import logging
from trainer import Trainer
import os
import json
import pandas as pd
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)

init_logger()


class MDDTrian(DataPreprocess):

    def __init__(self, config_params):

        self.config = Namespace(**config_params)
        self.model_save_path = self.config.model_dir

    def data_process(self):
        train_data = self.get_train_data(self.config.train_file_url)
        train_data_ = self.get_data(train_data)
        self.train_data = train_data_
        self.test_data = self.train_data
        self.dev_data = self.train_data

    def fit(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        self.config.model_save_path = self.model_save_path

        with codecs.open(os.path.join(self.model_save_path, 'mdd_config.json'), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               test_dataset=self.test_data,
                               dev_dataset=self.dev_data)
        self.trainer.train()

    def eval(self):
        self.trainer = Trainer(self.config)
        self.trainer.load_model()
        test_data = self.get_test_data(self.config.test_file_url)
        test_data_ = self.get_data(test_data)
        intent_list = self.trainer.evaluate_predict(test_data_)
        id = [i for i in range(1, 441)]
        df = pd.DataFrame()
        df['id'] = id
        df['label'] = intent_list
        df.to_csv('./data/test_001.csv', index=0)


if __name__ == '__main__':
    config_parms = {
        "max_seq_len": 128,
        "embedding_size": 500,
        "hidden_size": 500,
        "seed": 1234,

        "train_batch_size": 128,
        "eval_batch_size": 128,
        "max_steps": -1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 20,
        "learning_rate": 0.0001,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 0,
        "logging_steps": 200,
        "save_steps": 200,
        "dropout_rate": 0.1,

        "train_file_url": "E:\\bishai\\IEEE_MODMA\\Training Set",
        "test_file_url": "E:\\bishai\\IEEE_MODMA\\Validation Set\\data",
        "model_dir": "./output"
    }

    mddt = MDDTrian(config_parms)
    # mddt.data_process()
    # mddt.fit()
    mddt.eval()
    print("ok!!!")
