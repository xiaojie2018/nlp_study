# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/2 11:23
# software: PyCharm

import json
import os
from argparse import Namespace
from tqdm import tqdm
# from competition_1.src import test_file_path
from bert_classification_utils import DataProcess
from trainer import Trainer
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BertClassificationPredictModelHandler(DataProcess):

    def __init__(self, params_config):
        with open(params_config, 'r', encoding='utf-8') as f:
            params_config = json.load(f)
        self.config = Namespace(**params_config)
        self.labels = [k for k in self.config.label_id.keys()]
        print(vars(self.config))
        super(BertClassificationPredictModelHandler, self).__init__(self.config)

        self.tokenizer = self.load_tokenizer(self.config)
        print("load token success!!! ")
        self.trainer = Trainer(self.config)
        self.trainer.load_model()
        print("load model success!!! ")

    def data_process(self, texts):
        test_data = self._get_data(texts, self.config.label_id, "test")
        return test_data

    def predict(self, texts):
        test_data = self.data_process(texts)
        pred_list = self.trainer.evaluate_predict(test_data)

        return [[x[0][0][1], y] for x, y in zip(texts, pred_list)]


def read_test_data(file):
    data_user = pd.read_csv(file)
    user_title = ['phone_no_m', 'city_name', 'county_name', 'idcard_cnt', 'arpu_202004']

    data = []
    data_len = []
    for i in range(data_user.shape[0]):
        d1 = data_user.iloc[i].tolist()
        d2 = []
        nn = 0
        for x, y in zip(user_title, d1):
            d2.append((x, str(y)))
            nn += len(x)
            nn += len(str(y))
        data_len.append(nn)
        data.append([d2, str(0)])

    max_len = max(data_len) + len(user_title) * 2 + 2

    return data, max_len


if __name__ == '__main__':
    # params_config = "D:\\nlp_study\\kg\\Corona_Virus_Disease_2019\\competition_1\\src\\model\\params_config.json"
    for i in range(7, 12):
        params_config = "/home/hemei/xjie/sic/model_{}/params_config.json".format(i)
        bc = BertClassificationPredictModelHandler(params_config)
        # texts = ["上半身肥胖型", "运动传导束受累", "手术后反流性胃炎", "口腔黏膜嗜酸性溃疡"]
        # result = bc.predict(texts)
        # print(result)

        # test_file_path = "/home/hemei/xjie/bert_classification/ccks_7_1_competition_data/验证集"
        # texts = read_test_data(os.path.join(test_file_path, "entity_validation.txt"))

        texts, _ = read_test_data('./o_data/test_user.csv')

        result = bc.predict(texts)

        phone_no_m = [x[0] for x in result]
        label = [x[1] for x in result]
        df = pd.DataFrame()
        df['phone_no_m'] = phone_no_m
        df['label'] = label
        df.to_csv('./output/result_test_{}.csv'.format(i), index=0)
