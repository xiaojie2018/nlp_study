# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/2 11:23
# software: PyCharm

import json
import os
from argparse import Namespace
from tqdm import tqdm
from competition_1.src import test_file_path
from competition_1.src.bert_classification_utils import DataProcess
from competition_1.src.trainer import Trainer


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
        data = [[t, self.labels[0]] for t in texts]
        test_data = self._get_data(data, self.config.label_id, "test")
        return test_data

    def predict(self, texts):
        test_data = self.data_process(texts)
        pred_list = self.trainer.evaluate_predict(test_data)

        return [[x, y] for x, y in zip(texts, pred_list)]


def read_test_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


if __name__ == '__main__':
    params_config = "D:\\nlp_study\\kg\\Corona_Virus_Disease_2019\\competition_1\\src\\model\\params_config.json"
    bc = BertClassificationPredictModelHandler(params_config)
    # texts = read_test_data(os.path.join(test_file_path, "entity_validation.txt"))
    texts = ["上半身肥胖型", "运动传导束受累", "手术后反流性胃炎", "口腔黏膜嗜酸性溃疡"]
    result = bc.predict(texts)

    output_file = './output/result_bert.txt'
    f = open(output_file, 'w', encoding='utf-8')
    for s in result:
        f.write('\t'.join(s)+'\n')
    f.close()

