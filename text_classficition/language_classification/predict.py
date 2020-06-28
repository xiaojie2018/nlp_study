# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import ClassificationDataPreprocess
from argparse import Namespace
from trainer import Trainer


class LanguageModelClassificationPredict(ClassificationDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelClassificationPredict, self).__init__(self.config)
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, texts):
        data = []
        for t in texts:
            data.append((t, self.label_0))
        return data

    def predict(self, texts):
        test_data = self.process(texts)
        test_data_ = self._get_data(test_data, self.label_id, set_type='test')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for s in intent_preds_list_all:
            s1 = {}
            for k, v in s.items():
                s1[k] = round(v, 6)
            result.append(s1)

        return result


if __name__ == '__main__':
    file = '/output/model'
    texts = ['sss', 'aaa']
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)


