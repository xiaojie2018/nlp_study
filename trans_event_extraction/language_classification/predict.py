# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import ClassificationDataPreprocess
from argparse import Namespace
from trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class LanguageModelClassificationPredict(ClassificationDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelClassificationPredict, self).__init__(self.config)
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        self.is_muti_label = self.config.is_muti_label
        self.config.model_dir = config_file_name
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, texts):
        data = []
        if self.is_muti_label:
            for t in texts:
                data.append((t, []))
        else:
            for t in texts:
                data.append((t, self.label_0))
        return data

    def predict(self, texts):
        test_data = self.process(texts)
        test_data_ = self._get_data(test_data, self.label_id, set_type='predict')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for s in intent_preds_list_all:
            s1 = {}
            for k, v in s.items():
                s1[k] = round(v, 6)
            result.append(s1)

        return result
        # return [[x, y] for x, y in zip(texts, intent_preds_list)]


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


if __name__ == '__main__':

    file = "/home/hemei/xjie/trans_event_extraction/language_classification/output_1/model_ernie"
    texts = ["上述股权已于2012年9月25日办理了股权质押登记手续,股权质押期限自股权质押登记之日起至质权人办理解除质押登记为止。",
             "和房屋大部分系2015年度重大资产重组并收购广汇有限股权时进入上市公司广"]
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

