# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import EntityClassificationDataPreprocess
from argparse import Namespace
from trainer import Trainer
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class LanguageModelEntityClassificationPredict(EntityClassificationDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelEntityClassificationPredict, self).__init__(self.config)
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
                t1 = copy.deepcopy(t)
                t1['label'] = "neg_text"
                data.append(t1)
        return data

    def predict(self, texts):
        test_data = self.process(texts)

        test_data = self.trans(test_data, self.config.max_seq_len)

        test_data_ = self._get_data(test_data, self.label_id, self.config.types, set_type='predict')

        intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data_)

        result = []
        for s in intent_preds_list_all:
            s1 = {}
            for k, v in s.items():
                s1[k] = round(v, 6)
            result.append(s1)

        return result


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


if __name__ == '__main__':

    file = "/home/hemei/xjie/trans_event_extraction/language_classification/output_2/model_ernie"

    texts = [{"text": "兴发集团发布公告,控股股东宜昌兴发集团有限责任公司于2019年11月20日将2000万股进行质押,质押方为上海浦东发展银行股份有限公司宜昌分行,质押股数占其所持股份比例的8.50%,占公司总股本的2.15%。",
             "type": "质押",
             "entity": {"entity_type": "collateral", "start_pos": 43, "end_pos": 44, "word": "股"}},
            {"text": "兴发集团发布公告,控股股东宜昌兴发集团有限责任公司于2019年11月20日将2000万股进行质押,质押方为上海浦东发展银行股份有限公司宜昌分行,质押股数占其所持股份比例的8.50%,占公司总股本的2.15%。",
             "type": "质押",
             "entity": {"entity_type": "collateral", "start_pos": 85, "end_pos": 90, "word": "8.50%"}},
            {"text": "兴发集团发布公告,控股股东宜昌兴发集团有限责任公司于2019年11月20日将2000万股进行质押,质押方为上海浦东发展银行股份有限公司宜昌分行,质押股数占其所持股份比例的8.50%,占公司总股本的2.15%。",
             "type": "质押",
             "entity": {"entity_type": "obj-org", "start_pos": 53, "end_pos": 71, "word": "上海浦东发展银行股份有限公司宜昌分行"}
             }]
    lcp = LanguageModelEntityClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

