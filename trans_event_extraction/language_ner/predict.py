# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import NerDataPreprocess
from argparse import Namespace
from trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class LanguageModelNerPredict(NerDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'ner_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelNerPredict, self).__init__(self.config)

        self.labels = self.config.labels
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        model_file_path = config_file_name

        self.id_label = {int(k): v for k, v in self.config.id_label.items()}

        self.config.id_label = self.id_label

        self.trainer = Trainer(self.config)
        self.trainer.load_model(model_file_path)

    def process(self, texts):
        data = []
        for t in texts:
            data.append({"text": t, "entities": []})
        return data

    def predict(self, texts):
        test_data = self.process(texts)

        if self.config.model_decode_fc in ['softmax', 'crf']:
            test_data_, examples = self._get_data(test_data, self.labels, self.label_id, set_type='predict')
        elif self.config.model_decode_fc == 'span':
            test_data_, examples = self._get_span_data(test_data, self.labels, self.label_id, set_type='predict')

        res = self.trainer.evaluate_test(test_data_, examples)

        return res


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


if __name__ == '__main__':
    file = "/home/hemei/xjie/trans_event_extraction/language_ner/output/model_bert_crf"
    texts = ["上述股权已于2012年9月25日办理了股权质押登记手续,股权质押期限自股权质押登记之日起至质权人办理解除质押登记为止。",
             "和房屋大部分系2015年度重大资产重组并收购广汇有限股权时进入上市公司广"]
    lcp = LanguageModelNerPredict(file)
    res = lcp.predict(texts)
    print(res)

