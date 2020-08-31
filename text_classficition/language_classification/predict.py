# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import ClassificationDataPreprocess
from argparse import Namespace
from trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

        # return result
        return [[x, y] for x, y in zip(texts, intent_preds_list)]


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


if __name__ == '__main__':
    model_type = ["bert", "ernie", "albert", "roberta", "bert_www", "xlnet_base", "xlnet_mid",
                  'electra_base_discriminator', 'electra_small_discriminator']

    file = "./output/model_{}".format(model_type[6])
    # file = '/output/model'
    texts = ["上半身肥胖型", "运动传导束受累", "手术后反流性胃炎", "口腔黏膜嗜酸性溃疡"]
    lcp = LanguageModelClassificationPredict(file)
    res = lcp.predict(texts)
    print(res)

    test_file_path = "/home/hemei/xjie/bert_classification/ccks_7_1_competition_data/验证集"
    texts = read_test_data(os.path.join(test_file_path, "entity_validation.txt"))
    result = lcp.predict(texts)

    output_file = './output_data/result_{}_12.txt'.format(model_type[6])
    f = open(output_file, 'w', encoding='utf-8')
    for s in result:
        f.write('\t'.join(s) + '\n')
    f.close()

