# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/2 9:48
# software: PyCharm
import json
from argparse import Namespace

from competition_3.src.kg_utils import KGDataProcess
from competition_3.src.trainer import Trainer


class KGBertEmbeddingPredictModelHandler(KGDataProcess):
    def __init__(self, params_config):
        with open(params_config, 'r', encoding='utf-8') as f:
            params_config = json.load(f)
        self.config = Namespace(**params_config)
        print(vars(self.config))
        super(KGBertEmbeddingPredictModelHandler, self).__init__(self.config)

        self.tokenizer = self.load_tokenizer(self.config)
        print("load token success!!! ")
        self.trainer = Trainer(self.config)
        self.trainer.load_model()
        print("load model success!!! ")

    def data_process(self, texts):
        org_data, test_data_ = self.gen_predict_data(texts)
        test_data = self._get_data(test_data_, "test")
        return org_data, test_data

    def predict(self, texts):
        result = []
        for text in texts:
            org_data, test_data = self.data_process([text])
            info_preds = self.trainer.evaluate_predict(test_data)
            if org_data[0][-1] == 1:
                res1 = sorted([(x[0], y) for x, y in zip(org_data, info_preds)], key=lambda x: x[-1], reverse=True)
                result.append([x[0] for x in res1[:10]])

            elif org_data[0][-1] == 2:
                res1 = sorted([(x[2], y) for x, y in zip(org_data, info_preds)], key=lambda x: x[-1], reverse=True)
                result.append([x[0] for x in res1[:10]])

        return {"results": result}


if __name__ == '__main__':
    params_config = "D:\\nlp_study\\kg\\Corona_Virus_Disease_2019\\competition_3\\src\\model\\params_config.json"
    kgp = KGBertEmbeddingPredictModelHandler(params_config)
    texts = [["ENTITY_2824", "binding", "?"], ["ENTITY_1339", "binding", "?"],
             ["?", "interaction", "ENTITY_2063"], ["?", "interaction", "ENTITY_1571"]]

    result = kgp.predict(texts)

    output_file_ = "./output/result.json"
    with open(output_file_, 'w', encoding='utf-8') as f:
        json.dump(result, f)

