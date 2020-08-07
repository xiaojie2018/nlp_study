# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/8/7 10:13
# software: PyCharm

import json
import os
from argparse import Namespace
import torch
from sqlnet.utils import *
from sqlnet.model.sqlbert import SQLBert, BertTokenizer


class predictModel:

    def __init__(self, file_path):
        config = json.load(open(os.path.join(file_path, 'nl2sql_config.json'), 'r', encoding='utf-8'))
        self.args = Namespace(**config)

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model_dir, do_lower_case=True)
        self.model = SQLBert.from_pretrained(self.args.bert_model_dir)
        print("Loading from %s" % self.args.restore_model_path)
        self.model.load_state_dict(torch.load(self.args.restore_model_path))
        print("Loaded model from %s" % self.args.restore_model_path)

    def predict(self, texts):
        """
        :param texts: [{"question", "", "table": {} }, {}]
        :return:
        """

        test_sql, test_table = [], {}

        for ind, t in enumerate(texts):

            test_sql.append({
                "table_id": "{}".format(ind),
                "question": t['question']
            })
            test_table["{}".format(ind)] = t['table']

        batch_size = len(test_sql)

        result = predict_predict(self.model, batch_size, test_sql, test_table, tokenizer=self.tokenizer)

        return result


if __name__ == '__main__':
    model_save_path = "/home/hemei/xjie/nl2sql/nl2sql-code-submit/output/model"
    pp = predictModel(model_save_path)
    table = {"rows": [["死侍2：我爱我家", 10637.3, 25.8, 5.0],
                      ["白蛇：缘起", 10503.8, 25.4, 7.0],
                      ["大黄蜂", 6426.6, 15.6, 6.0],
                      ["密室逃生", 5841.4, 14.2, 6.0],
                      ["“大”人物", 3322.9, 8.1, 5.0],
                      ["家和万事惊", 635.2, 1.5, 25.0],
                      ["钢铁飞龙之奥特曼崛起", 595.5, 1.4, 3.0],
                      ["海王", 500.3, 1.2, 5.0],
                      ["一条狗的回家路", 360.0, 0.9, 4.0],
                      ["掠食城市", 356.6, 0.9, 3.0]],
             # "name": "Table_4d29d0513aaa11e9b911f40f24344a08",
             "title": "表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10",
             "header": ["影片名称", "周票房（万）", "票房占比（%）", "场均人次"],
             "common": "资料来源：艺恩电影智库，光大证券研究所",
             # "id": "4d29d0513aaa11e9b911f40f24344a08",
             "types": ["text", "real", "real", "real"]}

    texts = [{"question": "二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀", "table": table},
             {"question": "你好，你知道今年第四周密室逃生，还有那部大黄蜂它们票房总的占比吗", "table": table}]

    res = pp.predict(texts)

    print(res)
    [{'agg': [0],
      'cond_conn_op': 2,
      'sel': [0],
      'conds': [[0, 2, ''], [1, 0, '']]},

     {'agg': [0],
      'cond_conn_op': 0,
      'sel': [0],
      'conds': [[0, 2, '大黄蜂'], [1, 0, '']]}]
