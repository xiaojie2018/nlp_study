# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 23:52
# software: PyCharm

import json
from competition_3.src import attrs_file, entities_file, link_prediction_file, relationships_file, schema_file, virus2sequence_file


class DataProcess:
    
    def read_all_entity(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['all'] = data["Virus"] + data["Drug"] + data["Protein"]
            return data

    def read_relationships(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entitys = []
            relations = []
            for s in data["relationships"]:
                entitys.append(s[0])
                entitys.append(s[2])
                relations.append(s[1])
            entitys = sorted(list(set(entitys)))
            return data["relationships"], entitys

    def read_attrs(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            res = {k[0]: k for k in data["attrs"]}
            return res

    def read_virus2sequence(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    def read_test_data(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['relationships']

    def get_train_data(self):
        """
        :return: [[e1, r, e2, 0|1], [第二个样本], ...]   [实体1， 关系， 实体2， label]
        """
        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)

        print(1)


if __name__ == '__main__':
    dp = DataProcess()
    dp.get_train_data()
