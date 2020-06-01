# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 23:52
# software: PyCharm

import json
from competition_3.src import attrs_file, entities_file, link_prediction_file, relationships_file, schema_file, virus2sequence_file
import random
from tqdm import tqdm


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

    def gen(self, t):
        neg_relations1 = {
            "Drug": ["interaction", "produce", "binding"],
            "Virus": ["effect", "interaction", "binding"],
            "Protein": ["produce", "effect"]
        }
        pos_relations1 = {
            "Drug": ["effect"],
            "Virus": ["produce"],
            "Protein": ["interaction", "binding"]
        }

        neg_relations2 = {
            "Drug": ["effect", "interaction", "produce", "binding"],
            "Virus": ["produce", "interaction", "binding"],
            "Protein": ["effect"]
        }
        pos_relations2 = {
            "Drug": [],
            "Virus": ["effect"],
            "Protein": ["interaction", "binding", "produce"]
        }

        other = {
            "Drug": {
                "effect": ["Drug", "Protein"]
            },
            "Protein": {
                "interaction": ["Virus", "Drug"],
                "binding": ["Virus", "Drug"]
            },
            "Virus": {
                "produce": ["Drug", "Virus"]
            }
        }

        e1 = t[0]
        t1 = self.entity_type[e1]
        r = t[1]
        e2 = t[2]
        t2 = self.entity_type[e2]

        res = []
        # （1） 实体e1 e2 不变，改变一个关系
        # （2） e1 不变  每个负关系   找一个 e2
        # （3） 实体e1,r 不变，  找一个e2  （不同类型）  不好
        # （4） e2 不变  每个负关系   找一个 e1
        # （5） 实体e2,r 不变，  找一个e1  （不同类型）  不好

        # e1 不变 构建负样本
        neg_relation = neg_relations1[t1]
        random.shuffle(neg_relation)
        res.append(
            [e1, neg_relation[0], e2]
        )
        for nr in neg_relation:
            i = random.randint(0, len(self.entity_neg_e2[e1])-1)
            res.append(
                [e1, nr, self.entity_neg_e2[e1][i]]
            )
        ene = []
        for kr in other[t1][r]:
            ene += self.entities[kr]

        for _ in range(2):
            i = random.randint(0, len(ene)-1)
            res.append(
                [e1, r, ene[i]]
            )

        return res

    def get_train_data(self):
        """
        :return: [[e1, r, e2, 0|1], [第二个样本], ...]   [实体1， 关系， 实体2， label]
        """
        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)

        self.entity_type = {}
        for k, v in self.entities.items():
            if k == "all":
                continue
            for v1 in v:
                self.entity_type[v1] = k

        self.entity_pos_e2 = {}
        for d in self.triple:
            if d[0] not in self.entity_pos_e2:
                self.entity_pos_e2[d[0]] = []
            self.entity_pos_e2[d[0]].append(d[-1])
        entities_all = []
        for v in self.entities.values():
            entities_all += v
        entities_all = set(entities_all)

        self.entity_neg_e2 = {}
        for k, v in self.entity_pos_e2.items():
            self.entity_neg_e2[k] = list(entities_all.difference(set(v)))

        pos_triple_data = []
        neg_triple_data = []
        for t in tqdm(self.triple):
            pos_triple_data.append(t)
            neg_triple_data += self.gen(t)

        random.shuffle(pos_triple_data)
        random.shuffle(neg_triple_data)
        train_len1 = int(len(pos_triple_data) * 0.8)
        train_len2 = int(len(neg_triple_data) * 0.8)

        train_data = []
        for p in pos_triple_data[:train_len1]:
            p1 = p + [1]
            train_data.append(p1)
        for p in neg_triple_data[:train_len2]:
            p1 = p + [0]
            train_data.append(p1)

        test_data = []
        for p in pos_triple_data[train_len1:]:
            p1 = p + [1]
            test_data.append(p1)
        for p in neg_triple_data[train_len2:]:
            p1 = p + [0]
            test_data.append(p1)

        random.shuffle(train_data)
        random.shuffle(test_data)

        import pickle
        file1 = './data/train_0.pkl'
        file2 = './data/test_0.pkl'
        df = open(file1, 'wb')
        pickle.dump(train_data, df)
        df.close()
        df1 = open(file2, 'wb')
        pickle.dump(test_data, df1)
        df1.close()

    def gen_data(self, file):
        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)

        self.entity_type = {}
        for k, v in self.entities.items():
            if k == "all":
                continue
            for v1 in v:
                self.entity_type[v1] = k

        import pickle
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # 假如特殊字符 <e1> </e1>
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        # 假如属性

        # 假如类别
        res = []
        res_len = []
        for d in data:
            e1 = d[0]
            e2 = d[2]
            e11 = e1.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[0] + e1.split("_")[1] + \
                  self.ADDITIONAL_SPECIAL_TOKENS[1]
            e22 = e2.split('_')[0] + "_" + self.ADDITIONAL_SPECIAL_TOKENS[2] + e2.split("_")[1] + \
                  self.ADDITIONAL_SPECIAL_TOKENS[3]

            e11 += ", type: {}".format(self.entity_type[e1])
            e22 += ", type: {}".format(self.entity_type[e2])

            # if e1 in self.attrs:
            #     xe1 = self.attrs[e1][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])
            #
            #     e11 += ", " + ": ".join(xe10)
            # if e2 in self.attrs:
            #     xe1 = self.attrs[e2][1:]
            #     xe10 = []
            #     xe10.append(xe1[0])
            #     if isinstance(xe1[1], list):
            #         xe10.append(','.join(xe1[1]))
            #     else:
            #         xe10.append(xe1[1])
            #
            #     e22 += ", " + ": ".join(xe10)

            res.append([e11.lower(), d[1], e22.lower(), d[-1]])
            res_len.append(len(e11.lower() + d[1] + e22.lower()))

        print("max_len: ", max(res_len))
        print("min_len: ", min(res_len))
        print("meas: ", sum(res_len) / len(res_len))

        return res


if __name__ == '__main__':
    dp = DataProcess()
    # dp.get_train_data()
    dp.gen_data("./data/train_0.pkl")
