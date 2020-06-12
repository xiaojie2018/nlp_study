# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 15:20
# software: PyCharm

from tqdm import tqdm
from gensim.models import word2vec
import gensim
import numpy as np
import json


class DataProcess:

    def read_train_data(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data.append(line.replace('\n', '').split('\t'))
        return data

    def read_test_data(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data.append(line.replace('\n', ''))
        return data


class GenWord2Vec:

    def __init__(self, model_file):
        self.model = self.load(model_file)

    def fit(self, file):
        sentences = word2vec.Text8Corpus(file)
        word2vec = gensim.models.word2vec.Word2Vec(sentences, size=50, window=3, min_count=5, sg=1, hs=1, iter=10, 
                                                   workers=25)
        word2vec.save('../data/word2vec_medical')

    def load(self, model_file):
        model = gensim.models.Word2Vec.load(model_file)
        return model

    def get_word_vec(self, word):
        word = list(word)
        vecs = np.array([0.0]*self.model.vector_size)
        ind = 0
        for w in word:
            if w in self.model:
                vecs += self.model.wv[w]
                ind += 1
        return vecs/ind


def get_dict():
    with open("./o_data/字典.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    res = {}
    for d in data:
        if d['word'] not in res:
            res[d['word']] = d['natures']
    return res


if __name__ == '__main__':
    get_dict()
    output_file_ = '../data/entity_char.txt'
    gw = GenWord2Vec()
    # gw.fit(output_file_)
    model_file = '../data/word2vec_medical'
    model = gw.load(model_file)

# {'药物': [['用药'], ['用药', '化验项'], ['化验项', '用药'], ['用药', '用药']],
#
#  '疾病': [['诊断'], ['诊断', '化验组'],  ['化验项', '诊断', '化验组'], ['病症证物', '诊断'], ['诊断', '用药'], ['治疗行为', '功能描述']],
#
#  '症状': [['症状', '功能描述'], ['症状'], ['功能描述'], ['诊断', '功能描述', '症状'], ['数值', '病状描绘词'], ['病症证物', '症状'], ['组织']],
#
#  '检查科目': [['化验组']],
#
#  '细菌': [],
#
#  'NoneType': [],
#
#  '医学专科': [['科室']],
#
#  '病毒': []}


type_mapping2 = {'药物': ['用药'],
 '疾病': ['诊断'],
 '检查科目': ['化验组'],
 '医学专科': ['科室']}


type_mapping1 = {'药物': [['用药'], ['用药', '化验项'], ['化验项', '用药'], ['用药', '用药']],
 '疾病': [['诊断'], ['诊断', '化验组'],  ['化验项', '诊断', '化验组'], ['病症证物', '诊断'], ['诊断', '用药'], ['治疗行为', '功能描述'], ['用药', '诊断'], ['诊断', '病症证物'], ['化验组', '诊断']],
 '症状': [['症状', '功能描述'], ['症状'], ['功能描述'], ['诊断', '功能描述', '症状'], ['数值', '病状描绘词'], ['病症证物', '症状'], ['组织'], ['功能描述', '症状'], ['病状描绘词', '数值'], ['症状', '病症证物']],
 '检查科目': [['化验组']],
 '细菌': [],
 'NoneType': [],
 '医学专科': [['科室']],
 '病毒': []}


