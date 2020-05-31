# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 15:09
# software: PyCharm


from argparse import Namespace
from competition_1.src import train_file_path, test_file_path
from competition_1.src.utils import DataProcess, GenWord2Vec
import os
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pickle


class EntityType(DataProcess, GenWord2Vec):

    def __init__(self, args):
        self.config = Namespace(**args)
        self.train_data = self.read_train_data(self.config.train_file)
        self.test_data = self.read_test_data(self.config.test_file)
        self.gw = GenWord2Vec(self.config.word_2_vec_file)
        self.get_train_word_vec()
        # super(EntityType, self).__init__()

    def get_word2vec_text(self, output_file_):
        f = open(output_file_, 'w', encoding='utf-8')
        for t in self.train_data:
            f.write(' '.join(t[0])+'\n')
        for t in self.test_data:
            f.write(' '.join(t))
        f.close()
        
    def get_train_word_vec(self):
        self.train_word_vec = {}
        self.train_word_type = {}
        for w in tqdm(self.train_data):
            if w[0] not in self.train_word_vec:
                self.train_word_type[w[0]] = []
            self.train_word_type[w[0]].append(w[1])
            self.train_word_vec[w[0]] = self.gw.get_word_vec(w[0])

    def get_rank(self, vec):
        res = []
        for k, v in self.train_word_vec.items():
            s = 1.0 - cosine(v, vec)
            res.append((k, self.train_word_type[k], s))
        res1 = sorted(res, key=lambda x: x[-1], reverse=True)
        return res1

    def sifting(self, r):
        num_k = 1
        res = {}
        for i in range(num_k):
            if r[i][1][0] not in res:
                res[r[i][1][0]] = 1
            else:
                res[r[i][1][0]] += 1
        res1 = [(k, v) for k, v in res.items()]
        res2 = sorted(res1, key=lambda x: x[-1], reverse=True)
        return res2[0][0]

    def predict(self):
        result = []
        result1 = []
        for word in tqdm(self.test_data):
            vec = self.gw.get_word_vec(word)
            r = self.get_rank(vec)
            result1.append([word, r[:50]])
            t = self.sifting(r)
            result.append([word, t])
        return result, result1


if __name__ == '__main__':
    config_params = {
        "train_file": os.path.join(train_file_path, "entity_type.txt"),
        "test_file": os.path.join(test_file_path, "entity_validation.txt"),
        "word_2_vec_file": '../data/word2vec_medical'
    }
    et = EntityType(config_params)
    # output_file_ = '../data/entity_char.txt'
    # et.get_word2vec_text(output_file_)

    res, res1 = et.predict()
    output_file = '../output/result_top1.txt'
    f = open(output_file, 'w', encoding='utf-8')
    for s in res:
        f.write('\t'.join(s)+'\n')
    f.close()

    output_file1 = os.path.join(test_file_path, 'result_top50.pkl')
    f1 = open(output_file1, 'wb')
    pickle.dump(res1, f1)
    f1.close()
