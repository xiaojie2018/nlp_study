# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 15:20
# software: PyCharm

from tqdm import tqdm
from gensim.models import word2vec
import gensim
import numpy as np


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


if __name__ == '__main__':
    output_file_ = '../data/entity_char.txt'
    gw = GenWord2Vec()
    # gw.fit(output_file_)
    model_file = '../data/word2vec_medical'
    model = gw.load(model_file)
