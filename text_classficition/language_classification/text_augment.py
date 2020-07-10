# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/7/8 11:49
# software: PyCharm

import json
import copy
import collections
import numpy as np
import random


class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value


class UnifRep(EfficientRandomGen):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, token_prob, vocab):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, example):
        example.text = self.replace_tokens(list(example.text))
        if example.text_b:
            example.text_b = self.replace_tokens(list(example.text_b))
        return example

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()

        return ''.join(tokens)

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = [k for k in self.vocab.keys()]
        self.token_ptr = len(self.token_list) - 1
        random.shuffle(self.token_list)

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token


class TfIdfWordRep(EfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob, data_stats):
        super(TfIdfWordRep, self).__init__()
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max() - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (replace_prob / replace_prob.sum() * self.token_prob * len(all_words))
        return replace_prob

    def __call__(self, example):

        all_words = copy.deepcopy(example.word_list_a)
        if example.text_b:
            all_words += example.word_list_b

        replace_prob = self.get_replace_prob(all_words)
        example.word_list_a = self.replace_tokens(example.word_list_a, replace_prob[:len(example.word_list_a)])

        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b, replace_prob[len(example.word_list_a):])

        return example

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token


def word_level_augment(examples, vocab=[], data_stats={}, aug_ops='unif-0.5'):
    """
    :param examples:
    :param aug_ops:
    :param vocab: 词典
    :param data_stats: {"idf": {}, "tf_idf": {}}
    :return:
    """
    """Word level augmentations. Used before augmentation."""

    vocab = json.load(open('./o_data/vocab.json', 'r', encoding='utf-8'))

    if aug_ops:
        if aug_ops.startswith("unif"):
            token_prob = float(aug_ops.split("-")[1])
            op = UnifRep(token_prob, vocab)
            for i in range(len(examples)):
                examples[i] = op(examples[i])
        elif aug_ops.startswith("tf_idf"):
            token_prob = float(aug_ops.split("-")[1])
            op = TfIdfWordRep(token_prob, data_stats)
            for i in range(len(examples)):
                examples[i] = op(examples[i])
    return examples


class TextAugment(EfficientRandomGen):

    pass


