# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 13:15
# software: PyCharm

import logging
import random
import numpy as np
import torch
import json
import pickle


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


def prepare_doc_batch_dict(doc_fea_list):
    doc_batch_keys = ['ex_idx', 'doc_token_ids', 'doc_token_masks', 'doc_token_labels', 'valid_sent_num']
    doc_batch_dict = {}
    for key in doc_batch_keys:
        doc_batch_dict[key] = [getattr(doc_fea, key) for doc_fea in doc_fea_list]

    return doc_batch_dict

