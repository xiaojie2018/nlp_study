# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/4/24 9:42
# software: PyCharm

from bert_model_service import BertPreModel
import torch
# import argparse
# import os
import json
from argparse import Namespace
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class predictModel:
    def __init__(self, model_path, hyper_url='', vocab_url=''):
        with open(hyper_url, encoding="utf8") as f:
            config = json.load(f)
        config["model_name_or_path"] = model_path
        self.args = Namespace(**config)
        self.model = BertPreModel(self.args)

        self.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        self.model.to(self.device)

    def predict(self, params):
        # self.model.train()
        self.model.eval()
        res = self.model(params)
        # res = res.detach().cpu().numpy()
        res = res.astype(np.float16)
        return res


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    # parser.add_argument("--data_dir", default="./data", type=str,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    # parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    # parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    # parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    # parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")
    #
    # parser.add_argument("--model_type", default="bert", type=str, help="Model type model_dirselected in the list: ")
    #
    # parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    # parser.add_argument("--max_seq_len", default=70, type=int,
    #                     help="The maximum total input sequence length after tokenization.")
    # parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument("--num_train_epochs", default=5.0, type=float,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--max_steps", default=-1, type=int,
    #                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    # parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    #
    # parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")
    #
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    # args = parser.parse_args()

    # args.do_train = True
    # args.do_eval = True
    # args.no_cuda = False
    #
    # args.model_name_or_path = 'E:\\nlp_tools\\bert_models\\bert-base-chinese'
    #
    # args.model_type = "albert"
    # args.model_name_or_path = "E:\\nlp_tools\\bert_models\\albert_base_v1"
    # # args.model_name_or_path = "albert-xxlarge-v1"
    #
    # args.data_dir = "./cls_data"
    # args.batch_size = 2

    args = "D:\\nlp_study\\bert_service\\model_config.json"
    texts = ["有一首歌<e2><e1>摇啊摇啊摇摇到你身边</e1></e2>是什么歌", "吴坚强。的<e2><e1>个人空间</e1></e2>",
             "晴耕雨读,全情教学。如诗亦如歌:<e2><e1>蔡润光</e1></e2>老师_"]
    # texts = texts*20
    params = {
        "texts": texts,
        "max_seq_len": 128
    }
    pre_bert = predictModel(model_path="E:\\nlp_tools\\bert_models\\albert_base_v1", hyper_url=args)
    res = pre_bert.predict(params)
    print(res)
