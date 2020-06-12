# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/12 10:57
# software: PyCharm


from utils import InputFeatures, InputExample
from utils import MODEL_CLASSES
from utils import init_logger, get_train_sample
from trainer import Trainer
import torch
from tqdm import tqdm
import logging
import random
from argparse import Namespace
from torch.utils.data import TensorDataset
import numpy as np
import json
import os
import codecs
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
init_logger()

logger = logging.getLogger(__name__)


class BertEntityLinkTrainModelHandler:

    def __init__(self, config_params):

        config_params.update(config_params.get("hyper_param", {}))
        self.config = Namespace(**config_params)
        self.config.no_cuda = False
        self.config.use_crf = False
        # self.config.model_name_or_path = "E:\\nlp_tools\\bert_models\\bert-base-chinese"
        # self.config.sentence_num
        self.config.model_name_or_path = "/home/hemei/xjie/bert_models/bert-base-chinese"

        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        # load kb_data
        kb_file = os.path.join(self.config.train_file_url[0], "kb.json")
        self.data_by_alias, self.data_by_subject_id = self.get_kb_data(kb_file)
        self.typsss = ["mention", "type", "摘要", "义项描述", "标签"]
        # 负样本比例
        self.config.sentence_num = len(self.typsss)
        self.neg_num = self.config.neg_num

        # token
        self.tokenizer = self.load_tokenizer(self.config)
        ##########

        ##########
        self.model_save_path = self.config.model_dir

    @staticmethod
    def get_kb_data(file_path):

        logger.info('加载数据库 start')
        data_by_alias = {}
        data_by_subject_id = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = line.strip()
                line = eval(line)

                if line['subject'] not in line['alias']:
                    line['alias'].append(line['subject'])

                if line["subject_id"] in data_by_subject_id:
                    raise Exception('可能有重复的subject_id', "subject_id:", line["subject_id"])
                data_by_subject_id[line["subject_id"]] = line

                tmp = set()
                for alia in line['alias']:
                    tmp.add(alia)
                    # alia=alia.replace('《', '').replace('》', '')# 别名去除书名号
                    # tmp.add(alia)
                    tmp.add(alia.lower())
                line['alias'] = list(tmp)

                for alia in line['alias']:
                    if alia not in data_by_alias:
                        data_by_alias[alia] = []
                    data_by_alias[alia].append(line)
        return data_by_alias, data_by_subject_id

    def load_tokenizer(self, args):
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def get_train_data(self, file_path):
        train_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            idnn = 0
            for line in tqdm(file):
                line = line.strip()
                line = eval(line)

                text = line["text"]

                for mention_ in line["mention_data"]:
                    kb_id = mention_['kb_id']
                    mention = mention_['mention']
                    offset = int(mention_['offset'])

                    # 构建正样本
                    if kb_id in self.data_by_subject_id:
                        kb_dd = self.data_by_subject_id[kb_id]

                        data = get_train_sample(text, offset, mention, [kb_dd])
                        for d in data:
                            d2 = {k: d[2].get(k, "") for k in self.typsss}
                            train_data.append([d[0], d[1], d2, 1])

                    # 构建负样本
                    if mention in self.data_by_alias:
                        kb_dd = self.data_by_alias[mention]
                        kb_dd1 = [k for k in kb_dd if k["subject_id"] != kb_id]
                        random.shuffle(kb_dd1)
                        kb_dd2 = kb_dd1[:self.neg_num]
                        data = get_train_sample(text, offset, mention, kb_dd2)

                        for d in data:
                            d2 = {k: d[2].get(k, "") for k in self.typsss}
                            train_data.append([d[0], d[1], d2, 0])
                idnn += 1
                # if idnn > 10:
                #     break
        return train_data

    def convert_examples_to_features(self, examples, max_seq_len1, max_seq_len2, tokenizer,
                                     cls_token='[CLS]',
                                     sep_token='[SEP]',
                                     pad_token=0,
                                     pad_token_label_id=-100,
                                     cls_token_segment_id=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in tqdm(enumerate(examples)):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                print("Writing example %d of %d" % (ex_index, len(examples)))

            # Tokenize word by word
            # tokens1_ = tokenizer.tokenize(example.text1)
            text1_a = example.text1[:example.mask1[0]]
            text1_b = example.text1[example.mask1[0]: example.mask1[1]]
            text1_c = example.text1[example.mask1[1]:]

            tokens1_a = tokenizer.tokenize(text1_a)
            tokens1_b = tokenizer.tokenize(text1_b)
            tokens1_c = tokenizer.tokenize(text1_c)

            tokens1_ = tokens1_a + tokens1_b + tokens1_c
            mention_mask1_ = [0]*len(tokens1_a) + [1]*len(tokens1_b) + [0]*len(tokens1_c)

            # Add [CLS] [SEP] token
            tokens1_ = [cls_token] + tokens1_ + [sep_token]
            mention_mask1_ = [0] + mention_mask1_ + [0]
            token_type_ids1_ = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens1_) - 1)
            input_ids1_ = tokenizer.convert_tokens_to_ids(tokens1_)
            attention_mask1_ = [1 if mask_padding_with_zero else 0] * len(input_ids1_)

            # mention_mask1_ = [0]*len(tokens1_)
            #
            # word1_ = example.text1[example.mask1[0]: example.mask1[1]]
            # word_token1_ = tokenizer.tokenize(word1_)
            # for i in range(len(tokens1_)):
            #     if word_token1_ == tokens1_[i:i+len(word_token1_)]:
            #         for j in range(i, i+len(word_token1_)):
            #             mention_mask1_[j] = 1
            #         break
            # mention_mask1_ = [0] + example.mask1 + [0]

            tokens2 = []
            for word in example.text2:
                tokens2.append(tokenizer.tokenize(word))

            tokens2_ = []
            token_type_ids2_ = []

            for i, t in enumerate(tokens2):
                tokens2_ += t
                # Add [SEP] token
                tokens2_ += [sep_token]

                if i == 1 or i == 0:
                    token_type_ids2_ += [sequence_a_segment_id] * (len(t) + 1)
                else:
                    token_type_ids2_ += [sequence_b_segment_id] * (len(t) + 1)

            # Add [CLS] token
            tokens2_ = [cls_token] + tokens2_
            token_type_ids2_ = [cls_token_segment_id] + token_type_ids2_

            input_ids2_ = tokenizer.convert_tokens_to_ids(tokens2_)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask2_ = [1 if mask_padding_with_zero else 0] * len(input_ids2_)

            # sep_mask   :find the sep token
            sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

            sep_mask_ids = []
            for i, x in enumerate(input_ids2_):
                if x == sep_token_id:
                    sep_mask_ids.append(i)
            sep_masks = []
            for i in sep_mask_ids:
                sep_mask0 = [0] * len(input_ids2_)
                sep_mask0[i] = 1
                sep_masks.append(sep_mask0)

            # Zero-pad up to the sequence length.
            padding_length2 = max_seq_len2 - len(input_ids2_)
            input_ids2_ = input_ids2_ + ([pad_token_id] * padding_length2)
            attention_mask2_ = attention_mask2_ + ([0 if mask_padding_with_zero else 1] * padding_length2)
            token_type_ids2_ = token_type_ids2_ + ([pad_token_segment_id] * padding_length2)

            sep_masks111 = []
            for x in sep_masks:
                x = x + ([0] * padding_length2)
                sep_masks111.append(x)

            sep_masks2_ = np.array(sep_masks111)

            padding_length1 = max_seq_len1 - len(input_ids1_)
            input_ids1_ = input_ids1_ + ([pad_token_id] * padding_length1)
            attention_mask1_ = attention_mask1_ + ([0 if mask_padding_with_zero else 1] * padding_length1)
            token_type_ids1_ = token_type_ids1_ + ([pad_token_segment_id] * padding_length1)
            mention_mask1_ = mention_mask1_ + ([0] * padding_length1)

            assert len(input_ids2_) == max_seq_len2, "Error with input2 length {} vs {}".format(len(input_ids2_), max_seq_len2)
            assert len(attention_mask2_) == max_seq_len2, "Error with attention2 mask length {} vs {}".format(len(attention_mask2_), max_seq_len2)
            assert len(token_type_ids2_) == max_seq_len2, "Error with token2 type length {} vs {}".format(len(token_type_ids2_), max_seq_len2)

            assert len(input_ids1_) == max_seq_len1, "Error with input1 length {} vs {}".format(len(input_ids1_), max_seq_len1)
            assert len(attention_mask1_) == max_seq_len1, "Error with attention1 mask length {} vs {}".format(len(attention_mask1_), max_seq_len1)
            assert len(token_type_ids1_) == max_seq_len1, "Error with token1 type length {} vs {}".format(len(token_type_ids1_), max_seq_len1)
            assert len(mention_mask1_) == max_seq_len1, "Error with mention1 mask length {} vs {}".format(len(mention_mask1_), max_seq_len1)

            label = int(example.label)

            # if ex_index < 5:
            #     print("example")
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % example.guid)
            #     logger.info("tokens1: %s" % " ".join([str(x) for x in tokens1_]))
            #     logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1_]))
            #     logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1_]))
            #     logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1_]))
            #     logger.info("mention_mask1: %s" % " ".join([str(x) for x in mention_mask1_]))
            #
            #     logger.info("tokens2: %s" % " ".join([str(x) for x in tokens2_]))
            #     logger.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2_]))
            #     logger.info("attention_mask2: %s" % " ".join([str(x) for x in attention_mask2_]))
            #     logger.info("token_type_ids2: %s" % " ".join([str(x) for x in token_type_ids2_]))
            #     logger.info("sep_mask_ids: %s" % " ".join([str(x) for x in sep_mask_ids]))
            #
            #     logger.info("intent_label: %d" % example.label)

            features.append(
                InputFeatures(input_ids1=input_ids1_,
                              attention_mask1=attention_mask1_,
                              token_type_ids1=token_type_ids1_,
                              mention_masks=mention_mask1_,
                              input_ids2=input_ids2_,
                              attention_mask2=attention_mask2_,
                              token_type_ids2=token_type_ids2_,
                              sep_masks2=sep_masks2_,
                              labels=label
                              ))

        return features

    def change(self, tt2, max_seq_len=128):
        pingjun = max_seq_len/len(tt2)
        l11 = [len(x) for x in tt2]
        tt3 = []
        while sum(l11) > max_seq_len:
            max_id = l11.index(max(l11))
            l00 = 0
            for i in range(len(l11)):
                if i != max_id:
                    l00 += l11[i]
                if l00 > max_seq_len-pingjun:
                    l11[max_id] = pingjun
                else:
                    l11[max_id] = max_seq_len-l00
                if sum(l11) <= max_seq_len:
                    break
        for x, y in zip(tt2, l11):
            tt3.append(x[:int(y)])
        return tt3

    def _get_data(self, data, set_type="train"):

        examples = []
        random.shuffle(data)
        logger.info("train data num: {}".format(len(data)))
        for i, d in enumerate(data):

            text1 = d[0]
            mask1 = d[1]
            text2 = []
            for k, v in d[2].items():
                text2.append(k + "：" + v)

            # 对text2 进行最大长度选取
            text2 = self.change(text2, max_seq_len=self.config.max_seq_len2-len(text2)-2)

            label = d[-1]

            guid = "%s-%s" % (set_type, i)

            examples.append(InputExample(guid=guid, text1=text1, mask1=mask1, text2=text2, label=label))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len1, self.config.max_seq_len2,
                                                     self.tokenizer, pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_mention_masks = torch.tensor([f.mention_masks for f in features], dtype=torch.long)

        all_input_ids2 = torch.tensor([f.input_ids2 for f in features], dtype=torch.long)
        all_attention_mask2 = torch.tensor([f.attention_mask2 for f in features], dtype=torch.long)
        all_token_type_ids2 = torch.tensor([f.token_type_ids2 for f in features], dtype=torch.long)
        all_sep_masks2 = torch.tensor([f.sep_masks2 for f in features], dtype=torch.long)

        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_mention_masks,
                                all_input_ids2, all_attention_mask2, all_token_type_ids2, all_sep_masks2,
                                all_labels_ids)
        return dataset

    def get_test_data(self, file_path):
        """
        :param file_path:
        :return: [{"o_data": , "kb_data": [ , , ] }, {第二个例子}, ]
        """
        test_data = []
        idnn = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = line.strip()
                line = eval(line)
                text = line["text"]
                text_id = line["text_id"]

                for mention_ in line["mention_data"]:
                    mention = mention_['mention']
                    offset = int(mention_['offset'])
                    kb_id = mention_["kb_id"]
                    o_data = {"text": text, "mention": mention, "offset": offset, "text_id": text_id, "kb_id": kb_id}
                    kb_data_ = []
                    if mention not in self.data_by_alias:
                        test_data.append({"o_data": o_data, "kb_data": kb_data_})
                        continue
                    kb_data_1 = self.data_by_alias[mention]
                    kb_data_2 = get_train_sample(text, offset, mention, kb_data_1)

                    for d, y in zip(kb_data_2, kb_data_1):
                        d2 = {k: d[2].get(k, "") for k in self.typsss}
                        kb_data_.append([d[0], d[1], d2, y["subject_id"]])

                    test_data.append({"o_data": o_data, "kb_data": kb_data_})

                idnn += 1
                # if idnn > 100:
                #     break

        return test_data

    def _get_data_test(self, data, set_type="dev"):
        examples = []
        for i, d in enumerate(data):

            text1 = d[0]
            mask1 = d[1]
            text2 = []
            for k, v in d[2].items():
                text2.append(k + "：" + v)

            # 对text2 进行最大长度选取
            text2 = self.change(text2, max_seq_len=self.config.max_seq_len2 - len(text2) - 2)

            label = 0

            guid = "%s-%s" % (set_type, i)

            examples.append(InputExample(guid=guid, text1=text1, mask1=mask1, text2=text2, label=label))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len1, self.config.max_seq_len2,
                                                     self.tokenizer, pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_mention_masks = torch.tensor([f.mention_masks for f in features], dtype=torch.long)

        all_input_ids2 = torch.tensor([f.input_ids2 for f in features], dtype=torch.long)
        all_attention_mask2 = torch.tensor([f.attention_mask2 for f in features], dtype=torch.long)
        all_token_type_ids2 = torch.tensor([f.token_type_ids2 for f in features], dtype=torch.long)
        all_sep_masks2 = torch.tensor([f.sep_masks2 for f in features], dtype=torch.long)

        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_mention_masks,
                                all_input_ids2, all_attention_mask2, all_token_type_ids2, all_sep_masks2,
                                all_labels_ids)

        return dataset

    def data_process(self):
        import pickle
        o_file = self.config.train_file_url[0]

        train_file = os.path.join(o_file, "train.json")
        dev_file = os.path.join(o_file, "dev.json")

        train_data_ = self.get_train_data(train_file)
        test_data_ = self.get_test_data(dev_file)

        train_data_file = './o_data/train.pkl'
        test_data_file = './o_data/test.pkl'
        if os.path.exists(train_data_file) and os.path.exists(test_data_file):
            with open(train_data_file, 'rb') as tf:
                self.train_data = pickle.load(tf)
            with open(test_data_file, 'rb') as tf:
                self.test_data = pickle.load(tf)
        else:
            self.train_data = self._get_data(train_data_, set_type="train")

            self.test_data = []
            for td in tqdm(test_data_):
                o_d = td['o_data']
                kb_d = td['kb_data']
                if len(kb_d) == 0:
                    self.test_data.append({"o_data": o_d, "kb_data": None, "o_kb_data": []})
                else:
                    self.test_data.append({"o_data": o_d, "kb_data": self._get_data_test(kb_d, set_type="test"), "o_kb_data": kb_d})

            # tf = open(train_data_file, 'wb')
            # pickle.dump(self.train_data, tf)
            # tf.close()
            # tf = open(test_data_file, 'wb')
            # pickle.dump(self.test_data, tf)
            # tf.close()

        self.dev_data = self.test_data

    def init_workflow(self):
        pass

    def fit(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        self.config.model_save_path = self.model_save_path

        with codecs.open(os.path.join(self.model_save_path, 'params_config.json'), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data)
        self.trainer.train()

    def eval(self):
        self.trainer.load_model()
        eval_results = self.trainer.evaluate("test")
        print(eval_results)


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 19,
        "hyper_param_strategy": "CUSTOMED",
        "hyper_param": {
            "ADDITIONAL_SPECIAL_TOKENS": [],
            "model_dir": "./model",
            "data_dir": "./data",
            "model_type": "bert",
            "seed": 1234,
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "max_seq_len1": 64,
            "max_seq_len2": 256,
            "learning_rate": 5e-5,
            "num_train_epochs": 20,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "max_steps": -1,
            "warmup_steps": 0,
            "dropout_rate": 0.1,
            "logging_steps": 200,
            "save_steps": 200,
            "neg_num": 5,
            "no_cuda": False,
            "ignore_index": 0,
            "do_train": True,
            "do_eval": True,
            "sentence_num": 2,
            "is_heading": True,
            "ntn": False
        },
        "train_file_url": ["/home/hemei/xjie/erl-2019-master/ml/ccks2020_el_data",
                           "E:\\bishai\\数据集\\CCKS 2020 面向中文短文本的实体链指任务\\ccks2020_el_data"],
        "job_name": "entity_link"
    }

    bt = BertEntityLinkTrainModelHandler(config_params)
    bt.data_process()
    bt.fit()
    bt.eval()
