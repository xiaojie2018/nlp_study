# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/12 10:57
# software: PyCharm


from .utils import InputFeatures, InputExample
from .utils import MODEL_CLASSES
from .utils import init_logger
from .trainer import Trainer
import torch
import logging
from argparse import Namespace
from torch.utils.data import TensorDataset
import numpy as np
import json
import os
import codecs
init_logger()

logger = logging.getLogger(__name__)


class BertIntentTrainModelHandler:

    def __init__(self, config_params):

        config_params.update(config_params.get("hyper_param", {}))
        self.config = Namespace(**config_params)
        self.config.task = "atis"
        self.config.no_cuda = False
        self.config.use_crf = False
        self.config.model_name_or_path = "E:\\nlp_tools\\bert_models\\bert-base-chinese"
        # self.config.sentence_num

        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        # token
        self.tokenizer = self.load_tokenizer(self.config)
        ##########

        ##########
        self.model_save_path = ""

    def load_tokenizer(self, args):
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

    def change(self, tt2, max_seq_len=128):
        pingjun = max_seq_len/len(tt2)
        l11 = [len(x[0]) for x in tt2]
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
            tt3.append([x[0][:int(y)], x[1]])
        return tt3

    def get_data(self, file):
        """
        :param file:
        :return: [[ [ [text, label],[第二个句子],[第三个句子], ...  ], [第一个样本], [第二个样本],...    ],   [第二个对话],   [第三个对话], ....]
        """
        import pickle
        f = open(file, 'rb')
        o_data = pickle.load(f)
        sentence_num = self.config.sentence_num
        data = []
        for d in o_data:
            l = d["merged_dialog"]
            d1 = []
            for i in range(sentence_num, len(l)):
                l1 = l[i-sentence_num:i]
                l11 = []
                if self.config.is_heading:
                    for a in l1:
                        l11.append([a["speaker_label"]+'：'+a["corpus"], a["label"]])
                else:
                    for a in l1:
                        l11.append([a["corpus"], a["label"]])
                d1.append(l11)
            data.append(d1)

        max_seq_len = self.config.max_seq_len
        data1 = []
        labels = []
        for d in data:
            k = []
            for d1 in d:
                k.append(self.change(d1, max_seq_len))
                labels.extend([x[-1] for x in d1])
            data1.append(k)

        return data1, list(set(labels))

    def convert_examples_to_features(self, examples, max_seq_len, tokenizer,
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
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                print("Writing example %d of %d" % (ex_index, len(examples)))

            # Tokenize word by word
            tokens1 = []
            for word in example.words:
                tokens1.append(tokenizer.tokenize(word))

            tokens = []
            token_type_ids = []

            for i, t in example(tokens1):
                tokens += t
                # Add [SEP] token
                tokens += [sep_token]

                if i % 2 == 0:
                    token_type_ids += [sequence_a_segment_id] * (len(t) + 1)
                else:
                    token_type_ids += [sequence_b_segment_id] * (len(t) + 1)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # sep_mask   :find the sep token
            sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)[0]

            sep_mask_ids = []
            for i, x in example(input_ids):
                if x == sep_token_id:
                    sep_mask_ids.append(i)
            sep_masks = []
            for i in sep_mask_ids:
                sep_mask0 = [0] * len(input_ids)
                sep_mask0[i] = 1
                sep_masks.append(sep_mask0)

            # present_mask  : find the end sentence
            present_mask = [0] * len(input_ids)

            for i in range(sep_mask_ids[-2]+1, sep_mask_ids[-1]):
                present_mask[i] = 1

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            present_mask = present_mask + ([0] * padding_length)

            sep_masks111 = []
            for x in sep_masks:
                x = x + ([0] * padding_length)
                sep_masks111.append(x)

            sep_masks = np.array(sep_masks111)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)
            assert len(present_mask) == max_seq_len, "Error with token type length {} vs {}".format(
                len(present_mask), max_seq_len)

            intent_label_id = int(example.intent_label)

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
                logger.info("present_mask: %s" % " ".join([str(x) for x in present_mask]))
                logger.info("sep_mask_ids: %s" % " ".join([str(x) for x in sep_mask_ids]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=intent_label_id,
                              sep_masks=sep_masks,
                              present_mask=present_mask
                              ))

        return features

    def _get_data(self, data, labels_id, set_type="train"):

        data1 = []
        for d in data:
            for k in d:
                data1.append(k)

        examples = []
        for i, d in enumerate(data1):

            label = d[-1][-1]

            words = [k[0] for k in d]

            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            # words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = labels_id[label]
            assert len(words) == self.config.sentence_num
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len, self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.long)
        all_sep_masks = torch.tensor([f.sep_masks for f in features], dtype=torch.long)
        all_present_mask = torch.tensor([f.present_mask for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels_ids, all_sep_masks,
                                all_present_mask)
        return dataset

    def data_process(self):
        file = self.config.train_file_url[0]

        # [[ [ [text, label],[第二个句子],[第三个句子], ...  ], [第一个样本], [第二个样本],...    ],   [第二个对话],   [第三个对话], ....]
        data, labels = self.get_data(file)
        self.labels_id = {l: i for i, l in enumerate(labels)}
        self.id_labels = {i: l for i, l in enumerate(labels)}

        import random
        random.shuffle(data)
        train_len = int(len(data) * 0.8)
        train_data_ = data[:train_len]
        test_data_ = data[train_len:]

        self.train_data = self._get_data(train_data_, self.labels_id, set_type="train")
        self.test_data = self._get_data(test_data_, self.labels_id, set_type="test")
        self.dev_data = self.test_data

    def init_workflow(self):
        pass

    def fit(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        self.config.model_save_path = self.model_save_path
        self.config.labels_id = self.labels_id
        self.config.id_labels = self.id_labels

        with codecs.open(os.path.join(self.model_save_path, 'config.json'), 'w', encoding='utf-8') as fd:
            json.dump(self.config, fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data)
        self.trainer.train()

    def eval(self):
        self.trainer.load_model()
        self.trainer.evaluate("test")


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 19,
        "hyper_param_strategy": "CUSTOMED",
        "hyper_param": {
            "ADDITIONAL_SPECIAL_TOKENS": [],
            "model_dir": "./output/atis_model",
            "data_dir": "./data",
            "model_type": "bert",
            "seed": 1234,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "max_seq_len": 128,
            "learning_rate": 5e-5,
            "num_train_epochs": 10,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "max_steps": -1,
            "warmup_steps": 0,
            "dropout_rate": 0.1,
            "logging_steps": 200,
            "save_steps": 200,
            "no_cuda": False,
            "ignore_index": 0,
            "do_train": True,
            "do_eval": True,
            "sentence_num": 3,
            "is_heading": True
        },
        "train_file_url": ["D:\\nlp_tools\\nlp-platform\\model\\slot_intent\\dev.json"],
        "job_name": "dialog_intent_classification"
    }

    bt = BertIntentTrainModelHandler(config_params)
    bt.data_process()
    bt.fit()
    bt.eval()
