# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/26 16:40
# software: PyCharm


from utils import MODEL_CLASSES, init_logger, get_train_sample, InputExample, InputFeatures
from trainer import Trainer
from argparse import Namespace
import json
import os
import logging
from tqdm import tqdm
from torch.utils.data import TensorDataset
import numpy as np
import torch
init_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logger = logging.getLogger(__name__)


class BertEntityLinkPredictModelHandler:

    def __init__(self, model_path):
        with open(os.path.join(model_path, 'config.json'), encoding="utf8") as f:
            config_params = json.load(f)
        self.config = Namespace(**config_params)
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        kb_file = os.path.join(self.config.train_file_url[0], "kb.json")
        self.data_by_alias, self.data_by_subject_id = self.get_kb_data(kb_file)
        self.typsss = ["mention", "type", "摘要", "义项描述", "标签"]
        # 负样本比例
        self.config.sentence_num = len(self.typsss)
        self.neg_num = self.config.neg_num

        # token
        self.tokenizer = self.load_tokenizer(self.config)
        self.trainer = Trainer(self.config)
        # self.trainer.load_model()

    def load_tokenizer(self, args):
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

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

    def get_test_data(self, file_path):
        """
        :param file_path:
        :return: [{"o_data": , "kb_data": [ , , ] }, {第二个例子}, ]
        """
        test_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = line.strip()
                line = eval(line)
                text = line["text"]
                text_id = line["text_id"]

                for mention_ in line["mention_data"]:
                    mention = mention_['mention']
                    offset = int(mention_['offset'])
                    o_data = {"text": text, "mention": mention, "offset": offset, "text_id": text_id}
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
        return test_data

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
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                # print("Writing example %d of %d" % (ex_index, len(examples)))
                pass

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

    def _get_data(self, data, set_type="predict"):
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

    def predict(self, output_file):
        o_file = self.config.train_file_url[0]
        test_file = os.path.join(o_file, "test.json")
        test_data = self.get_test_data(test_file)
        test_result = []
        for td in tqdm(test_data):
            o_d = td['o_data']
            kb_d = td['kb_data']

            if len(kb_d) == 0:
                # test_result.append({"o_data": o_d, "kb_id": "NIL"})
                o_d["pre_kb_id"] = "NIL"
            else:
                kb_d_1 = self._get_data(kb_d)

                link_preds = self.trainer.evaluate_predict(kb_d_1)
                # 解析 kb_id

                link_preds = link_preds.tolist()
                ind = link_preds.index(max(link_preds))

                pre_kb_id = kb_d[ind][3]

                o_d["pre_kb_id"] = pre_kb_id
                # test_result.append({"o_data": o_d, "kb_id": pre_kb_id})
                test_result.append(o_d)
                """
                o_d {"text": , "mention": , "offset": , "text_id": , "pre_kb_id": }
                
                """

        result = {}
        for tr in test_result:
            if int(tr["text_id"]) not in result:
                result[int(tr["text_id"])] = []
            result[int(tr["text_id"])].append({
                "text_id": tr["text_id"],
                "text": tr["text"],
                "mention_data": [
                    {
                        "kb_id": tr["pre_kb_id"],
                        "mention": tr["mention"],
                        "offset": tr["offset"]
                    }
                ]
            })

        result1 = {}
        for k, v in result.items():
            mention_data_s = []
            for v1 in v:
                mention_data_s.append(v1["mention_data"])
            # 对mention 排序

            mention_data_s1 = sorted(mention_data_s, key=lambda x: int(x["offset"]))

            result1[k] = {
                "text_id": v[0]["text_id"],
                "text": v[0]["text"],
                "mention_data": mention_data_s1
            }

        f = open(output_file, 'w', encoding='utf-8')
        for k, v in result1.items():
            json.dump(v, f)
            f.write('\n')
        f.close()


if __name__ == '__main__':
    model_path = './model'
    output_file = './output/res_test_predict.json'
    bel = BertEntityLinkPredictModelHandler(model_path)
    bel.predict(output_file)
