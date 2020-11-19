# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm


from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from config import MODEL_CLASSES
import logging
import copy
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset
from mertics import metrics_report

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) string. The intent label of the example.
    """

    def __init__(self, guid, text, text_b=None, label=None):
        self.guid = guid
        self.text = text
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, attention_mask_a, attention_mask_b, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.attention_mask_a = attention_mask_a
        self.attention_mask_b = attention_mask_b
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


class SimilarityDataPreprocess:

    def __init__(self, config):

        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
        self.tokenizer = self.load_tokenizer(self.config)

    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta"]:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            return tokenizer
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer

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

            tokens = tokenizer.tokenize(example.text)
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            attention_mask_a = [1]*len(tokens)
            attention_mask_b = [0]*len(tokens)

            # # Add [SEP] token
            # tokens += [sep_token]
            # token_type_ids += [sequence_a_segment_id]

            # for ty in ["药物", "疾病", "检查科目", "症状", "细菌", "NoneType", "医学专科", "病毒"]:
            #     tokens_y = tokenizer.tokenize(ty)
            #     tokens += tokens_y
            #     token_type_ids += [sequence_a_segment_id]*len(tokens_y)
            #
            #     tokens += [sep_token]
            #     token_type_ids += [sequence_a_segment_id]

            # Add [CLS] token
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            attention_mask_a = [0] + attention_mask_a
            attention_mask_b = [0] + attention_mask_b

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids += [sequence_a_segment_id]
            attention_mask_a += [0]
            attention_mask_b += [0]

            # 对 text_b 进行转换 编码
            max_seq_len_b = max_seq_len - len(tokens)
            tokens_b = []
            attention_mask_bs = []
            token_type_ids_bs = []
            if len(example.text_b) == 1:
                tokenb = tokenizer.tokenize(example.text_b[0])
                # 截断
                tokenb = tokenb[: max_seq_len_b-1]
                tokens_b += tokenb
                attention_mask_bs += [1]*len(tokenb)
                # Add [SEP] token
                tokens_b += [sep_token]
                attention_mask_bs += [0]
                token_type_ids_bs += [sequence_b_segment_id]*(len(tokenb)+1)

            elif len(example.text_b) > 1:
                for tb in example.text_b[:-1]:
                    tokenb = tokenizer.tokenize(tb)
                    tokens_b += tokenb

                    # Add [SEP] token
                    tokens_b += [sep_token]

                tokenb = tokenizer.tokenize(example.text_b[-1])
                if len(tokenb) < int(max_seq_len_b/2):
                    l = max_seq_len_b - len(tokenb) - 1
                else:
                    l = int(max_seq_len_b/2)-1
                tokens_b = tokens_b[-l:]
                attention_mask_bs += [0]*(len(tokens_b))
                token_type_ids_bs += [sequence_a_segment_id] * (len(tokens_b))

                l = max_seq_len_b - len(tokens_b) - 1
                tokenb = tokenb[:l]
                tokens_b += tokenb

                # Add [SEP] token
                tokens_b += [sep_token]
                attention_mask_bs += [1] * len(tokenb)
                attention_mask_bs += [0]
                token_type_ids_bs += [sequence_b_segment_id] * (len(tokenb) + 1)

            # 拼起来
            tokens += tokens_b
            token_type_ids += token_type_ids_bs
            attention_mask_a += [0]*len(tokens_b)
            attention_mask_b += attention_mask_bs

            # 去掉长度

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)

            if padding_length > 0:

                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                attention_mask_a = attention_mask_a + ([0 if mask_padding_with_zero else 1] * padding_length)
                attention_mask_b = attention_mask_b + ([0 if mask_padding_with_zero else 1] * padding_length)

            elif padding_length < 0:
                input_ids = input_ids[:max_seq_len-1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_seq_len-1] + [attention_mask[-1]]
                token_type_ids = token_type_ids[:max_seq_len-1] + [token_type_ids[-1]]
                attention_mask_a = attention_mask_a[:max_seq_len - 1] + [attention_mask_a[-1]]
                attention_mask_b = attention_mask_b[:max_seq_len - 1] + [attention_mask_b[-1]]

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)
            assert len(attention_mask_a) == max_seq_len
            assert len(attention_mask_b) == max_seq_len

            label_id = example.label

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                # logger.info("intent_label: %s " % " ".join([str(x) for x in label_id]))
                logger.info("attention_mask_a: %s" % " ".join([str(x) for x in attention_mask_a]))
                logger.info("attention_mask_b: %s" % " ".join([str(x) for x in attention_mask_b]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              attention_mask_a=attention_mask_a,
                              attention_mask_b=attention_mask_b,
                              labels=label_id
                              ))

        return features

    def _get_data(self, data, label_id, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        leng_label = len(label_id)
        examples = []
        for i, d in enumerate(data):
            labels = d['label']
            text = d['text1']
            textb = d['text2']
            if isinstance(d['text2'], str):
                textb = [d['text2']]
            if len(textb[-1]) == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            label_list = [0.0] * leng_label
            # for label in labels:
            label_list[label_id[labels]] = 1.0
            # label_list = float(labels)

            examples.append(InputExample(guid=guid, text=text, text_b=textb, label=label_list))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len, self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.float32)
        all_attention_mask_a = torch.tensor([f.attention_mask_a for f in features], dtype=torch.long)
        all_attention_mask_b = torch.tensor([f.attention_mask_b for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_attention_mask_a, all_attention_mask_b, all_labels_ids)
        return dataset

    def get_data(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(eval(line))
        return data


def compute_metrics(intent_preds_list, out_intent_label_list):
    metrics, report = metrics_report(out_intent_label_list, intent_preds_list)

    return metrics, report
