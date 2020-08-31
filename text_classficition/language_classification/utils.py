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
from text_augment import word_level_augment


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

    def __init__(self, input_ids, attention_mask, token_type_ids, labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
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


class ClassificationDataPreprocess:

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

    # def get_data_txt(self, file):
    #     contents = []
    #     labels = []
    #     with open(file, 'r', encoding='UTF-8') as f:
    #         for line in tqdm(f):
    #             lin = line.strip()
    #             if not lin:
    #                 continue
    #             content, label = lin.split('\t')
    #             contents.append((content, label))
    #             labels.append(label)
    #
    #             if len(contents) > 50:
    #                 break
    #     return contents, labels

    @classmethod
    def _get_data_txt(cls, file):
        contents = []
        labels = []
        with open(file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.replace('\n', '').split('\t')
                content, label = lin[0], lin[1]
                contents.append((content, label))
                labels.append(label)
        return contents, labels

    @classmethod
    def _get_unsup_data_txt(cls, file):
        contents = []
        with open(file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.replace('\n', '').split('\t')
                content = lin[0]
                contents.append(content)
        return contents

    @classmethod
    def get_data_json(self, file):
        contents = []
        labels = []
        with open(file, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = eval(line)
                if not lin:
                    continue
                content, label = lin['text'], lin['label']
                contents.append((content, label))
                labels.append(label)
        return contents, labels

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

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids += [sequence_a_segment_id]

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

            elif padding_length < 0:
                input_ids = input_ids[:max_seq_len-1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_seq_len-1] + [attention_mask[-1]]
                token_type_ids = token_type_ids[:max_seq_len-1] + [token_type_ids[-1]]

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            label_id = example.label

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s " % " ".join([str(x) for x in label_id]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=label_id
                              ))

        return features

    def _get_data(self, data, label_id, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        leng_label = len(label_id)
        examples = []
        for i, d in enumerate(data):
            label = d[-1]
            text = d[0]
            guid = "%s-%s" % (set_type, i)
            label_list = [0.0] * leng_label
            label_list[label_id[label]] = 1.0

            examples.append(InputExample(guid=guid, text=text, label=label_list))

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(examples, self.config.max_seq_len, self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels_ids)
        return dataset

    def _get_uda_train_data(self, data, unsup_train_data, label_id, train_batch_size, unsup_ratio, set_type="train"):

        if set_type == 'train':
            random.shuffle(data)

        leng_label = len(label_id)
        examples = []
        for i, d in enumerate(data):
            label = d[-1]
            text = d[0]
            guid = "%s-%s" % (set_type, i)
            label_list = [0.0] * leng_label
            label_list[label_id[label]] = 1.0

            examples.append(InputExample(guid=guid, text=text, label=label_list))

        unsup_examples = []
        for i, d in enumerate(unsup_train_data):
            text = d
            guid = "%s-%s" % (set_type, i+len(examples))
            label_list = [0.0] * leng_label
            label_list[0] = 1.0

            unsup_examples.append(InputExample(guid=guid, text=text, label=label_list))

        unsup_batch_size = train_batch_size*unsup_ratio
        o_examples = []
        ind = 0
        e = []
        for i in range(len(examples)):
            e.append(examples[i])
            if (i+1) % train_batch_size == 0 and len(e) > 0:
                start_pos = ind*unsup_batch_size
                end_pos = (ind+1)*unsup_batch_size
                ind += 1
                if start_pos > len(unsup_examples) or end_pos > len(unsup_examples):
                    ind = 0
                    start_pos = ind * unsup_batch_size
                    end_pos = (ind + 1) * unsup_batch_size
                e1 = unsup_examples[start_pos: end_pos]
                e2 = word_level_augment(e1)
                e += e1
                e += e2

                o_examples.append(e)
                e = []
        if len(e) > 0:
            end_pos = len(e)*unsup_ratio
            e1 = unsup_examples[:end_pos]
            e2 = word_level_augment(e1)
            e += e1
            e += e2
            o_examples.append(e)
            e = []

        o_examples1 = []
        for o in o_examples:
            o_examples1 += o

        pad_token_label_id = self.config.ignore_index
        features = self.convert_examples_to_features(o_examples1, self.config.max_seq_len, self.tokenizer,
                                                     pad_token_label_id=pad_token_label_id)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels_ids = torch.tensor([f.labels for f in features], dtype=torch.float32)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels_ids)
        return dataset


def compute_metrics(intent_preds_list, out_intent_label_list):
    metrics, report = metrics_report(out_intent_label_list, intent_preds_list)

    return metrics, report

