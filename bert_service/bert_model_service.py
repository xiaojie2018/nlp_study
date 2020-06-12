# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/4/23 18:04
# software: PyCharm

from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
import json
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os


PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'
}

# ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
# ADDITIONAL_SPECIAL_TOKENS = []


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertPreModel(BertPreTrainedModel):
    def __init__(self, args):

        self.config_class, _ = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)
        self.tokenizer = self.load_tokenizer(args)
        super(BertPreModel, self).__init__(bert_config)
        self.batch_size = args.batch_size
        self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)
        # albert = AlbertModel.from_pretrained(args.model_name_or_path, from_tf=True)
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        # self.additional_special_tokens = []

        # tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
        # albert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
        # print(1)

    def load_tokenizer(self, args):
        # tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        # tokenizer.add_special_tokens({"additional_special_tokens": self.additional_special_tokens})
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        return tokenizer

    def convert_examples_to_features(self, examples, max_seq_len=128, cls_token='[CLS]', cls_token_segment_id=0,
                                     sep_token='[SEP]', pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, add_sep_token=False, mask_padding_with_zero=True):

        features = []
        for example in examples:

            if isinstance(example, str):
                tokens_a = self.tokenizer.tokenize(example)
                if add_sep_token:
                    special_tokens_count = 2
                else:
                    special_tokens_count = 1
                if len(tokens_a) > max_seq_len - special_tokens_count:
                    tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

                tokens = tokens_a
                if add_sep_token:
                    tokens += [sep_token]

                token_type_ids = [sequence_a_segment_id] * len(tokens)

                tokens = [cls_token] + tokens
                token_type_ids = [cls_token_segment_id] + token_type_ids

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            elif isinstance(example, list) and len(example) == 2:
                example_a, example_b = example[0], example[1]
                tokens_a = self.tokenizer.tokenize(example_a)
                tokens_b = self.tokenizer.tokenize(example_b)
                tokens = tokens_a + [sep_token]
                token_type_ids = [sequence_a_segment_id] * len(tokens)

                tokens += tokens_b + [sep_token]
                token_type_ids += [sequence_b_segment_id] * len(tokens_b + [sep_token])

                tokens = [cls_token] + tokens
                token_type_ids = [cls_token_segment_id] + token_type_ids

                if len(tokens) > max_seq_len - 1:
                    tokens = tokens[:(max_seq_len - 1)]
                    token_type_ids = token_type_ids[:max_seq_len]
                    tokens += [sep_token]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            elif isinstance(example, list) and len(example) > 2:
                pass

            else:
                pass

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset

    def forward(self, params):

        texts = params.get("texts", [])
        max_seq_len = params.get("max_seq_len", 128)
        dataset = self.convert_examples_to_features(texts, max_seq_len=max_seq_len)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size)

        sequence_matrix = None
        sequence_array = None
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2]}
            input_ids = batch[0]
            attention_mask = batch[1]
            token_type_ids = batch[2]

            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            sequence_output = outputs[0]
            # pooled_output = outputs[1]  # [CLS]

            if sequence_array is None:
                sequence_array = sequence_output.detach().cpu().numpy()
            else:
                sequence_array = np.append(sequence_array, sequence_output.detach().cpu().numpy(), axis=0)

            # if step == 0:
            #     sequence_matrix = sequence_output
            # else:
            #     sequence_matrix = torch.cat([sequence_matrix, sequence_output], dim=0)

        return sequence_array
