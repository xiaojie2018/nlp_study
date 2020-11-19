# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:56
# software: PyCharm


from utils import NerDataPreprocess, init_logger
from argparse import Namespace
from trainer import Trainer
import logging
import os
import json
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
init_logger()
logger = logging.getLogger(__name__)


class LanguageModelNerTrain(NerDataPreprocess):

    def __init__(self, config_params):
        self.config = Namespace(**config_params)
        self.config.no_cuda = False

        self.config.task_type = "ner"

        self.model_save_path = self.config.model_save_path

        self.config.model_name_or_path = self.config.pretrained_model_path

        super(LanguageModelNerTrain, self).__init__(self.config)

        self.config.model_dir = self.model_save_path

        if self.config.model_decode_fc not in ['softmax', 'crf', 'span']:
            raise Exception("model_decode_fc might be missing ... ")

    def data_preprocess(self):

        train_data, labels1 = self.get_data(self.config.train_file_url)
        test_data, labels2 = self.get_data(self.config.test_file_url)
        dev_data, labels3 = self.get_data(self.config.dev_file_url)

        if self.config.model_decode_fc in ['softmax', 'crf']:

            label_list = sorted(list(set(labels1 + labels2 + labels3)))

            labels = ['O']
            for l in label_list:
                labels.append("B-{}".format(l))
                labels.append("I-{}".format(l))

            self.labels = labels
            self.config.num_labels = len(labels)
            self.label_id = {l: ind for ind, l in enumerate(labels)}
            self.id_label = {ind: l for ind, l in enumerate(labels)}
            self.config.label_id = self.label_id
            self.config.id_label = self.id_label
            self.config.labels = self.labels

            self.train_data, self.train_examples = self._get_data(train_data, labels, self.label_id, set_type="train")
            logger.info("train data num: {} ".format(str(len(train_data))))
            self.test_data, self.test_examples = self._get_data(test_data, labels, self.label_id, set_type="test")
            logger.info("test data num: {} ".format(str(len(test_data))))
            self.dev_data, self.dev_examples = self._get_data(dev_data, labels, self.label_id, set_type="dev")
            logger.info("dev data num: {} ".format(str(len(dev_data))))

        elif self.config.model_decode_fc == 'span':
            label_list = sorted(list(set(labels1 + labels2 + labels3)))
            labels = ['O'] + label_list

            self.labels = labels
            self.config.num_labels = len(labels)
            self.label_id = {l: ind for ind, l in enumerate(labels)}
            self.id_label = {ind: l for ind, l in enumerate(labels)}
            self.config.label_id = self.label_id
            self.config.id_label = self.id_label
            self.config.labels = self.labels

            self.train_data, self.train_examples = self._get_span_data(train_data, labels, self.label_id, set_type="train")
            logger.info("train data num: {} ".format(str(len(train_data))))
            self.test_data, self.test_examples = self._get_span_data(test_data, labels, self.label_id, set_type="test")
            logger.info("test data num: {} ".format(str(len(test_data))))
            self.dev_data, self.dev_examples = self._get_span_data(dev_data, labels, self.label_id, set_type="dev")
            logger.info("dev data num: {} ".format(str(len(dev_data))))

    def fit(self):

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
        self.config.model_save_path = self.model_save_path
        self.config.model_dir = self.model_save_path

        vocab_file = os.path.join(self.config.pretrained_model_path, "vocab.txt")
        out_vocab_file = os.path.join(self.model_save_path, "vocab.txt")

        f_w = open(out_vocab_file, 'w', encoding='utf-8')
        with open(vocab_file, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                f_w.write(line)
        f_w.close()
        f_r.close()

        with codecs.open(os.path.join(self.model_save_path, 'ner_config.json'), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data,
                               train_examples=self.train_examples,
                               test_examples=self.test_examples,
                               dev_examples=self.dev_examples)
        self.trainer.train()

    def eval(self):
        model_file_path = self.model_save_path
        self.trainer.load_model(model_file_path)
        test_results = self.trainer.evaluate("test")
        return test_results


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 116,
        "hyper_param_strategy": "CUSTOMED",
        "ADDITIONAL_SPECIAL_TOKENS": [],
        "model_dir": "./output",
        "data_dir": "./data",
        "model_type": "bert",
        "task_type": "classification",
        "model_name_or_path": ["E:\\nlp_tools\\bert_models\\bert-base-chinese", "/home/hemei/xjie/bert_models/bert-base-chinese"][1],
        "seed": 42,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "max_seq_len": 128,
        "learning_rate": 5e-5,
        "num_train_epochs": 50,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "warmup_proportion": 0.1,
        "dropout_rate": 0.1,
        "logging_steps": 300,
        "save_steps": 300,
        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "is_attention": False,
        "is_lstm": False,
        "is_cnn": False,
        "train_file_url": "./ccf_data/train_add_agg_1118.json",
        "test_file_url": "./ccf_data/test.json",
        "dev_file_url": "./ccf_data/test.json",
        "job_name": "ner",
        "model_save_path": "./output/model",
        "model_decode_fc": ["softmax", "crf", "span"][1],
        "loss_type": ['lsr', 'focal', 'ce', 'bce', 'bce_with_log'][3],
        "do_adv": False,
        "adv_epsilon": 1.0,
        "adv_name": 'word_embeddings',
        "crf_learning_rate": 5e-5,
        "start_learning_rate": 0.001,
        "end_learning_rate": 0.001
    }

    model_type = ["bert", "ernie", "albert", "roberta", "bert_www", "xlnet_base", "xlnet_mid",
                  'electra_base_discriminator', 'electra_small_discriminator']

    pre_model_path = {
        "bert": "D:\\bert_model\\bert-base-chinese",
        "ernie": "D:\\bert_model\\ernie-1.0",
        "albert": "E:\\nlp_tools\\bert_models\\albert_base_v1",
        "roberta": "E:\\nlp_tools\\bert_models\\chinese_roberta_wwm_ext_pytorch",
        "bert_www": "E:\\nlp_tools\\bert_models\\chinese_wwm_pytorch",
        "xlnet_base": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_base_pytorch",
        "xlnet_mid": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_mid_pytorch",
        "electra_base_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_discriminator_pytorch",
        # "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
        "electra_small_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_discriminator_pytorch",
        # "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    }
    # lag_path = '/home/hemei/xjie/bert_models'
    # lag_path = "D:\\bert_model"
    # pre_model_path = {
    #     "bert": f"{lag_path}\\bert-base-chinese",
    #     "ernie": f"{lag_path}\\ernie-1.0",
    #     "albert": f"{lag_path}/albert_base_v1",
    #     "roberta": f"{lag_path}/chinese_roberta_wwm_ext_pytorch",
    #     "bert_www": f"{lag_path}/chinese_wwm_pytorch",
    #     "xlnet_base": f"{lag_path}/chinese_xlnet_base_pytorch",
    #     "xlnet_mid": f"{lag_path}/chinese_xlnet_mid_pytorch",
    #     "electra_base_discriminator": f"{lag_path}/chinese_electra_base_discriminator_pytorch",
    #     "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
    #     "electra_small_discriminator": f"{lag_path}/chinese_electra_small_discriminator_pytorch",
    #     "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    # }

    config_params['model_type'] = model_type[1]
    config_params['model_name_or_path'] = pre_model_path[config_params['model_type']]
    config_params['pretrained_model_path'] = pre_model_path[config_params['model_type']]
    config_params['model_save_path'] = "./output/model_{}_{}_1118_1".format(config_params['model_type'], config_params['model_decode_fc'])
    lc = LanguageModelNerTrain(config_params)
    lc.data_preprocess()
    lc.fit()
