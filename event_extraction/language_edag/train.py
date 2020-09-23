# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 9:14
# software: PyCharm


from argparse import Namespace
from utils import EventExtractionDataPreprocess
from helper import init_logger
from trainer import Trainer
import logging
import os
import json
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
init_logger()
logger = logging.getLogger(__name__)


class LanguageModelEventExtractionTrain(EventExtractionDataPreprocess):

    def __init__(self, config_params):

        self.config = Namespace(**config_params)
        self.model_save_path = self.config.model_save_path
        self.config.model_name_or_path = self.config.pretrained_model_path
        super(LanguageModelEventExtractionTrain, self).__init__(self.config)
        self.config.model_dir = self.model_save_path

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

    def data_preprocess(self):

        train_data = self.get_data(self.config.train_file_url)
        test_data = self.get_data(self.config.test_file_url)
        dev_data = self.get_data(self.config.dev_file_url)

        self.entity_label_list = self.get_entity_label_list()
        self.event_type_fields_pairs = self.get_event_type_fields_pairs()
        self.entity_label2index = {entity_label: idx for idx, entity_label in enumerate(self.entity_label_list)}

        self.event_type2index = {}
        self.event_type_list = []
        self.event_fields_list = []
        for idx, (event_type, event_fields) in enumerate(self.event_type_fields_pairs):
            self.event_type2index[event_type] = idx
            self.event_type_list.append(event_type)
            self.event_fields_list.append(event_fields)

        self.config.entity_label_list = self.entity_label_list
        self.config.event_type_fields_pairs = self.event_type_fields_pairs
        self.config.entity_label2index = self.entity_label2index
        self.config.event_type2index = self.event_type2index
        self.config.event_type_list = self.event_type_list
        self.config.event_fields_list = self.event_fields_list

        self.train_examples, self.train_features, self.train_dataset = self._get_data(train_data)
        self.test_examples, self.test_features, self.test_dataset = self._get_data(test_data)
        self.dev_examples, self.dev_features, self.dev_dataset = self._get_data(dev_data)

    def fit(self):

        self.config.model_save_path = self.model_save_path
        self.config.model_dir = self.model_save_path

        vocab_file = os.path.join(self.config.pretrained_model_path, "vocab.txt")
        out_vocab_file = os.path.join(self.model_save_path, "vocab.txt")

        f_w = open(out_vocab_file, 'w')
        with open(vocab_file, 'r') as f_r:
            for line in f_r:
                f_w.write(line)
        f_w.close()
        f_r.close()

        with codecs.open(os.path.join(self.model_save_path, 'ner_config.json'), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_dataset,
                               dev_dataset=self.dev_dataset,
                               test_dataset=self.test_dataset,
                               train_examples=self.train_examples,
                               test_examples=self.test_examples,
                               dev_examples=self.dev_examples,
                               train_features=self.train_features,
                               test_features=self.test_features,
                               dev_features=self.dev_features)
        self.trainer.train()

    def eval(self):

        pass


if __name__ == '__main__':
    config_params = {
        "model_dir": "./output",
        "data_dir": "./data",
        "model_name_or_path": "",
        "pred_model_type": "bert",
        "model_type": ["Doc2EDAG", "DCFEE"][0],
        "task_name": "dage",
        "seed": 1234,
        "train_batch_size": 16,
        "eval_batch_size": 4,
        "max_seq_len": 128,
        "max_sent_len": 128,
        "max_sent_num": 64,
        "learning_rate": 5e-5,
        "num_train_epochs": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "local_rank": -1,
        "loss_scale": 128,
        "warmup_steps": 0,
        "warmup_proportion": 0.1,
        "dropout_rate": 0.1,
        "optimize_on_cpu": False,
        "fp16": False,
        "rearrange_sent": False,
        "rearrange_sent_flag": False,
        "use_crf_layer": True,
        "min_teacher_prob": 0.1,
        "schedule_epoch_start": 10,
        "schedule_epoch_length": 10,
        "loss_lambda": 0.05,
        "loss_gamma": 1.0,
        "add_greedy_dec": True,
        "use_token_role": True,
        "seq_reduce_type": ['MaxPooling', 'MeanPooling', 'AWA'][0],
        "hidden_size": 768,
        "dropout": 0.1,
        "ff_size": 1024,
        "num_tf_layers": 4,
        "use_path_mem": True,
        "use_scheduled_sampling": True,
        "use_doc_enc": True,
        "neg_field_loss_scaling": 3.0,

        "logging_steps": 50,
        "save_steps": 50,
        "no_cuda": False,
        "ignore_index": 0,
        "train_file_url": "./data/sample_train.json",
        "test_file_url": "./data/sample_train.json",
        "dev_file_url": "./data/sample_train.json",
    }

    model_type = ["bert", "ernie", "albert", "roberta", "bert_www", "xlnet_base", "xlnet_mid",
                  'electra_base_discriminator', 'electra_small_discriminator']

    pre_model_path = {
        "bert": "E:\\nlp_tools\\bert_models\\bert-base-chinese",
        "ernie": "E:\\nlp_tools\\ernie_models\\ERNIE",
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
    # pre_model_path = {
    #     "bert": f"{lag_path}/bert-base-chinese",
    #     "ernie": f"{lag_path}/ERNIE",
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

    config_params['pred_model_type'] = model_type[0]
    config_params['model_name_or_path'] = pre_model_path[config_params['pred_model_type']]
    config_params['pretrained_model_path'] = pre_model_path[config_params['pred_model_type']]
    config_params['model_save_path'] = "./output/model_{}".format(config_params['pred_model_type'])
    lc = LanguageModelEventExtractionTrain(config_params)
    lc.data_preprocess()
    lc.fit()



