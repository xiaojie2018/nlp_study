# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:56
# software: PyCharm


from utils import ClassificationDataPreprocess, init_logger
from argparse import Namespace
from trainer import Trainer
import logging
import os
import json
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
init_logger()
logger = logging.getLogger(__name__)


class LanguageModelClassificationTrain(ClassificationDataPreprocess):

    def __init__(self, config_params):
        self.config = Namespace(**config_params)
        self.config.no_cuda = False
        self.model_save_path = self.config.model_save_path

        super(LanguageModelClassificationTrain, self).__init__(self.config)

    def data_preprocess(self):
        if self.config.train_file_url.endswith('txt'):
            train_data, labels1 = self._get_data_txt(self.config.train_file_url)
            # train_data1, labels11 = self._get_data_txt('./output_data/result_ernie_2.txt')
            train_data1 = []
            train_data = train_data + train_data1
            test_data, labels2 = train_data, labels1
            dev_data, labels3 = train_data, labels1
        # train_data, labels1 = self.get_data_json(self.config.train_file_url)
        # test_data, labels2 = self.get_data_json(self.config.test_file_url)
        # dev_data, labels3 = self.get_data_json(self.config.dev_file_url)

        labels = sorted(list(set(labels1 + labels2 + labels3)))

        self.labels = labels
        self.config.num_classes = len(labels)

        self.label_id = {l: ind for ind, l in enumerate(labels)}
        self.id_label = {ind: l for ind, l in enumerate(labels)}
        self.config.label_id = self.label_id
        self.config.id_label = self.id_label
        self.config.labels = self.labels

        if self.config.is_uda_model:
            unsup_train_data = self._get_unsup_data_txt(self.config.unsup_file_url)

            self.train_data = self._get_uda_train_data(train_data, unsup_train_data, self.label_id,
                                                       self.config.train_batch_size, self.config.unsup_ratio,
                                                       set_type="train")

        else:
            self.train_data = self._get_data(train_data, self.label_id, set_type="train")

        logger.info("train data num: {} ".format(str(len(train_data))))
        self.test_data = self._get_data(test_data, self.label_id, set_type="test")
        logger.info("test data num: {} ".format(str(len(test_data))))
        self.dev_data = self._get_data(dev_data, self.label_id, set_type="dev")
        logger.info("dev data num: {} ".format(str(len(dev_data))))

    def fit(self):

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
        self.config.model_save_path = self.model_save_path
        self.config.model_dir = self.model_save_path

        with codecs.open(os.path.join(self.model_save_path, '{}_config.json'.format(self.config.task_type)), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data)
        self.trainer.train()

    def eval(self):
        self.trainer.load_model()
        test_results = self.trainer.evaluate("test")
        return test_results


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 19,
        "hyper_param_strategy": "CUSTOMED",
        "ADDITIONAL_SPECIAL_TOKENS": [],
        "model_dir": "./output",
        "data_dir": "./data",
        "model_type": "bert",
        "task_type": "classification",
        "model_name_or_path": "E:\\nlp_tools\\bert_models\\bert-base-chinese",
        "seed": 1234,
        "train_batch_size": 16,
        "eval_batch_size": 64,
        "max_seq_len": 34,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "dropout_rate": 0.1,
        "logging_steps": 1000,
        "save_steps": 1000,
        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "is_attention": False,
        "is_lstm": False,
        "is_cnn": False,
        "train_file_url": "./o_data/ronghe_train_3.txt",
        "test_file_url": "./o_data/entity_type.txt",
        "dev_file_url": "./o_data/entity_type.txt",
        "unsup_file_url": "./o_data/entity_validation.txt",
        "job_name": "dialog_intent_classification",
        "model_save_path": "./output/model",
        "is_uda_model": True,
        "unsup_ratio": 3,
        "uda_coeff": 1,
        # "tsa": "linear_schedule",
        "tsa": False,
        "uda_softmax_temp": 0.85,
        "uda_confidence_thresh": 0.45,
    }

    model_type = ["bert", "ernie", "albert", "roberta", "bert_www", "xlnet_base", "xlnet_mid",
                  'electra_base_discriminator', 'electra_small_discriminator']

    # pre_model_path = {
    #     "bert": "E:\\nlp_tools\\bert_models\\bert-base-chinese",
    #     "ernie": "E:\\nlp_tools\\ernie_models\\ERNIE",
    #     "albert": "E:\\nlp_tools\\bert_models\\albert_base_v1",
    #     "roberta": "E:\\nlp_tools\\bert_models\\chinese_roberta_wwm_ext_pytorch",
    #     "bert_www": "E:\\nlp_tools\\bert_models\\chinese_wwm_pytorch",
    #     "xlnet_base": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_base_pytorch",
    #     "xlnet_mid": "E:\\nlp_tools\\xlnet_models\\chinese_xlnet_mid_pytorch",
    #     "electra_base_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_discriminator_pytorch",
    #     # "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
    #     "electra_small_discriminator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_discriminator_pytorch",
    #     # "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    # }
    lag_path = '/home/hemei/xjie/bert_models'
    pre_model_path = {
        "bert": f"{lag_path}/bert-base-chinese",  # jindong  bert-base-chinese
        "ernie": f"{lag_path}/ERNIE_stable-1.0.1-pytorch",  # ERNIE_stable-1.0.1-pytorch   ERNIE  ERNIE_1.0_max-len-512-pytorch
        "albert": f"{lag_path}/albert_base_v1",
        "roberta": f"{lag_path}/chinese_roberta_wwm_ext_pytorch",
        "bert_www": f"{lag_path}/chinese_wwm_pytorch",
        "xlnet_base": f"{lag_path}/chinese_xlnet_base_pytorch",
        "xlnet_mid": f"{lag_path}/chinese_xlnet_mid_pytorch",
        "electra_base_discriminator": f"{lag_path}/chinese_electra_base_discriminator_pytorch",
        # "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
        "electra_small_discriminator": f"{lag_path}/chinese_electra_small_discriminator_pytorch",
        # "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    }

    config_params['model_type'] = model_type[0]
    config_params['model_name_or_path'] = pre_model_path[config_params['model_type']]
    config_params['model_save_path'] = "./output/model_{}".format(config_params['model_type'])
    lc = LanguageModelClassificationTrain(config_params)
    lc.data_preprocess()
    lc.fit()
