# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:56
# software: PyCharm


from utils import EntityClassificationDataPreprocess, init_logger
from argparse import Namespace
from trainer import Trainer
import logging
import os
import json
import codecs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
init_logger()
logger = logging.getLogger(__name__)


class LanguageModelEntityClassificationTrain(EntityClassificationDataPreprocess):

    def __init__(self, config_params):
        self.config = Namespace(**config_params)
        self.config.no_cuda = False
        self.model_save_path = self.config.model_save_path

        super(LanguageModelEntityClassificationTrain, self).__init__(self.config)

    def data_preprocess(self):
        if self.config.train_file_url.endswith('txt'):
            train_data, labels1 = self._get_data_txt(self.config.train_file_url)
            # train_data1, labels11 = self._get_data_txt('./output_data/result_ernie_2.txt')
            train_data1 = []
            train_data = train_data + train_data1
            test_data, labels2 = train_data, labels1
            dev_data, labels3 = train_data, labels1
        train_data, labels1 = self.get_event_data_json(self.config.train_file_url)

        # train_data, labels1 = self.get_event_data_json(self.config.test_file_url)

        types = self.tongji(train_data)

        self.config.types = types
        """
        {'collateral': 2020, 'proportion': 1146, 'obj-org': 1697, 'trigger': 4682, 'number': 706, 'date': 1770, 
         'sub-org': 1355, 'sub': 1575, 'obj': 1700, 'money': 803, 'target-company': 1555, 'neg_text': 5799, 
         'share-org': 238, 'title': 285, 'sub-per': 553, 'obj-per': 189, 'share-per': 15}
        
        """
        train_data = self.trans(train_data, self.config.max_seq_len)
        train_data = train_data
        test_data = train_data
        dev_data = train_data
        labels2 = labels1
        labels3 = labels1
        # test_data, labels2 = self.get_data_json(self.config.test_file_url)
        # dev_data, labels3 = self.get_data_json(self.config.dev_file_url)

        labels = sorted(list(set(labels1 + labels2 + labels3)) + ["neg_text11"])

        self.labels = labels
        self.config.num_classes = len(labels)

        self.label_id = {l: ind for ind, l in enumerate(labels)}
        self.id_label = {ind: l for ind, l in enumerate(labels)}
        self.config.label_id = self.label_id
        self.config.id_label = self.id_label
        self.config.labels = self.labels

        self.train_data = self._get_data(train_data, self.label_id, types, set_type="train")

        logger.info("train data num: {} ".format(str(len(train_data))))
        self.test_data = self._get_data(test_data, self.label_id, types, set_type="test")
        logger.info("test data num: {} ".format(str(len(test_data))))
        self.dev_data = self._get_data(dev_data, self.label_id, types, set_type="dev")
        logger.info("dev data num: {} ".format(str(len(dev_data))))

    def fit(self):

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
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
        "seed": 42,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "max_seq_len": 200,
        "learning_rate": 5e-5,
        "num_train_epochs": 16,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "dropout_rate": 0.5,
        "logging_steps": 500,
        "save_steps": 500,
        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "is_attention": False,
        "is_lstm": False,
        "is_cnn": False,
        "train_file_url": "../ccks_3_nolabel_data//train_base.json",
        "test_file_url": "../ccks_3_nolabel_data//trans_train.json",
        "dev_file_url": "./o_data/entity_type.txt",
        "job_name": "dialog_intent_classification",
        "model_save_path": "./output/model",
        "is_muti_label": False,
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
        "bert": "{}/bert-base-chinese".format(lag_path),  # jindong  bert-base-chinese
        "ernie": "{}/ERNIE_stable-1.0.1-pytorch".format(lag_path),  # ERNIE_stable-1.0.1-pytorch   ERNIE  ERNIE_1.0_max-len-512-pytorch
        "albert": "{}/albert_base_v1".format(lag_path),
        "roberta": "{}/chinese_roberta_wwm_ext_pytorch".format(lag_path),
        "bert_www": "{}/chinese_wwm_pytorch".format(lag_path),
        "xlnet_base": "{}/chinese_xlnet_base_pytorch".format(lag_path),
        "xlnet_mid": "{}/chinese_xlnet_mid_pytorch".format(lag_path),
        "electra_base_discriminator": "{}/chinese_electra_base_discriminator_pytorch".format(lag_path),
        # "electra_base_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_base_generator_pytorch",
        "electra_small_discriminator": "{}/chinese_electra_small_discriminator_pytorch".format(lag_path),
        # "electra_small_generator": "E:\\nlp_tools\\electra_models\\chinese_electra_small_generator_pytorch",
    }

    config_params['model_type'] = model_type[1]
    config_params['model_name_or_path'] = pre_model_path[config_params['model_type']]
    config_params['pretrained_model_path'] = pre_model_path[config_params['model_type']]
    config_params['model_save_path'] = "./output_2/model_{}".format(config_params['model_type'])
    lc = LanguageModelEntityClassificationTrain(config_params)
    lc.data_preprocess()
    lc.fit()


    {14526: "责问客服信息", 14532: "投诉或不满", 11071: "投诉或不满", 11069: "客户端录音", 14527: "客户端录音",
     14528: "代表银行-最终答复", 14530: "询问信访和上级信息", 14531: "制度规定依据", 11070: "制度规定依据",
     14533: "询问投诉监督电话", 11072: "询问投诉监督电话",
     14534: "威胁安全", 13535: "媒体曝光", 14536: "总行高管投诉", 14537: "银行监管机构投诉", 11073: "银行监管机构投诉",
     14538: "群众渠道投诉", 11075: "起诉建行", 14540: "起诉建行", 11074: "道歉赔偿", 14539: "道歉赔偿", 14541: "投诉倾向-其他",
     14542: "明确投诉-其他", 14529: "升级倾向"}

