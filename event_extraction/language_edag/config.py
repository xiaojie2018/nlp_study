# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 9:43
# software: PyCharm


from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel, AutoConfig, AutoTokenizer, AutoModel, BertModel, BertPreTrainedModel, \
    RobertaModel, AlbertModel


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertModel),
    'bert_www': (BertConfig, BertTokenizer, BertModel),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaModel),
    'albert': (AlbertConfig, AlbertTokenizer, AlbertModel),
    'ernie': (BertConfig, BertTokenizer, BertModel),
    "xlnet_base": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "xlnet_mid": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "electra_base_discriminator": (AutoConfig, AutoTokenizer, AutoModel),
    # "electra_base_generator": (AutoConfig, AutoTokenizer, AutoModel),
    "electra_small_discriminator": (AutoConfig, AutoTokenizer, AutoModel),
    # "electra_small_generator": (AutoConfig, AutoTokenizer, AutoModel),
}

