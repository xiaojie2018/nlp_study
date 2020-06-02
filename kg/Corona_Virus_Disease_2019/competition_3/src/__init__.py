# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 23:43
# software: PyCharm

import os

# file_path = "E:\\bishai\\数据集\\CCKS 2020：新冠知识图谱构建与问答评测（三）新冠科研抗病毒药物图谱的链接预测\\ccks_7_3_data_nolabel"
file_path = "/home/hemei/xjie/kg_bert/ccks_7_3_data_nolabel"

attrs_file = os.path.join(file_path, "attrs.json")
entities_file = os.path.join(file_path, "entities.json")
link_prediction_file = os.path.join(file_path, "link_prediction.json")
relationships_file = os.path.join(file_path, "relationships.json")
schema_file = os.path.join(file_path, "schema.json")
virus2sequence_file = os.path.join(file_path, "virus2sequence.json")
