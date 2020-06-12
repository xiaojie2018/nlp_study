# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/4/18 9:40
# software: PyCharm

import json
import os
path = "D:\\ner\\mrc-for-flat-nested-ner-master\\org_data\\ner_n\\event"
# path = "/home/hemei/xjie/mrc-for-flat-nested-ner-master/org_data/ner_n/event"

with open(os.path.join(path, 'event_schema.json'), 'r', encoding='utf-8') as f:
    id2label, label2id, n = {}, {}, 0
    label2type, label2query, label2query1 = {}, {}, {}
    for l in f:
        l = json.loads(l)
        for role in l['role_list']:
            key = (l['event_type'], role['role'])
            id2label[n] = key
            label2id[key] = n
            label2type[key] = "A{}".format(n)
            label2query[key] = (key[0] + ',' + key[1]).replace('/', ',')
            label2query1[label2type[key]] = (key[0] + ',' + key[1]).replace('/', ',')
            n += 1

tags = [s for s in label2type.values()]
zh_event_ner = {
    "tags": tags,
    "natural_query": label2query1,
    "psedo_query": label2query1,
    "label2type": label2type
}
label2type = label2type

# zh_event_ner = {
#     "tags": ["ORG", "NAME", "RACE", "TITLE", "EDU", "LOC", "PRO", "CONT"],
#
#     "natural_query": {
#         "ORG": "组织或机构",
#         "LOC": "地点",
#         "NAME": "姓名",
#         "RACE": "种族",
#         "TITLE": "职称",
#         "EDU": "教育机构",
#         "PRO": "职业背景",
#         "CONT": "国家"
#     },
#     "psedo_query": {
#         "ORG": "组织",
#         "LOC": "地点",
#         "NAME": "姓名",
#         "RACE": "种族",
#         "TITLE": "职称",
#         "EDU": "教育",
#         "PRO": "职业",
#         "CONT": "国家"
#     }
# }