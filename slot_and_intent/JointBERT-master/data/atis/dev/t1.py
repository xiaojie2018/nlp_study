# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/7 11:14
# software: PyCharm

import json


def read_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


words = read_file('./seq.in')
slot_labels = read_file('./seq.out')
intent_label = read_file('./label')

data = []
for x, y, z in zip(words, slot_labels, intent_label):
    data.append({
        "words": x.split(' '),
        "slot_labels": y.split(' '),
        "intent_label": z
    })

f = open('./dev.json', 'w', encoding='utf-8')
for d in data:
    json.dump(d, f)
    f.write('\n')
f.close()
print()
