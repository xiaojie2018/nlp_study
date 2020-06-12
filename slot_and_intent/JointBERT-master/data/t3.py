# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/8 15:17
# software: PyCharm

import os


def read_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


def jiexi(labels):
    res = []
    for l in labels:
        if l.startswith("B-"):
            res.append(l[2:])
    return res


if __name__ == '__main__':
    ty1 = ["atis", "snips", "other"][2]
    num_slot_intent = {}
    for ty in ["train", "dev", "test"]:
        file1 = os.path.join(os.path.join(ty1, ty), "seq.out")
        file2 = os.path.join(os.path.join(ty1, ty), "label")

        res_slot = read_file(file1)
        res_intent = read_file(file2)

        num_slots = {}
        num_intents = {}

        for x, y in zip(res_slot, res_intent):
            x = x.split(' ')
            slots = jiexi(x)
            for s in slots:
                if s not in num_slots:
                    num_slots[s] = 0
                num_slots[s] += 1
            if y not in num_intents:
                num_intents[y] = 0
            num_intents[y] += 1
            if y not in num_slot_intent:
                num_slot_intent[y] = []
            for s in slots:
                num_slot_intent[y].append(s)

        print(1)
    for k, v in num_slot_intent.items():
        num_slot_intent[k] = sorted(list(set(v)))
    print(1)











