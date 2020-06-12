# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/8 16:30
# software: PyCharm

import json
import re
import random
import os


def jiexi(data):
    texts = []
    labels = []
    intents = []
    for d in data:
        text = d['text'].replace(' ', '')
        intent = d['intent']
        slots = d['slots']
        if len(slots) == 0:
            texts.append(list(text))
            labels.append(["O"]*len(list(text)))
            intents.append(intent)
            continue
        elif len(slots) == 1:
            for k, v in slots.items():
                v = v.replace(' ', '')
                pattern = re.compile(u"{}".format(v))
                res = []
                for r in pattern.finditer(text):
                    res.append([r.group(), r.regs[0]])
                l = ["O"]*len(list(text))
                for r in res:
                    for i in range(r[1][0], r[1][1]):
                        l[i] = "I-{}".format(k)
                    l[r[1][0]] = "B-{}".format(k)
                assert len(list(text)) == len(l)
                texts.append(list(text))
                labels.append(l)
                intents.append(intent)

        else:
            inv_slots = {v: k for k, v in slots.items()}
            vvv = [v for v in slots.values()]
            vvv = sorted(vvv, key=lambda x: len(x), reverse=True)
            pattern = re.compile(u"({})".format("|".join(vvv)))
            res = []
            for r in pattern.finditer(text):
                res.append([r.group(), r.regs[0]])
            l = ["O"]*len(list(text))
            for r in res:
                k = r[0]
                for i in range(r[1][0], r[1][1]):
                    l[i] = "I-{}".format(inv_slots[k])
                l[r[1][0]] = "B-{}".format(inv_slots[k])
            assert len(list(text)) == len(l)
            texts.append(list(text))
            labels.append(l)
            intents.append(intent)
    res = []
    for x, y, z in zip(texts, labels, intents):
        res.append((x, y, z))
    return res


def wrr(data, ty="train"):
    fi = './{}'.format(ty)
    os.makedirs(fi)
    texts = []
    labels = []
    intents = []
    for d in data:
        texts.append(d[0])
        labels.append(d[1])
        intents.append(d[2])
    file1 = os.path.join(fi, 'seq.in')
    file2 = os.path.join(fi, 'seq.out')
    file3 = os.path.join(fi, "label")
    f1 = open(file1, 'w', encoding='utf-8')
    f2 = open(file2, 'w', encoding='utf-8')
    f3 = open(file3, 'w', encoding='utf-8')

    for s in texts:
        f1.write(' '.join(s))
        f1.write('\n')
    f1.close()

    for s in labels:
        f2.write(' '.join(s))
        f2.write('\n')
    f2.close()

    for s in intents:
        f3.write(s+'\n')
    f3.close()


if __name__ == '__main__':
    data = json.load(open('./smp_2019_task1_train.json', 'r', encoding='utf-8'))
    res = jiexi(data)
    random.shuffle(res)
    train_len = int(len(res)*0.8)
    train_data = res[:train_len]
    test_data = res[train_len:len(res)]
    wrr(train_data, "train")
    wrr(test_data, "test")
    wrr(test_data, "dev")
    print(1)
