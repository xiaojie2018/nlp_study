# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/8 14:41
# software: PyCharm


import json
import os


def read_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


if __name__ == '__main__':
    ty1 = ["atis", "snips", "other"][2]
    ts = []
    for ty in ["train", "dev", "test"]:
        file = os.path.join(os.path.join(ty1, ty), "seq.in")
        data = read_file(file)
        print(ty, len(data))
        for d in data:
            ts.append(len(d.split(' ')))

    print(max(ts))
    print(min(ts))
    print(sum(ts)/len(ts))



