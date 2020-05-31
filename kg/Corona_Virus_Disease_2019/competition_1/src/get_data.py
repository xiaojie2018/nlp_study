# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/28 17:10
# software: PyCharm

import os
from competition_1.src import test_file_path, train_file_path
from tqdm import tqdm


def read_(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', '').split('\t'))
    return data


def read_xlm_3(file):
    from xml.dom.minidom import parse
    labels = ["id", "ns", "title"]
    labels1 = ["comment", "text"]
    DOMTree = parse(file)
    DOMTree_childNodes = DOMTree._get_childNodes()[0].childNodes
    data = []
    for dc in DOMTree_childNodes:
        o = {}
        if dc.nodeName == "page":
            for b in dc.childNodes:
                if b.nodeName in labels:
                    o[b.nodeName] = b.childNodes[0].data
                if b.nodeName == "revision":
                    for c in b.childNodes:
                        if c.nodeName in labels1:
                            o[c.nodeName] = c.childNodes[0].data
            data.append(o)
    return data


def read_xlm_1(file):
    from xml.dom.minidom import parse
    DOMTree = parse(file)
    print(1)


if __name__ == '__main__':

    # entity_type_file = os.path.join(train_file_path, "entity_type.txt")
    # entity_type = read_(entity_type_file)

    # entity_pages_3_file = os.path.join(train_file_path, "entity_pages_3.xml")
    # read_xlm_3(entity_pages_3_file)

    entity_pages_1_file = os.path.join(train_file_path, "entity_pages_4.xml")
    read_xlm_1(entity_pages_1_file)
