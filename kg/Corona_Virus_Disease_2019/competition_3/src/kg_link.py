# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/29 23:49
# software: PyCharm


from competition_3.src import attrs_file, entities_file, link_prediction_file, relationships_file, schema_file, virus2sequence_file
from competition_3.src.utils import DataProcess


class KGLink(DataProcess):

    def __init__(self):
        self.entities = self.read_all_entity(entities_file)
        self.triple, entities1 = self.read_relationships(relationships_file)
        self.attrs = self.read_attrs(attrs_file)
        self.virus2sequence = self.read_virus2sequence(virus2sequence_file)
        self.test_data = self.read_test_data(link_prediction_file)


if __name__ == '__main__':
    kgl = KGLink()




