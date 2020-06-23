# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/11 10:03
# software: PyCharm

import os
import pandas as pd

file_path = "E:\\bishai\\四川创新数字大赛\\诈骗电话识别\\诈骗电话号码识别-0527"
file_path = '/home/hemei/xjie/sic/o_data'

train_file_path = os.path.join(file_path, "train")
test_file_path = os.path.join(file_path, "test")

types = ['app', 'sms', 'user', 'voc']

t = types[3]
# train_file = os.path.join(train_file_path, 'train_{}.csv'.format(t))
# test_file = os.path.join(test_file_path, 'test_{}.csv'.format(t))


train_file_app = os.path.join(train_file_path, 'train_{}.csv'.format('app'))
train_file_sms = os.path.join(train_file_path, 'train_{}.csv'.format('sms'))
train_file_user = os.path.join(train_file_path, 'train_{}.csv'.format('user'))
train_file_voc = os.path.join(train_file_path, 'train_{}.csv'.format('voc'))

test_file_app = os.path.join(test_file_path, 'test_{}.csv'.format('app'))
test_file_sms = os.path.join(test_file_path, 'test_{}.csv'.format('sms'))
test_file_user = os.path.join(test_file_path, 'test_{}.csv'.format('user'))
test_file_voc = os.path.join(test_file_path, 'test_{}.csv'.format('voc'))


def get_data():
    data_user = pd.read_csv(train_file_user)
    user_title = ['phone_no_m', 'city_name', 'county_name', 'idcard_cnt', 'arpu_201908', 'arpu_201909', 'arpu_201910',
                  'arpu_201911', 'arpu_201912', 'arpu_202001', 'arpu_202002', 'arpu_202003', 'label'][:5]
    
    data = []
    data_len = []
    labels = []
    for i in range(data_user.shape[0]):
        d1 = data_user.iloc[i].tolist()
        d2 = []
        nn = 0
        for x, y in zip(user_title, d1):
            d2.append((x, str(y)))
            nn += len(x)
            nn += len(str(y))
        data_len.append(nn)
        data.append([d2, str(d1[-1])])
        labels.append(str(d1[-1]))

    max_len = max(data_len) + len(user_title)*2+2
    labels = sorted(list(set(labels)))
    
    return data, labels, max_len


