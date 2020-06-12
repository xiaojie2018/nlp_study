# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/28 17:10
# software: PyCharm

import os

file_path = "E:\\bishai\\四川创新数字大赛\\诈骗电话识别\\诈骗电话号码识别-0527"

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


import pandas as pd

data_app = pd.read_csv(train_file_app)
data_sms = pd.read_csv(train_file_sms)
data_user = pd.read_csv(train_file_user)
data_voc = pd.read_csv(train_file_voc)


phone_app = set(data_app['phone_no_m'].tolist())
phone_sms = set(data_sms['phone_no_m'].tolist())
phone_user = set(data_user['phone_no_m'].tolist())
phone_voc = set(data_voc['phone_no_m'].tolist())




