# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/12 8:51
# software: PyCharm


import pandas as pd


res = {}
for i in range(4, 12):
    data_user = pd.read_csv('./result_test_{}.csv'.format(i))
    user_title = ['phone_no_m', 'label']
    data = data_user['phone_no_m']
    label = data_user['label']
    for x, y in zip(data, label):
        if x not in res:
            res[x] = {}
        if y not in res[x]:
            res[x][y] = 1
        else:
            res[x][y] += 1

phone_no_m = []
label = []
for k, v in res.items():
    v2 = sorted([(k1, v1) for k1, v1 in v.items()], key=lambda x: x[-1], reverse=True)
    phone_no_m.append(k)
    label.append(v2[0][0])

df = pd.DataFrame()
df['phone_no_m'] = phone_no_m
df['label'] = label
df.to_csv('./result_test_all_4_12.csv', index=0)

