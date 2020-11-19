
import os
import json
import random


def read_query_file(file):
    data = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').split('\t')
            if line[0] not in data:
                data[line[0]] = line[1]
            else:
                print(line)
    return data


def read_reply_file(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').split('\t')
            data.append(line)
    return data


def get_data(query_data, reply_data, output_file):

    data = []
    for d in reply_data:
        q = query_data[d[0]]
        r = d[2]
        l = 0
        if len(d) > 3:
            l = int(d[3])
        data.append({"id1": d[0], 'id2': d[1], "text1": q, "text2": r, "label": l})

    f = open(output_file, 'w', encoding='utf-8')
    for d in data:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    f.close()


def split_train_test(file, train_file, test_file):

    data = {}
    data_len = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = eval(line)
            if line['label'] not in data:
                data[line['label']] = []
            data[line['label']].append(line)
            data_len.append(len(line['text1']+line['text2']))
            # data_len.append(len(line['text1']))
            # data_len.append(len(line['text2']))

    # train_data, test_data = [], []
    # for v in data.values():
    #     random.shuffle(train_data)
    #     train_len = int(len(v)*0.8)
    #     train_data += v[:train_len]
    #     test_data += v[train_len:]
    #
    # f = open(train_file, 'w', encoding='utf-8')
    # for d in train_data:
    #     json.dump(d, f, ensure_ascii=False)
    #     f.write('\n')
    # f.close()
    # f = open(test_file, 'w', encoding='utf-8')
    # for d in test_data:
    #     json.dump(d, f, ensure_ascii=False)
    #     f.write('\n')
    # f.close()

    print(max(data_len))
    print(min(data_len))
    print(sum(data_len)/len(data_len))


if __name__ == '__main__':

    flag = ['train', 'test'][1]

    query_file = "D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\{}\\{}.query.tsv".format(flag, flag)
    reply_file = "D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\{}\\{}.reply.tsv".format(flag, flag)

    query_data = read_query_file(query_file)
    reply_data = read_reply_file(reply_file)

    output_file = "D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\{}\\{}.json".format(flag, flag)
    get_data(query_data, reply_data, output_file)

    # train_file = "D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\{}\\{}_train.json".format(flag, flag)
    # test_file = "D:\\bishai\\data\\ccf2020\\房产行业聊天问答匹配\\{}\\{}_test.json".format(flag, flag)
    # split_train_test(output_file, train_file, test_file)
