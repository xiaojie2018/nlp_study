# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
import json
from utils import NerDataPreprocess, jiexi
from argparse import Namespace
from trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LanguageModelNerPredict(NerDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'ner_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelNerPredict, self).__init__(self.config)

        self.labels = self.config.labels
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        model_file_path = self.config.model_save_path

        self.id_label = {int(k): v for k, v in self.config.id_label.items()}

        self.config.id_label = self.id_label

        self.trainer = Trainer(self.config)
        self.trainer.load_model(model_file_path)

    def cut_text_len(self, t, max_len):
        t1 = []
        t11 = ''
        for x in t:
            t11 += x
            if len(t11) >= max_len:
                t1.append(t11)
                t11 = ''
        if len(t11) > 0:
            t1.append(t11)
        return t1

    def process(self, texts):
        data = []
        id1_id2 = {}
        indd = 0
        for ind, t in enumerate(texts):
            if len(t) < self.config.max_seq_len-5:
                data.append({"text": t, "entities": []})
                if ind not in id1_id2:
                    id1_id2[ind] = [indd]
                    indd += 1
            else:
                t1 = self.cut_text_len(t, self.config.max_seq_len-2)
                for t11 in t1:
                    data.append({"text": t11, "entities": []})
                    if ind not in id1_id2:
                        id1_id2[ind] = [indd]
                    else:
                        id1_id2[ind].append(indd)
                    indd += 1
        return data, id1_id2

    def tt(self, data):
        res = []
        for d in data:
            label = ['O']*len(d['text'])
            for e in d['entities']:
                for i in range(e['start_pos'], e['end_pos']):
                    label[i] = "I-{}".format(e['entity_type'])
                label[e['start_pos']] = "B-{}".format(e['entity_type'])
            res.append((list(d['text']), label))
        return res

    def predict(self, texts):
        test_data, id1_id2 = self.process(texts)

        if self.config.model_decode_fc in ['softmax', 'crf']:
            test_data_, examples = self._get_data(test_data, self.labels, self.label_id, set_type='predict')
        elif self.config.model_decode_fc == 'span':
            test_data_, examples = self._get_span_data(test_data, self.labels, self.label_id, set_type='predict')

        res = self.trainer.evaluate_test(test_data_, examples)
        if self.config.model_decode_fc == 'span':
            res = self.tt(res)

        result = []
        for ind, t in enumerate(texts):
            r1 = []
            r2 = []
            # for i in id1_id2[ind]:
            #     r1 += res[i][0]
            #     r2 += res[i][1]
            for r in res:
                r1 += r[0]
                r2 += r[1]
            e = jiexi(r1, r2)
            # assert t == ''.join(r1), print(ind)
            if t != ''.join(r1):
                print(ind)
                print(''.join(r1))
                print(t)
                print('*'*70)
            result.append({'text': "".join(r1), "entities": e})
        return result


def read_test_data(file):
    from tqdm import tqdm
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(line.replace('\n', ''))
    return data


def read_json(file):
    data = json.load(open(file, 'r', encoding='utf-8'))
    return data


if __name__ == '__main__':
    file = "./output/model_ernie_softmax_1025_1"
    file = "./output/model_ernie_span_1025_2"
    file = "./output/model_ernie_crf_1025_3"
    file = "./output/model_bert_crf_1029_1"
    file = './output/model_bert_crf_1118_2'
    texts = ['他们也很渴望魔兽比赛。', '为魔兽起到了很好的推动作用。']
    lcp = LanguageModelNerPredict(file)
    res = lcp.predict(texts)
    print(res)
    # predict.csv

    file1 = './ccf_data/eval.json'
    out_file = './tijiao/predict1118_2.csv'
    # ID	Category	Pos_b	Pos_e	Privacy
    predict_data = read_json(file1)
    res = []
    from tqdm import tqdm
    for d in tqdm(predict_data):
        r = lcp.predict([d])
        res += r

    # res = lcp.predict(predict_data)
    import pandas as pd
    df = pd.DataFrame()
    result = []
    for ind, r in enumerate(res):
        for e in r['entities']:
            w = e['word']
            if w.isdigit() and e['entity_type'] == 'mobile':
                w = int(w)
            result.append([ind, e['entity_type'], e['start_pos'], e['end_pos']-1, w])
    df['ID'] = [s[0] for s in result]
    df['Category'] = [s[1] for s in result]
    df['Pos_b'] = [s[2] for s in result]
    df['Pos_e'] = [s[3] for s in result]
    df['Privacy'] = [s[4] for s in result]
    df.to_csv(out_file, encoding='utf-8', index=None)
