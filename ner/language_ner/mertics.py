import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,confusion_matrix, \
    multilabel_confusion_matrix
import numpy as np
from util import *


def txt_classification_metric(pred_prob_infos, std_label_infos, texts, labels=None, is_multilabel=False, threshold=0.5, enclude_neg_label=True):
    """

    :param pred_label_infos: 模型预测结果 [{label1: prob1, label2: prob2, label3: prob3...}, ...]
    :param std_label_infos: 标准数据集  [{label1: 0, label2: 1, label3: 0...}, ...]， 每个样本对应一个字典，字典键为label，
                            值为1表示样本为当前label, 多标签模式下，可能多个label为1。neg text 模式下，所有label为0，表示为neg_text
    :param labels:  需要计算指标的label集合,默认为 None,表示全部label需要计算指标，例子[label1, label2...]
    :param is_multilabel: 是否为多标签
    :param threshold 阈值，默认为 0.5
    :param enclude_neg_label 是否使用neg_text模式，如果为真，存在neg_text样本，但是labels 不包含 neg_text, pred_prob_infos, std_label_infos 也不包含
    :return:
        rpt_info: 各类别的指标，包括 f1， precision, recall, roc_auc, support count
        {label1: {f1: value, 'precision': value, 'recall': value, "roc_auc": value, 'support': value},
         label2: {f1: value, 'precision': value, 'recall': value, "roc_auc": value, 'support': value},
         total: {f1: value, 'precision': value, 'recall': value, "roc_auc": value, 'support': value},

        confusion_matrix: 混淆矩阵，
         如果为多分类，
               enclude_neg_label = False 为一个 n*n的矩阵， n=len(labels),
               enclude_neg_label = True 为 (n+1)*(n+1) 的矩阵，n=len(labels)
         如果为多标签分类，则为 n*2*2的矩阵， n=len(labels)
    """
    assert len(pred_prob_infos) == len(std_label_infos), "pred, std not match"
    if labels is None:
        labels = list(std_label_infos[0].keys())
    pred_prob_df = pd.DataFrame(pred_prob_infos)
    std_label_df = pd.DataFrame(std_label_infos)
    pred_prob_df = pred_prob_df[labels]
    if not is_multilabel:
        if enclude_neg_label:
            pred_label_df = pred_prob_df.applymap(lambda x : 1 if x > threshold else 0)
        else:
            pred_label_ids = pred_prob_df.values.argmax(axis=1)
            pred_label_vecs = []
            for _id in pred_label_ids:
                tmp_vec = np.zeros(len(labels))
                tmp_vec[_id] = 1
                pred_label_vecs.append(tmp_vec)
            pred_label_df = pd.DataFrame(pred_label_vecs, columns=labels)
    else:
        pred_label_df = pred_prob_df.applymap(lambda x : 1 if x > threshold else 0)

    std_label_df = std_label_df[labels]
    f1_all_class = f1_score(std_label_df.values, pred_label_df.values, average=None)
    precision_all_class = precision_score(std_label_df.values, pred_label_df.values, average=None)
    recall_all_class = recall_score(std_label_df.values, pred_label_df.values, average=None)
    auc_all_class = roc_auc_score(std_label_df.values, pred_prob_df.values, average=None)
    support_s = std_label_df.values.sum(axis=0)
    total_f1 = f1_score(std_label_df.values, pred_label_df.values, average='micro')
    total_precision = precision_score(std_label_df.values, pred_label_df.values, average='micro')
    total_recall = recall_score(std_label_df.values, pred_label_df.values, average='micro')
    total_auc = roc_auc_score(std_label_df.values, pred_prob_df.values, average='micro')
    total_support = support_s.sum()
    labels.append("total")
    f1_all_class = np.append(f1_all_class, total_f1)
    precision_all_class = np.append(precision_all_class, total_precision)
    recall_all_class = np.append(recall_all_class, total_recall)
    auc_all_class = np.append(auc_all_class, total_auc)
    support_s = np.append(support_s, total_support)
    rpt_df = pd.DataFrame({'f1': f1_all_class, 'precision': precision_all_class, "recall": recall_all_class,
                           "roc_auc": auc_all_class, "support": support_s}, index=labels)
    rpt_info = rpt_df.T.to_dict()

    if not is_multilabel:
        std_label_ids = transfer_label_vec_to_label_id(std_label_df, enclude_neg_label)
        pred_label_ids = transfer_label_vec_to_label_id(pred_label_df, enclude_neg_label)
        tgt_labels = list(std_label_df.columns)
        if enclude_neg_label:
            tgt_labels.insert(0, "neg_text")
        std_labels = [tgt_labels[_] for _ in std_label_ids]
        pred_labels = [tgt_labels[_] for _ in pred_label_ids]
        accuracy = accuracy_score(std_labels, pred_labels)
        cf_mat = confusion_matrix(std_labels, pred_labels, labels=tgt_labels)
        cf_mat_df = pd.DataFrame(cf_mat, columns=tgt_labels, index=tgt_labels)
        rpt_info['total'].update({'accuracy': accuracy})
        show_pred_probs = [{pred_label: prob_info[pred_label]} if pred_label in prob_info else {} for pred_label, prob_info in zip(pred_labels, pred_prob_infos) ]
        std_pred_cmp_df = pd.DataFrame({"standard_label": std_labels, "pred_label": pred_labels, "pred_prob": show_pred_probs, "text": texts, "standard_label_index": std_label_infos})

    else:
        tgt_labels = list(std_label_df.columns)
        cf_mat = multilabel_confusion_matrix(std_label_df.values, pred_label_df.values)
        cf_mat_s = []
        for i, label in enumerate(tgt_labels):
            tmp_cf_mat_df = pd.DataFrame(cf_mat[i], index=[label, label])
            cf_mat_s.append(tmp_cf_mat_df)
        cf_mat_df = pd.concat(cf_mat_s)
        pred_multilabels = transfer_label_vec_to_multilabel(pred_label_df)
        std_multilabels = transfer_label_vec_to_multilabel(std_label_df)
        show_pred_probs = [{label: prob for label, prob in prob_info.items()} for prob_info in
                            pred_prob_infos]
        std_pred_cmp_df = pd.DataFrame({"standard_label": std_multilabels, "pred_label": pred_multilabels,
                                        "pred_prob": show_pred_probs, "text": texts, "standard_label_index": std_label_infos})
    std_pred_cmp_infos = std_pred_cmp_df.to_dict('records')
    return rpt_info, cf_mat_df, std_pred_cmp_infos


def name_entity_recognition_metric(pred_ner_infos, std_ner_infos):
    """
    实体识别，槽位识别 算法结果指标计算
    :param pred_ner_infos:
          [{'text': "", "entities": [{"entity_type": "org", "start_pos": 4, "end_pos": 6, "word": ""}]}]
    :param std_ner_infos:
    :return:
    rpt_info， 实体级别的指标
    {"entity_type1": {"f1": value, "precision": value, "recall": value, "support": value},
     "entity_type2": {"f1": value, "precision": value, "recall": value, "support": value},
     "total": {"f1": value, "precision": value, "recall": value, "support": value}}
    confusion_matrix 标签级别的指标
    (n+1)*(n+1)的矩阵， n为实体类型数量
    """
    assert len(pred_ner_infos) == len(std_ner_infos), "pred, std sentence not match"
    entity_types = set()
    std_entity_count = {}
    pred_entity_count = {}
    match_entity_count = {}
    pred_std_cmp_ner_infos = []
    for pred_ner_info, std_ner_info in zip(pred_ner_infos, std_ner_infos):
        pred_std_cmp_ner_infos.append({"text": std_ner_info['text'], "pred_entites": pred_ner_info["entities"],
                                       "standard_entities": std_ner_info['entities']})
        text_pred_entity_pos_infos = {}
        for entity_info in pred_ner_info['entities']:
            entity_type = entity_info['entity_type']

            start_pos = entity_info['start_pos']
            end_pos = entity_info['end_pos']
            if entity_type in pred_entity_count:
                pred_entity_count[entity_type] += 1
            else:
                pred_entity_count.update({entity_type: 1})
            if entity_type in text_pred_entity_pos_infos:
                text_pred_entity_pos_infos[entity_type].append([start_pos, end_pos])
            else:
                text_pred_entity_pos_infos.update({entity_type: [[start_pos, end_pos]]})

        for entity_info in std_ner_info['entities']:
            entity_type = entity_info['entity_type']
            entity_types.add(entity_type)
            start_pos = entity_info['start_pos']
            end_pos = entity_info['end_pos']
            if entity_type in std_entity_count:
                std_entity_count[entity_type] +=1
            else:
                std_entity_count.update({entity_type: 1})
            entity_pos = [start_pos, end_pos]
            entity_type_pred_pos_info = text_pred_entity_pos_infos.get(entity_type, [])
            if entity_pos in entity_type_pred_pos_info:
                if entity_type in match_entity_count:
                    match_entity_count[entity_type] += 1
                else:
                    match_entity_count.update({entity_type: 1})

    rpt_info = {}
    for entity_type in entity_types:
        pred_count = pred_entity_count.get(entity_type, 0)
        std_count = std_entity_count.get(entity_type, 0)
        match_count = match_entity_count.get(entity_type, 0)
        p, r, f1 = calculate_f1_p_r(pred_count, std_count, match_count)
        rpt_info.update({entity_type: {'f1': f1, 'precision': p, 'recall': r, 'support': std_count}})
    total_pred_count = sum(pred_entity_count.values())
    total_std_count = sum(std_entity_count.values())
    total_match_count = sum(match_entity_count.values())
    total_p, total_r, total_f1 = calculate_f1_p_r(total_pred_count, total_std_count, total_match_count)
    rpt_info.update({"total": {'f1': total_f1, 'precision': total_p, "recall": total_r, "support": total_std_count}})
    total_std_labels = transfer_ner_results_to_token_labels(std_ner_infos)
    total_pred_labels = transfer_ner_results_to_token_labels(pred_ner_infos)
    assert len(total_std_labels) == len(total_pred_labels), "std label count not match pred label count"
    entity_type_list = list(entity_types)
    entity_type_list.append("0")
    cf_mat = confusion_matrix(total_std_labels, total_pred_labels, labels=entity_type_list)
    cf_mat_df = pd.DataFrame(cf_mat, columns=entity_type_list, index=entity_type_list)
    return rpt_info, cf_mat_df, pred_std_cmp_ner_infos, entity_type_list


def relation_extraction_metric(pred_relation_infos, std_relation_infos):
    """
    关系抽取的评价指标
    :param pred_relation_infos:
    :param std_relation_infos:
    :return:
    metric_dict 各种关系的 f1, precision, recall, support
    {relation_type1: {"f1": value, "precision": value, "recall": value, "support": value},
     relation_type2: {"f1": value, "precision": value, "recall": value, "support": value},
     "total": {"f1": value, "precision": value, "recall": value, "support": value}}
     confusion_matrix 混淆矩阵
     n*n 的矩阵，n为关系类型数量
    """
    assert len(pred_relation_infos) == len(std_relation_infos), "pred, std sentence not match"

    pred_std_cmp_relation_infos = []
    for pred_relation_info, std_relation_info in zip(pred_relation_infos, std_relation_infos):
        txt = std_relation_info['text']
        pred_std_cmp_relation_infos.append({'text': txt,
                                            "pred_relations": pred_relation_info['relationships'],
                                            "standard_relations": std_relation_info['relationships']})
    flatten_pred_relation_infos = transfer_relation_infos_to_flatten(pred_relation_infos)
    flatten_std_relation_infos = transfer_relation_infos_to_flatten(std_relation_infos)
    relation_set = list(set([relation_type for _, relation_type in flatten_std_relation_infos]))

    flatten_pred_relation_df = pd.DataFrame(flatten_pred_relation_infos, columns=['pos', 'pred_relation_type'])
    flatten_std_relation_df = pd.DataFrame(flatten_std_relation_infos, columns=['pos', 'std_relation_type'])
    flatten_relation_df = pd.merge(flatten_pred_relation_df, flatten_std_relation_df, how='inner', on='pos')
    pred_relation_counts = flatten_pred_relation_df['pred_relation_type'].value_counts().to_dict()
    std_relation_counts = flatten_std_relation_df['std_relation_type'].value_counts().to_dict()
    is_match = flatten_relation_df.apply(lambda s: s['pred_relation_type'] == s['std_relation_type'], axis=1)
    match_counts = flatten_relation_df[is_match]['std_relation_type'].value_counts().to_dict()
    cf_mat = confusion_matrix(flatten_relation_df['std_relation_type'].values, flatten_relation_df['pred_relation_type'].values, relation_set)
    rpt_info = {}
    for relation_type in relation_set:
        pred_relation_count = pred_relation_counts.get(relation_type, 0)
        std_relation_count = std_relation_counts.get(relation_type, 0)
        match_count = match_counts.get(relation_type, 0)
        precision, recall, f1 = calculate_f1_p_r(pred_relation_count, std_relation_count, match_count)
        rpt_info.update({relation_type: {"f1": f1, "precision": precision, "recall": recall, "support": std_relation_count}})
    total_p, total_r, total_f1 = calculate_f1_p_r(sum(pred_relation_counts.values()), sum(std_relation_counts.values()), sum(match_counts.values()))
    rpt_info.update({'total': {"f1": total_f1, "precision": total_p, "recall": total_r, "support": sum(std_relation_counts.values())}})
    cf_mat_df = pd.DataFrame(cf_mat, columns=relation_set, index=relation_set)
    return rpt_info, cf_mat_df


def event_extraction_metric(pred_event_infos, std_event_infos, event_argument_info):
    """
    事件抽取 评估指标
    :param pred_event_infos:
    :param std_event_infos:
    :return:
    rpt_info 事件 及事件要素的 f1, precision, recall, support
    {"event_type1": {"event": {"f1": value, "precision": value, "recall": value, "support": value},
                      "event_argument1": {"f1": value, "precision": value, "recall": value, "support": value},
                      }}
    """
    assert len(pred_event_infos) == len(std_event_infos), "pred, std sentence not match"
    pred_std_cmp_event_infos = []
    for pred_event_info, std_event_info in zip(pred_event_infos, std_event_infos):
        text = pred_event_info['text']
        pred_std_cmp_event_infos.append({"text": text,
                                         "pred_event": pred_event_info['events'],
                                         "standard_event": std_event_info['events']})
    event_types = list(event_argument_info.keys())
    flatten_pred_event_info = transfer_event_infos_to_flatten(pred_event_infos, event_argument_info)
    flatten_std_event_info = transfer_event_infos_to_flatten(std_event_infos, event_argument_info)
    rpt_info = {}
    for event_type in event_types:
        pred_count_dict = {}
        std_count_dict = {}
        match_count_dict = {}
        key_arguments = ['idx']
        argument_info = event_argument_info[event_type]
        [key_arguments.append(info["name"]) for info in argument_info if info['is_key']]
        pred_info_list = flatten_pred_event_info.get(event_type, [])
        std_info_list = flatten_std_event_info.get(event_type, [])

        if len(pred_info_list):
            pred_event_df = pd.DataFrame(pred_info_list)
            pred_event_df.drop_duplicates(subset=key_arguments, inplace=True)
            pred_event_count = len(pred_event_df)
            for info in argument_info:
                argument_name = info['name']
                argument_pred_count = pred_event_df[argument_name].notnull().sum()
                pred_count_dict.update({argument_name: argument_pred_count})
            pred_count_dict.update({'event': pred_event_count})
        else:
            pred_event_count = 0
            pred_count_dict = {info['name']: 0 for info in argument_info}
            pred_count_dict.update({'event': pred_event_count})

        if len(std_info_list):
            std_event_df = pd.DataFrame(std_info_list)
            std_event_df.drop_duplicates(subset=key_arguments, inplace=True)
            std_event_count = len(std_event_df)
            for info in argument_info:
                argument_name = info['name']
                argument_std_count = std_event_df[argument_name].notnull().sum()
                std_count_dict.update({argument_name: argument_std_count})
            std_count_dict.update({'event': std_event_count})
        else:
            std_count_dict = {info['name']: 0 for info in argument_info}
            std_event_count = 0
            std_count_dict.update({'event': std_event_count})
        if std_event_count and pred_event_count:
            merged_event_df = pd.merge(pred_event_df, std_event_df, on=key_arguments, how='inner')
            # print(merged_event_df)
            match_event_count = len(merged_event_df)

            for info in argument_info:
                argument_name = info['name']
                is_key = info['is_key']
                if is_key:
                    argument_match_count = len(merged_event_df)
                else:
                    if len(merged_event_df):
                        argument_match = merged_event_df.apply(lambda s: s["%s_x" % argument_name] == s['%s_y' % argument_name], axis=1)

                        argument_match_count = argument_match.sum()
                    else:
                        argument_match_count = 0
                match_count_dict.update({argument_name: argument_match_count})

            match_count_dict.update({'event': match_event_count})

        else:
            match_count_dict = {info['name']: 0 for info in argument_info}
            match_event_count = 0
            match_count_dict.update({"event": match_event_count})
        argument_rpt_info = {}
        for info in argument_info:
            argument_name = info['name']
            p, r, f1 = calculate_f1_p_r(pred_count_dict[argument_name], std_count_dict[argument_name], match_count_dict[argument_name])
            argument_rpt_info.update({argument_name: {"f1": f1, 'precision': p, "recall": r, "support": std_count_dict[argument_name]}})
        event_p, event_r, event_f1 = calculate_f1_p_r(pred_count_dict["event"], std_count_dict["event"], match_count_dict["event"])
        tmp_rpt_info = \
            {
                "event": {"f1": event_f1, "precision": event_p, "recall": event_r, "support": std_count_dict["event"]},
                "argument": argument_rpt_info
            }

        rpt_info.update({event_type: tmp_rpt_info})
    return rpt_info, pred_std_cmp_event_infos

