# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/6/9 10:04
# @author  : Mo
# @function:


from typing import Dict, List, Any
from collections import Counter
# import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, fbeta_score


def metrics_report(y_true: List, y_pred: List, rounded: int=3, epsilon: float=1e-9, use_draw: bool=True):
    """
    calculate metrics and draw picture, 计算评估指标并画图
    code: some code from: 
    Args:
        y_true: list, label of really true, eg. ["TRUE", "FALSE"]
        y_pred: list, label of model predict, eg. ["TRUE", "FALSE"]
        rounded: int, bit you reserved , eg. 6
        epsilon: float, ε, constant of minimum, eg. 1e-6
        use_draw: bool, whether draw picture or not, True
    Returns:
        metrics(precision, recall, f1, accuracy, support), report
    """

    def calculate_metrics(datas: Dict) -> Any:
        """
        calculate metrics after of y-true nad y-pred, 计算准确率, 精确率, 召回率, F1-score
        Args:
            datas: Dict, eg. {"TP": 5, "FP": 3, "TN": 8, "FN": 9}
        Returns:
            accuracy, precision, recall, f1
        """
        accuracy = (datas["TP"] + datas["TN"]) / (datas["TP"] + datas["TN"] + datas["FP"] + datas["FN"] + epsilon)
        precision = datas["TP"] / (datas["TP"] + datas["FP"] + epsilon)
        recall = datas["TP"] / (datas["TP"] + datas["FN"] + epsilon)
        f1 = (precision * recall * 2) / (precision + recall + epsilon)
        accuracy = round(accuracy, rounded)
        precision = round(precision, rounded)
        recall = round(recall, rounded)
        f1 = round(f1, rounded)
        return accuracy, precision, recall, f1

    label_counter = dict(Counter(y_true))
    # 统计每个类下的TP, FP, TN, FN等信息, freq of some (one label)
    freq_so = {yti: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for yti in sorted(set(y_true))}
    for idx in range(len(y_true)):
        correct = y_true[idx]
        predict = y_pred[idx]
        # true
        if correct == predict:
            for k, v in freq_so.items():
                if k == correct:
                    freq_so[correct]["TP"] += 1
                else:
                    freq_so[k]["TN"] += 1
        # flase
        else:
            for k, v in freq_so.items():
                if k == correct:
                    freq_so[correct]["FN"] += 1
                elif k == predict:
                    freq_so[k]["FP"] += 1
                else:
                    freq_so[k]["TN"] += 1
    # 统计每个类下的评估指标("accuracy", "precision", "recall", "f1", "support")
    metrics = {}
    freq_to = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    keys = list(freq_to.keys())
    for k, v in freq_so.items():
        for key in keys:
            freq_to[key] += v[key]
        accuracy, precision, recall, f1 = calculate_metrics(v)
        metrics[k] = {"precision": precision,
                      "recall": recall,
                      "f1-score": f1,
                      "accuracy": accuracy,
                      "support": label_counter[k]}
    # 计算平均(mean)评估指标
    mean_metrics = {}
    for mmk in list(metrics.values())[0].keys():
        for k, _ in metrics.items():
            k_score = sum([v[mmk] for k, v in metrics.items()]) / len(metrics)
            mean_metrics["mean_{}".format(mmk)] = round(k_score, rounded)
    # 计算总计(sum)评估指标
    sum_accuracy, sum_precision, sum_recall, micro_f1 = calculate_metrics(freq_to)
    metrics['mean'] = mean_metrics
    metrics['sum'] = {"sum_precision": sum_precision,
                      "sum_recall": sum_recall,
                      "sum_f1-score": micro_f1,
                      "sum_accuracy": sum_accuracy,
                      "sum_support": sum(label_counter.values())
                      }
    report = None
    if use_draw:
        # 打印头部
        sign_tol = ["mean", "sum"]
        labels = list(label_counter.keys())
        target_names = [u"%s" % l for l in labels] + sign_tol
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, rounded)
        headers = ["precision", "recall", "f1-score", "accuracy", "support"]
        head_fmt = u"{:>{width}s} " + u" {:>9}" * len(headers)
        report = "\n\n" + head_fmt.format("", *headers, width=width)
        report += "\n\n"
        # 具体label的评估指标
        row_fmt = u"{:>{width}s} " + u" {:>9.{rounded}}" * (len(headers)-1) + u" {:>9}\n"
        for li in labels:
            [p, r, f1, a, s] = [metrics[li][hd] for hd in headers]
            report += row_fmt.format(str(li), p, r, f1, a, int(s), width=width, rounded=rounded)
        report += "\n"
        # 评估指标sum, mean
        for sm in sign_tol:
            [p, r, f1, a, s] = [metrics[sm][sm + "_" + hd] for hd in headers]
            report += row_fmt.format(str(sm), p, r, f1, a, int(s), width=width, rounded=rounded)
        report += "\n"
        # logger.info(report)

    return metrics, report


def text_multi_label_classification_evaluate(texts, std_labels, pred_probs):
    assert len(pred_probs) == len(std_labels), "pred, std not match"
    assert len(pred_probs) > 0, "data count must bigger than zero"

    pred_prob_df = pd.DataFrame(pred_probs)
    labels = list(pred_prob_df.columns.values)

    update_std_labels = []
    for _ in std_labels:
        tmp_labels = [tmp_label for tmp_label in _.split(";") if tmp_label in labels]
        tmp_labels.sort()
        update_std_labels.append(tmp_labels)

    std_label_infos = transfer_label_2_vec(update_std_labels, labels)
    std_label_df = pd.DataFrame(std_label_infos, columns=labels)

    pred_label_vecs = []
    pred_labels = []
    for prob_info in pred_probs:
        tmp_pred_labels = []
        for label, prob in prob_info.items():
            if prob > 0.5:
                tmp_pred_labels.append(label)
        tmp_pred_labels.sort()
        pred_labels.append(tmp_pred_labels)
        tmp_vec = [1 if _ in tmp_pred_labels else 0 for _ in labels]
        pred_label_vecs.append(tmp_vec)

    pred_label_df = pd.DataFrame(pred_label_vecs, columns=labels)

    f1_all_class = f1_score(std_label_df.values, pred_label_df.values, average=None)
    precision_all_class = precision_score(std_label_df.values, pred_label_df.values, average=None)
    recall_all_class = recall_score(std_label_df.values, pred_label_df.values, average=None)

    auc_all_class = []
    for label in labels:
        if std_label_df[label].sum() == 0 or std_label_df[label].sum() == len(std_label_df):
            auc_all_class.append(0.5)
        else:
            auc_all_class.append(roc_auc_score(std_label_df[label].values, pred_label_df[label].values))

    support_s = std_label_df.values.sum(axis=0)
    total_f1 = f1_score(std_label_df.values, pred_label_df.values, average='micro')
    total_precision = precision_score(std_label_df.values, pred_label_df.values, average='micro')
    total_recall = recall_score(std_label_df.values, pred_label_df.values, average='micro')
    total_auc = roc_auc_score(std_label_df.values, pred_label_df.values, average='micro')
    total_support = support_s.sum()
    labels.append("total")
    f1_all_class = np.append(f1_all_class, total_f1)
    precision_all_class = np.append(precision_all_class, total_precision)
    recall_all_class = np.append(recall_all_class, total_recall)
    auc_all_class = np.append(auc_all_class, total_auc)
    support_s = np.append(support_s, total_support)
    support_s = [int(_) for _ in support_s]

    rpt_df = pd.DataFrame({"f1": f1_all_class,
                           "precision": precision_all_class,
                           "recall": recall_all_class,
                           "roc_auc": auc_all_class,
                           "support": support_s}, index=labels)

    rpt_info = rpt_df.T.to_dict()

    show_pred_probs = [{_: prob_info[_] for _ in pred_label} for pred_label, prob_info in zip(pred_labels, pred_probs)]

    std_pred_cmp_df = pd.DataFrame({"standard_label": update_std_labels,
                                    "pred_label": pred_labels,
                                    "pred_prob": show_pred_probs,
                                    "text": texts})

    std_pred_cmp_infos = std_pred_cmp_df.to_dict("records")

    cf_mat_infos = []

    return rpt_info, cf_mat_infos, std_pred_cmp_infos, labels


def transfer_label_2_vec(label_list, label_set):
    label_2_id = {_: idx for idx, _ in enumerate(label_set)}
    label_vecs = []
    for label in label_list:
        tmp_vec = [0 for _ in label_set]
        if type(label) == str:
            if label in label_2_id:
                tmp_vec[label_2_id[label]] = 1
        else:
            for _ in label:
                if _ in label_2_id:
                    tmp_vec[label_2_id[_]] = 1

        label_vecs.append(tmp_vec)

    return label_vecs


if __name__ == '__main__':
    y_true = [1, 2, 3]
    y_pred = [1, 2, 2]
    metrics, report = metrics_report(y_true, y_pred)
    print(metrics)
    print(report)
