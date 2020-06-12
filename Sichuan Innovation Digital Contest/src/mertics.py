# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/6/9 10:04
# @author  : Mo
# @function:


from typing import Dict, List, Any
from collections import Counter


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


if __name__ == '__main__':
    y_true = [1, 2, 3]
    y_pred = [1, 2, 2]
    metrics, report = metrics_report(y_true, y_pred)
    print(metrics)
    print(report)
