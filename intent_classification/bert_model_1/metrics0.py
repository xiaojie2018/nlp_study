# -*- coding: UTF-8 -*-
# !/usr/bin/python


from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import hinge_loss, hamming_loss, log_loss, label_ranking_loss, brier_score_loss, zero_one_loss
from sklearn.preprocessing import OneHotEncoder
from tensorboard import summary as summary_lib
import numpy as np


zero_bit = 1e-9


def best_threshold(fpr,tpr,thre):
    """
    :param fpr: np.array
    :param tpr: np.array
    :param thre: np.array
    :return:
    """
    max_value_ = 0
    best_thre_ = 0
    value_ = tpr-fpr
    for i,v in enumerate(value_):
        if v > max_value_:
            max_value_ = v
            best_thre_ = thre[i]
    return best_thre_


def pr_curve(label_list, prob_list, class_number):
    truth = np.array(label_list).reshape(-1,1)

    lb = OneHotEncoder(n_values=class_number, sparse=False)
    lb.fit(truth)

    labels = lb.transform(truth)

    summary_proto = summary_lib.pr_curve_pb(
        name='PR_Curve',
        predictions=np.array(prob_list),
        labels=labels)
    return summary_proto


def multiclass_roc_auc_score(truth, prob, average, class_number):
    truth = np.array(truth).reshape(-1,1)
    prob = np.array(prob)

    lb = OneHotEncoder(n_values=class_number, sparse=False)
    lb.fit(truth)

    truth = lb.transform(truth)

    return roc_auc_score(truth, prob, average=average)


def get_all_metrics(label_list, prediction_list, prob_list, class_number):
    '''
    :param label_list:
    :param prediction_list:
    :return:

    '''
    label_dict = {i: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for i in sorted(set(label_list))}
    wrong_indexs_all = []
    for index in range(len(label_list)):
        label = label_list[index]
        predict = prediction_list[index]
        if label == predict:
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['tp'] += 1
                else:
                    label_dict[k]['tn'] += 1
        else:
            wrong_indexs_all.append(index)
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['fn'] += 1
                elif k == predict:
                    label_dict[k]['fp'] += 1
                else:
                    label_dict[k]['tn'] += 1
    metrics = {}
    FPR, TPR, THRES = {}, {}, {}
    all_data = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    mean_precision, mean_recall, macro_f1, mean_fpr = 0, 0, 0, 0
    for k, v in label_dict.items():
        all_data = {'tp': all_data['tp'] + v['tp'],
                    'fp': all_data['fp'] + v['fp'],
                    'tn': all_data['tn'] + v['tn'],
                    'fn': all_data['fn'] + v['fn'],
                    }
        precision, recall, f1 = get_metrics(v)

        FPR[k], TPR[k], THRES[k] = roc_curve(
            (np.array(label_list) == k).astype(int),
            np.array(prob_list)[:, k])
        metrics[k] = {'precision': precision,
                      'recall': recall,
                      'f1': f1,
                      'auc': auc(FPR[k], TPR[k]),
                      'best_threshold': best_threshold(FPR[k], TPR[k], THRES[k])}

        mean_precision += precision
        mean_recall += recall
        macro_f1 += f1

    mean_precision = round(mean_precision / len(metrics), 9)
    mean_recall = round(mean_recall / len(metrics), 9)
    macro_f1 = round(macro_f1 / len(metrics), 9)
    sum_precision, sum_recall, micro_f1 = get_metrics(all_data)
    metrics['mean'] = {'mean_precision': mean_precision,
                       'mean_recall': mean_recall,
                       'macro_f1': macro_f1,
                       'macro_auc': multiclass_roc_auc_score(label_list, prob_list, "macro",class_number)}
    metrics['sum'] = {'sum_precision': sum_precision,
                      'sum_recall': sum_recall,
                      'micro_f1': micro_f1,
                      'micro_auc': multiclass_roc_auc_score(label_list, prob_list, 'micro',class_number)}

    conf_mat = confusion_matrix(label_list, prediction_list)
    metrics['confusion_matrix'] = {'confusion_matrix': {i:conf_mat[i,:].tolist() for i in range(conf_mat.shape[0])}}
    return metrics, wrong_indexs_all


def get_metrics(data):
    precision = data['tp'] / (data['tp'] + data['fp'] + 0.000000001)
    recall = data['tp'] / (data['tp'] + data['fn'] + 0.000000001)
    f1 = (2 * precision * recall) / (precision + recall + 0.000000001)
    return round(precision, 9), round(recall, 9), round(f1, 9)


def get_all_metrics_two(label_list, prediction_list):
    min_unit = 1e-9
    round_bit = 9
    def get_metrics(data):
        # precision, recall, f_score, true_sum = precision_recall_fscore_support(index_y, index_pred, labels=None,
        #                                                                        pos_label=1, average=None,
        #                                                                        warn_for=(
        #                                                                        'precision', 'recall', 'f-score'),
        #                                                                        sample_weight=None)
        # return round(precision, round_bit), round(recall, round_bit), round(f1, round_bit)
        precision = data['tp'] / (data['tp'] + data['fp'] + min_unit)
        recall = data['tp'] / (data['tp'] + data['fn'] + min_unit)
        f1 = (2 * precision * recall) / (precision + recall + min_unit)
        return round(precision, round_bit), round(recall, round_bit), round(f1, round_bit)

    label_dict = {i: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for i in sorted(set(label_list))}
    wrong_indexs_all = []
    for index in range(len(label_list)):
        label = label_list[index]
        predict = prediction_list[index]
        if label == predict:
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['tp'] += 1
                else:
                    label_dict[k]['tn'] += 1
        else:
            wrong_indexs_all.append(index)
            for k, v in label_dict.items():
                if k == label:
                    label_dict[label]['fn'] += 1
                elif k == predict:
                    label_dict[k]['fp'] += 1
                else:
                    label_dict[k]['tn'] += 1
    metrics = {}
    all_data = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    mean_precision, mean_recall, macro_f1 = 0, 0, 0
    for k, v in label_dict.items():
        all_data = {'tp': all_data['tp'] + v['tp'],
                    'fp': all_data['fp'] + v['fp'],
                    'tn': all_data['tn'] + v['tn'],
                    'fn': all_data['fp'] + v['fn'],
                    }
        precision, recall, f1 = get_metrics(v)
        metrics[k] = {'precision': precision,
                      'recall': recall,
                      'f1': f1}
        mean_precision += precision
        mean_recall += recall
        macro_f1 += f1

    mean_precision = round(mean_precision / len(metrics), round_bit)
    mean_recall = round(mean_recall / len(metrics), round_bit)
    macro_f1 = round(macro_f1 / len(metrics), round_bit)
    sum_precision, sum_recall, micro_f1 = get_metrics(all_data)
    metrics['mean'] = {'mean_precision': mean_precision,
                       'mean_recall': mean_recall,
                       'macro_f1': macro_f1}
    metrics['sum'] = {'sum_precision': sum_precision,
                      'sum_recall': sum_recall,
                      'micro_f1': micro_f1}
    return metrics, wrong_indexs_all


def get_multi_label_from_threshold(probs, threshold=0.5):
    """
        多标签分类, 将预测值转为1, 0
    :param probs: numpy, logitis进行sigmoid后的值
    :param threshold: float, 阀值
    :return: 
    """

    pred_multi = np.zeros_like(probs)  # 全零numpy
    # 全局设大于阀值的元素为1
    pred_multi[probs >= threshold] = 1
    enum_max = pred_multi.max()
    if enum_max != 1:  # 如果没有大于阀值的元素,则取最大的一个元素设置为类标
        # np.argmax(probs, axis=1),这个是获取每一行的最大值
        # 下面函数的意思是将每行最大元素置为1
        pred_multi[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
    return pred_multi


def cosine_distance(v1, v2): # 余弦距离
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if v1.all() and v2.all():
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0


def euclidean_distance(v1, v2):  # 欧氏距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def manhattan_distance(v1, v2):  # 曼哈顿距离
    return np.sum(np.abs(v1 - v2))


def chebyshev_distance(v1, v2):  # 切比雪夫距离
    return np.max(np.abs(v1 - v2))


def minkowski_distance(v1, v2):  # 闵可夫斯基距离
    return np.sqrt(np.sum(np.square(v1 - v2)))


def euclidean_distance_standardized(v1, v2):  # 标准化欧氏距离
    v1_v2 = np.vstack([v1, v2])
    sk_v1_v2 = np.var(v1_v2, axis=0, ddof=1)
    return np.sqrt(((v1 - v2) ** 2 / (sk_v1_v2 + zero_bit * np.ones_like(sk_v1_v2))).sum())


def mahalanobis_distance(v1, v2):  # 马氏距离
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([v1, v2])
    XT = X.T

    # 方法一：根据公式求解
    S = np.cov(X)  # 两个维度之间协方差矩阵
    try:
        SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵  todo
    except:
        SI = np.zeros_like(S)
    # 马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    distance_all = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            distance_1 = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            distance_all.append(distance_1)
    return np.sum(np.abs(distance_all))


def bray_curtis_distance(v1, v2):  # 布雷柯蒂斯距离, 生物学生态距离
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    up_v1_v2 = np.sum(np.abs(v2 - v1))
    down_v1_v2 = np.sum(v1) + np.sum(v2)
    return up_v1_v2 / (down_v1_v2 + zero_bit)


def pearson_distance(v1, v2):  # 皮尔逊相关系数（Pearson correlation）
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1_v2 = np.vstack([v1, v2])
    return np.corrcoef(v1_v2)[0][1]

