# -*- coding: UTF-8 -*-
# !/usr/bin/python


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


if __name__ == '__main__':
    label_list = [1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 4]
    prediction_list = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4]
    metrics, wrong_indexs_all = get_all_metrics_two(label_list, prediction_list)
