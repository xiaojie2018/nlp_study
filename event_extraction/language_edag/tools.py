# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/23 10:47
# software: PyCharm

from dee_metric import measure_event_table_filling
from helper import default_dump_json
from utils import Feature


def measure_dee_prediction(event_type_fields_pairs, features, event_decode_results, dump_json_path=None):
    pred_record_mat_list = []
    gold_record_mat_list = []
    for term in event_decode_results:
        ex_idx, pred_event_type_labels, pred_record_mat = term[:3]
        pred_record_mat = [
            [
                [
                    tuple(arg_tup) if arg_tup is not None else None
                    for arg_tup in pred_record
                ] for pred_record in pred_records
            ] if pred_records is not None else None
            for pred_records in pred_record_mat
        ]
        doc_fea = features[ex_idx]
        assert isinstance(doc_fea, Feature)
        gold_record_mat = [
            [
                [
                    tuple(doc_fea.span_token_ids_list[arg_idx]) if arg_idx is not None else None
                    for arg_idx in event_arg_idxs
                ] for event_arg_idxs in event_arg_idxs_objs
            ] if event_arg_idxs_objs is not None else None
            for event_arg_idxs_objs in doc_fea.event_arg_idxs_objs_list
        ]

        pred_record_mat_list.append(pred_record_mat)
        gold_record_mat_list.append(gold_record_mat)

    g_eval_res = measure_event_table_filling(
        pred_record_mat_list, gold_record_mat_list, event_type_fields_pairs, dict_return=True
    )

    if dump_json_path is not None:
        default_dump_json(g_eval_res, dump_json_path)

    return g_eval_res
