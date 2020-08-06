import numpy as np
import pandas as pd

def transfer_label_vec_to_label_id(label_vec_df, enclude_neg_label):
    label_ids = []
    if enclude_neg_label:
        label_vec_values = label_vec_df.values
        for j in range(len(label_vec_values)):
            tmp_label_vec = label_vec_values[j]
            if tmp_label_vec.sum():
                label_id = tmp_label_vec.argmax()
            else:
                label_id = -1
            label_ids.append(label_id)
    else:
        label_vec_values = label_vec_df.values
        for j in range(len(label_vec_values)):
            tmp_label_vec = label_vec_values[j]
            label_id = tmp_label_vec.argmax()
            label_ids.append(label_id)
    return label_ids


def transfer_label_vec_to_multilabel(label_vec_df):
    total_labels = []
    for info in label_vec_df.to_dict("records"):
        tmp_label = []
        for label, label_value in info.items():
            if label_value == 1:
                tmp_label.append(label)
        total_labels.append(",".join(tmp_label))
    return total_labels


def calculate_f1_p_r(pred_count, std_count, match_count):
    precision = match_count/pred_count if pred_count else 0
    recall = match_count/std_count if std_count else 0
    f1 = 2*match_count/(pred_count+std_count) if (pred_count + std_count) else 0
    return precision, recall, f1


def transfer_ner_results_to_token_labels(ner_infos):
    total_labels = []
    for ner_info in ner_infos:
        txt = ner_info['text']
        tmp_labels = ['0' for _ in txt]
        entities = ner_info['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            entity_type = entity['entity_type']
            for j in range(start_pos, end_pos):
                tmp_labels[j] = entity_type
        total_labels.extend(tmp_labels)
    return total_labels


def transfer_relation_infos_to_flatten(relation_infos):
    flatten_relation_infos = []
    for idx, sentence_info in enumerate(relation_infos):
        relations = sentence_info['relationships']
        for relation in relations:
            entity1 = relation['entity1']
            entity2 = relation['entity2']
            pos_info = "%s_%s_%s_%s_%s" % (idx, entity1['start_pos'], entity1['end_pos'], entity2['start_pos'], entity2['end_pos'])
            relation_type = relation['relation_type']
            flatten_relation_infos.append((pos_info, relation_type))
    return flatten_relation_infos


def transfer_event_infos_to_flatten(event_infos, event_argument_info):
    flatten_event_infos = {}
    for idx, event_info in enumerate(event_infos):
        events = event_info['events']
        for event in events:
            event_type = event['event_type']
            event_arguments = event['event_arguments']
            event_type_arguments = [_['name'] for _ in event_argument_info[event_type]]
            event_argument_dict = {event_argument['argument_role']: event_argument['word'] for event_argument in event_arguments}
            event_argument_dict.update({"idx": idx})
            for _ in event_type_arguments:
                if _ not in event_argument_dict:
                    event_argument_dict.update({_: None})
            if event_type in flatten_event_infos:
                flatten_event_infos[event_type].append(event_argument_dict)
            else:
                flatten_event_infos.update({event_type: [event_argument_dict]})
    return flatten_event_infos



if __name__ == '__main__':
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [9, 0, 7]})
    df2 = pd.DataFrame({'a': [1, 4], 'b': [4, 2], 'c':[0, 2]})
    df = pd.merge(df1, df2, how='inner', on=['a', 'b'])
    print(df)