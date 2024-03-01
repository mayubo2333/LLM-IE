import re
import torch
import numpy as np
from random import seed
from collections import Counter, defaultdict

from parameters import ORD_A


def set_seed(args):
    if isinstance(args, int):
        seed(args)
        np.random.seed(args)
        torch.manual_seed(args)
    else:
        seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


def eval_score(gt_list, pred_list):
    gt_num, pred_num, correct_num = 0, 0, 0
    for gt, pred in zip(gt_list, pred_list):
        if isinstance(gt, list):
            gt, pred = set(gt), set(pred)
            gt_num += len(gt)
            pred_num += len(pred)
            correct_num += len(gt.intersection(pred))
        else:
            if gt!="None":
                gt_num += 1
            if pred!="None":
                pred_num += 1
                if gt==pred:
                    correct_num += 1
            
    precision = correct_num/pred_num if pred_num else 0
    recall = correct_num/gt_num if gt_num else 0 
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else 0
    return precision, recall, f1


def ensemble_pred_res(pred_list, repeat_time, occur_th=0, task='ED'):
    ensemble_pred_list, ensemble_counter_list = list(), list()
    pred_list_list = list_merge(pred_list, repeat_time)
    
    for pred_list_this_example in pred_list_list:
        ensemble_pred = list()
        if task in ['ED', 'NER']:       
            span_counter = defaultdict(list) 
            for pred_list_one_trace in pred_list_this_example:
                if pred_list_one_trace=="failed":
                    continue
                for label, span in pred_list_one_trace:
                    span_counter[span].append(label)
            for span, labels in span_counter.items():
                voted_label, occur_time = Counter(labels).most_common(1)[0]
                if occur_time>occur_th:
                    ensemble_pred.append((voted_label, span))
            ensemble_pred_list.append(ensemble_pred)
            ensemble_counter_list.append(
                {span:Counter(labels).most_common() for span, labels in span_counter.items()}
            )
        elif task=="EE":
            span_counter = defaultdict(list) 
            for pred_list_one_trace in pred_list_this_example:
                if pred_list_one_trace=="failed":
                    continue
                for event, trigger, label, span in pred_list_one_trace:
                    key = "___|___".join([event, trigger, span])
                    span_counter[key].append(label)
            for key, labels in span_counter.items():
                event, trigger, span = key.split("___|___")
                voted_label, occur_time = Counter(labels).most_common(1)[0]
                if occur_time>occur_th:
                    ensemble_pred.append((event, trigger, voted_label, span))
            ensemble_pred_list.append(ensemble_pred)
            ensemble_counter_list.append(
                {span:Counter(labels).most_common() for key, labels in span_counter.items()}
            )
        else:
            relation_counter = Counter([pred for pred in pred_list_this_example if pred!="failed"]).most_common()
            if not relation_counter:
                voted_label = "None"
            else:
                voted_label, occur_time = relation_counter[0]
                if occur_time<=occur_th:
                    voted_label = "None"
            ensemble_pred_list.append(voted_label)
            ensemble_counter_list.append(relation_counter)       

    return ensemble_pred_list, ensemble_counter_list


def list_merge(input_list, repeat_time):
    merged_list = list()
    assert(len(input_list)%repeat_time==0)
    for idx, input in enumerate(input_list):
        pred_idx, repeat_idx = idx//repeat_time, idx%repeat_time
        if repeat_idx==0:
            merged_list.append(list())
        merged_list[pred_idx].append(input)
    return merged_list


def parse_res(raw_res, candidate_answer, topk):
    choice = re.search(
        "Correct Answer:\s?\([a-zA-Z]\)", raw_res
    ).group().lstrip("Correct Answer:").strip().lstrip('(').rstrip(')').lower()
    if ord(choice)-ORD_A==topk:
        pred = "None"
    else:
        pred = candidate_answer[ord(choice)-ORD_A]
    return pred