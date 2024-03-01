import os
import json
import openai
import argparse
import numpy as np

from random import sample
from collections import OrderedDict
from utils import eval_score, set_seed


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_res_path", default="./SLM_results/ner_fewnerd_FSLS_roberta-large/5-shot/test_result_idx0.json", type=str)
    parser.add_argument("--demo_path", default="./prompt/demo_ner_fewnerd.json", type=str)
    parser.add_argument("--output_path", default="./outputs/ner_fewnerd_FSLS_roberta-large/5-shot/reranked_result_idx0", type=str)
    parser.add_argument("--dataset_name", default="NER_FewNERD", type=str, choices=['NER_FewNERD', 'RE_TACREV', 'ED_ACE'])
    
    parser.add_argument("--model_name", default="gpt-4", type=str)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--repeat_time", default=1, type=int)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument('--is_shuffle', action='store_true', default=False, help="whether to shuffle the order of candidate label list in the multi-choice question. If not, the order will be determined by their confidence scores.")
    parser.add_argument('--demo_given', action='store_true', default=False, help="whether to provide demonstratons or not.")
    parser.add_argument("--demo_num", default=4, type=int)
    parser.add_argument('--explanation_given', action='store_true', default=False, help="CoT or direct answer")
    
    parser.add_argument('--auto_th', action='store_true', default=False, help="whether to determine the confidence threshold by cross-validating the dev set. If not, the reranked interval will be set as (th_lb, th_ub)")
    parser.add_argument("--th_lb", default=0.0, type=float, help="the fixed lower bound of the confidence score for reranked samples. Disabled if auto_th is True")
    parser.add_argument("--th_ub", default=0.6, type=float, help="the fixed upper bound of the confidence score for reranked samples. Disabled if auto_th is True")
    parser.add_argument("--dev_res_path", default="./SLM_results/ner_fewnerd_FSLS_roberta-large/5-shot/dev_result_idx0.json", type=str, help="samples for cross validation. Disabled if auto_th is False")
    parser.add_argument("--dev_th_ub", default=0.7, type=float, help="the start of threshold's grid search. Disabled if auto_th is False")
    parser.add_argument("--dev_th_lb", default=0.3, type=float, help="the end of threshold's grid search. Disabled if auto_th is False")
    parser.add_argument("--search_step", default=0.02, type=float)
    args = parser.parse_args()
    
    args.task, args.dataset = args.dataset_name.split("_")
    if args.task=="RE":
        from chat_re_reranker import rerank, get_res_list
    else:
        from chat_ner_reranker import rerank, get_res_list
    set_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    res_dict = OrderedDict({
        "rerank_prf": None,
        "original_prf": None,
        "model_name": args.model_name,
        "topk": args.topk,
        "auto_th": args.auto_th,
        "demo_given": args.demo_given,
        "demo_num": args.demo_num,
        "explanation_given": args.explanation_given,
        "th": args.th_ub,
        "dev_th": args.dev_th_ub,
    })
    
    if args.demo_given:
        with open(args.demo_path) as f:
            demo_candidate_list = json.load(f)
        if args.demo_num<len(demo_candidate_list):
            demo_candidate_list = sample(demo_candidate_list, args.demo_num)
    else:
        demo_candidate_list = list()
            
    res_list = get_res_list(args.old_res_path)
    if args.auto_th:
        dev_res_list = get_res_list(args.demo_path)
        best_th, dev_best_f1 = 0.0, 0.0
        
        for i, dev_th in enumerate(np.arange(args.dev_th_lb, args.dev_th_ub+args.search_step, step=args.search_step)):     
            if i==0:
                dev_rerank_res_list = [res for res in dev_res_list if "max_prob" in res and res["max_prob"]<dev_th]
                dev_rerank_id_list = [i for i, res in enumerate(dev_res_list) if "max_prob" in res and res["max_prob"]<dev_th]
            else:
                dev_rerank_res_list = [res for res in dev_res_list if "max_prob" in res and res["max_prob"]<dev_th and res["max_prob"]>(dev_th-args.search_step)]
                dev_rerank_id_list = [i for i, res in enumerate(dev_res_list) if "max_prob" in res and res["max_prob"]<dev_th and res["max_prob"]>(dev_th-args.search_step)]
            rerank(
                args, 
                res_list = dev_res_list,
                rerank_res_list=dev_rerank_res_list, 
                rerank_id_list=dev_rerank_id_list,
                demo_candidate_list=demo_candidate_list,
                explanation_given=args.explanation_given,
            )
            _, _, dev_f1 = eval_score(gt_list=[res["gt"] for res in dev_res_list], pred_list=[res["pred"] for res in dev_res_list])
            print(dev_th, dev_f1)
            if dev_f1 > dev_best_f1:
                dev_best_f1 = dev_f1
                best_th = dev_th
        res_dict["best_th"] = best_th; print(best_th)
        args.th_lb, args.th_ub = 0, best_th

    rerank_res_list = [res for res in res_list if "max_prob" in res and res["max_prob"]<args.th_ub and res["max_prob"]>args.th_lb]
    rerank_id_list = [i for i, res in enumerate(res_list) if "max_prob" in res and res["max_prob"]<args.th_ub and res["max_prob"]>args.th_lb]
    
    res_dict["overall_num"] = len(res_list)
    print("Overall sample number: {}".format(res_dict["overall_num"]))
    res_dict["rerank_num"] = len(rerank_res_list)
    print("Difficult sample number: {}".format(res_dict["rerank_num"]))
    res_dict["rerank_hit_num"] = len([res for res in rerank_res_list if res['gt'] in res['candidate_label_list'][:args.topk] or res['gt']=="None"])
    print("Difficult and Hit sample (SLM's top-{} predictions including correct answer) number: {}".format(args.topk, res_dict["rerank_hit_num"]))
    
    res_dict["original_prf"] = eval_score(gt_list=[res["gt"] for res in res_list], pred_list=[res["pred"] for res in res_list])
    print("Overall precision, recall and F1 (before reranking): {}".format(res_dict["original_prf"]))
    res_dict["original_sub_prf"] = eval_score(
        gt_list=[res["gt"] for res in rerank_res_list],
        pred_list=[res["pred"] for res in rerank_res_list],
    )
    print("Difficult subset precision, recall and F1 (before reranking): {}".format(res_dict["original_sub_prf"]))
 
    rerank_gt_list, rerank_pred_list, rerank_output_list = rerank(
        args, 
        res_list = res_list,
        rerank_res_list=rerank_res_list, 
        rerank_id_list=rerank_id_list, 
        demo_candidate_list=demo_candidate_list,
        explanation_given=args.explanation_given
    )   
    
    res_dict["rerank_prf"] = eval_score(gt_list=[res["gt"] for res in res_list], pred_list=[res["pred"] for res in res_list])
    print("Overall precision, recall and F1 (after reranking): {}".format(res_dict["rerank_prf"]))
    res_dict["rerank_sub_prf"] = eval_score(rerank_gt_list, rerank_pred_list)
    print("Difficult subset precision, recall and F1 (after reranking): {}".format(res_dict["rerank_sub_prf"]))
    with open(os.path.join(args.output_path, "rerank_result.json"), 'w') as f:
        json.dump(rerank_output_list, f)
    with open(os.path.join(args.output_path, "metric.json"), 'w') as f:
        json.dump(res_dict, f)