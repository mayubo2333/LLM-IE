import json
import asyncio

from tqdm import tqdm
from itertools import chain
from random import shuffle

from utils import parse_res, ensemble_pred_res, list_merge
from parameters import TEMPLATE_TACREV, ORD_A
from openai_wrapper import dispatch_openai_requests


NEW_INSENT_TOKEN = "<t>"
ORD_A = 97


def prompt_generation(res, topk=3, explanation_given=False, is_shuffle=False):
    for label in TEMPLATE_TACREV:
        if label not in res["candidate_label_list"]:
            res["candidate_label_list"].append(label)
        if len(res["candidate_label_list"])>=26:
            break
    
    if "explanation" in res:
        input = "\n\n".join(res["input"].split("\n\n")[:-1])
        if explanation_given:
            output = "Analysis: " + res["explanation"] + "\n" + res["input"].split("\n\n")[-1]
        else:
            output = res["input"].split("\n\n")[-1]
    else:   
        sentence = res["sent"]
        subj, obj = res["subj"], res["obj"]
        if is_shuffle:
            shuffle(res["candidate_label_list"])
            candidate_relations = res["candidate_label_list"]
        else:
            candidate_relations = res["candidate_label_list"][:topk]  
            if "None" not in candidate_relations:
                candidate_relations.append("None")  
        input = ""
        input += "Sentence: {}\n\n".format(sentence)
        input += "Candidate Choices: "
        for i, candidate_relation in enumerate(candidate_relations):
            template = TEMPLATE_TACREV[candidate_relation].replace("{subj}", subj).replace("{obj}", obj)
            input += "({}): {}\n".format(chr(ORD_A+i), template)
        output = None
    return input, output


def sample_demonstration(demo_candidate_list, explanation_given=False, is_shuffle=False):
    demo_prompt = [
        {
            "role": "system",
            "content": "Please answer the multi-choice questions below. Note that you need to strictly follow the format ending with 'Correct Answer: ([your choice])'"
        }
    ]
    for demo_sample in demo_candidate_list:
        input, output = prompt_generation(demo_sample, explanation_given=explanation_given, is_shuffle=is_shuffle)
        demo_prompt.extend([
            {
                "role": "user",
                "content": input,
            },
            {
                "role": "assistant",
                "content": output,
            }
        ])
    return demo_prompt



def get_res_list(res_path, filter_no_prob=False):
    with open(res_path) as f:
        res_list = json.load(f)
    return res_list


def rerank(
    args, 
    res_list,
    rerank_res_list,
    rerank_id_list,
    demo_candidate_list=None,
    explanation_given=False,
):
    raw_res_list, prompt_list = list(), list()
    repeated_rerank_res_list = list(chain(*[[example]*args.repeat_time for example in rerank_res_list]))
    for curr in tqdm(range(0, len(repeated_rerank_res_list), args.batch_size)):
        batch_rerank_list = repeated_rerank_res_list[curr:(curr+args.batch_size)]
        batch_demo_list = sample_demonstration(demo_candidate_list, explanation_given=explanation_given)
        batch_prompt = [batch_demo_list + [{"role": "user", "content": prompt_generation(res, topk=args.topk, is_shuffle=args.is_shuffle)[0]}] for res in batch_rerank_list] 
        prompt_list.extend(batch_prompt)
        response = asyncio.run(
            dispatch_openai_requests(
                model=args.model_name,
                messages_list=batch_prompt,
                temperature=args.temperature,
            )
        )
        batch_response = [
            x['choices'][0]['message']['content'] if isinstance(x, dict) else "OpenAI Output Error" for x in response
        ]
        raw_res_list.extend(batch_response) 
    assert(len(raw_res_list)//args.repeat_time==len(rerank_res_list))
    merged_prompt_list = list_merge(prompt_list, args.repeat_time)
    raw_res_list, _ = ensemble_pred_res(raw_res_list, args.repeat_time, task="RE")
    
    rerank_gt_list, rerank_pred_list, output_list = list(), list(), list()
    for rerank_id, raw_res, res, prompt in zip(rerank_id_list, raw_res_list, rerank_res_list, merged_prompt_list):
        gt, old_pred, candidate_answer = res["gt"], res["pred"], res["candidate_label_list"]
        try:
            pred = parse_res(raw_res, candidate_answer, args.topk)
        except:
            pred = old_pred
        res_list[rerank_id]["pred"] = pred
        rerank_gt_list.append(gt)
        rerank_pred_list.append(pred)
        output = {
            "sent": res["sent"],
            "subj": res["subj"],
            "obj": res["obj"],
            "gt": gt,
            "old_pred": old_pred,
            "rerank_pred": pred,
            "prompt": prompt,
            "raw_res": raw_res,
        }
        output_list.append(output)
    
    return rerank_gt_list, rerank_pred_list, output_list