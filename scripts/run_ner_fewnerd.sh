export SLM_RES_PATH="./SLM_results/ner_fewnerd_FSLS_roberta-large/5-shot/test_result_idx0.json"
export OUT_PATH="./outputs/ner_fewnerd_FSLS_roberta-large/5-shot/reranked_result_idx0"

python run_gpt4_reranker.py \
    --dataset_name NER_FewNERD \
    --old_res_path $SLM_RES_PATH \
    --demo_path ./prompt/demo_ner_fewnerd.json \
    --output_path $OUT_PATH \
    --model_name gpt-4 \
    --batch_size 5 \
    --temperature 0.0 \
    --repeat_time 1 \
    --topk 3 \
    --demo_given \
    --demo_num 4 \
    --explanation_given \
    --th_ub 0.6