export SLM_RES_PATH="TO BE FILLED"
export OUT_PATH="TO BE FILLED"

python run_gpt4_reranker.py \
    --dataset_name ED_ACE \
    --old_res_path $SLM_RES_PATH \
    --demo_path ./prompt/demo_ed_ace.json \
    --output_path $OUT_PATH \
    --model_name gpt-4 \
    --batch_size 5 \
    --temperature 0.0 \
    --repeat_time 1 \
    --topk 3 \
    --demo_given \
    --demo_num 4 \
    --explanation_given \
    --th_ub 0.7