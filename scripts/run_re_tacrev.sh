export SLM_RES_PATH="TO BE FILLED"
export OUT_PATH="TO BE FILLED"

python run_reranker.py \
    --dataset_name RE_TACREV \
    --old_res_path $SLM_RES_PATH \
    --demo_path ./prompt/demo_re_tacrev.json \
    --output_path $OUT_PATH \
    --model_name $MODEL_NAME \
    --batch_size 4 \
    --temperature 0.0 \
    --repeat_time 1 \
    --topk 3 \
    --demo_given \
    --demo_num 4 \
    --explanation_given \
    --th_ub 0.7