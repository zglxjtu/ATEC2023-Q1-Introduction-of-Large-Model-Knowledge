LR=4e-3
TRAIN_FILE=./data/dev_new.jsonl
MODEL_PATH=./chatglm2-6b
OUTPUT=./output/

python src/get_train_qa.py
python src/get_rag_data.py
echo "get feature done!"

torchrun --standalone --nnodes=1 --nproc-per-node=1 src/main.py \
    --do_train \
    --train_file $TRAIN_FILE \
    --preprocessing_num_workers 10 \
    --prompt_column question \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --max_source_length 768 \
    --max_target_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 2400 \
    --logging_steps 100 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len 128
