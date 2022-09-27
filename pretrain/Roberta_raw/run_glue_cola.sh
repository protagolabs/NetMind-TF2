#!/bin/sh

# batch sizes: 8, 16, 32, 64, 128
# learning rates: 3e-4, 1e-4, 5e-5, 3e-5

for bs in 128 64 32 16 8; do for lr in 3e-4 1e-4 5e-5 3e-5 1e-5 5e-6 3e-6 1e-6
    do
    CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path="roberta_saved_model_ep5/" \
    --tokenizer_name="roberta-base" \
    --task_name="cola" \
    --per_device_train_batch_size=$bs \
    --learning_rate=$lr \
    --num_train_epochs=4 \
    --do_train \
    --do_eval \
    --max_seq_length=512 \
    --overwrite_output_dir \
    --output_dir="./cola_saved_model"
    done; done

