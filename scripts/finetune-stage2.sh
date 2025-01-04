#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570
N_EPOCH=2


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT v2xumllm/train/train_mem.py \
    --deepspeed ./scripts/config.json \
    --lora_enable True \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/train_val_videoxum_f.json \
    --feat_folder /p61/ytang37/data/anet_cap/v_feat \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/v2xumllm-$MODEL_VERSION-stage2 \
    --bf16 True \
    --num_train_epochs $N_EPOCH \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
