#!/bin/bash

export WANDB_PROJECT=llava-tune
export TRANSFORMERS_CACHE=/mnt/swordfish-pool2/models/transformers_cache
export MODEL=$TRANSFORMERS_CACHE/llava-v1.6-mistral-7b
export VISENTAIL=/mnt/swordfish-pool2/asaakyan/visEntail
export DATA=$VISENTAIL/data/LLaVA-Instruct-150K/detail_23k.json

deepspeed ../../llava/train/train_mem.py \
    --deepspeed ../zero3.json \
    --model_name_or_path $MODEL \
    --version v1 \
    --data_path $DATA \
    --output_dir $VISENTAIL/checkpoints/$MODEL-test \
    --group_by_modality_length False \
    --image_folder $VISENTAIL/data/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
