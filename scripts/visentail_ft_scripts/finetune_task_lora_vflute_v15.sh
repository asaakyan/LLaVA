#!/bin/bash

export WANDB_PROJECT=vflute-v2
export TRANSFORMERS_CACHE=/mnt/swordfish-pool2/models/transformers_cache
export MODEL=llava-v1.5-7b
export VISENTAIL=/mnt/swordfish-pool2/asaakyan/visEntail
export DATA_DIR=../../../data/flute-v-llava-clean

deepspeed ../../llava/train/train_mem.py \
    --seed 42 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ../zero3.json \
    --model_name_or_path $TRANSFORMERS_CACHE/$MODEL \
    --version v1 \
    --data_path $DATA_DIR/vflute-v2-train.json \
    --eval_data_path $DATA_DIR/vflute-v2-valid.json \
    --output_dir $VISENTAIL/checkpoints/$MODEL-vflute-v2-lora \
    --group_by_modality_length False \
    --image_folder $VISENTAIL/data/v-flute \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
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
