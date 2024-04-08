#!/bin/bash

export WANDB_PROJECT=llava-tune-evil
export TRANSFORMERS_CACHE=/mnt/swordfish-pool2/models/transformers_cache
export MODEL=llava-v1.5-7b
export VISENTAIL=/mnt/swordfish-pool2/asaakyan/visEntail
export DATA_DIR=../../../data/evil-llava

while ps -p 3152439 > /dev/null; do
    sleep 300
done

deepspeed ../../llava/train/train_mem.py \
    --seed 42 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ../zero3.json \
    --model_name_or_path $TRANSFORMERS_CACHE/$MODEL \
    --version v1 \
    --data_path $DATA_DIR/train_data.json \
    --eval_data_path $DATA_DIR/valid_data.json \
    --output_dir $VISENTAIL/checkpoints/$MODEL-evil-lora \
    --group_by_modality_length False \
    --image_folder $VISENTAIL/data/evil/flickr30k_images/flickr30k_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1293 \
    --save_strategy "steps" \
    --save_steps 1293 \
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
