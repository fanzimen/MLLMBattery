export CUDA_VISIBLE_DEVICES=0,1,2,3

# 例如格式：2025-09-26_15-58-30
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
#--lock-text \
    # --precision amp \

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 src/open_clip_train/main.py \
    --model "btr-2048-vicuna7b" \
    --pretrained_text "Vicuna-7B" \
    --precision bf16 \
    --train-data "../data_preprocess/soh_data/train_10" \
    --val-data "../data_preprocess/soh_data/test_10" \
    --description-root "../data_preprocess/full_descriptions_withmoreinfo_decimal3" \
    --dataset-type timeseries \
    --batch-size 256 \
    --epochs 150 \
    --lr 1e-4 \
    --lr-scheduler const-cooldown \
    --epochs-cooldown 20 \
    --lr-cooldown-power 1.0 \
    --lr-cooldown-end 1e-6 \
    --name "battery_tllm_btr2048_vicuna7b_${NOW}" \
    --report-to wandb \
    --accum-freq 1 \
    --warmup 1000 \
    --grad-clip-norm 1.0 \
    --workers 56 \
    --force-timeseries-seq-len 40 \
    --window_size 40 \
    --force_context_length 110 \
    --lock-text \
    --grad-checkpointing \

