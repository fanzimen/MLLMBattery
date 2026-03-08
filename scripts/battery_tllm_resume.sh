
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 指定要恢复的实验名称
EXP_NAME="battery_qwen_1024_2026-01-12_15-11-23"  # 从日志中获取

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 src/open_clip_train/main.py \
    --model btr-1024-qwen2.5 \
    --pretrained_text "Qwen2.5" \
    --precision bf16 \
    --train-data "../data_preprocess/soh_data/train_10" \
    --val-data "../data_preprocess/soh_data/test_10" \
    --description-root "../data_preprocess/full_descriptions_withmoreinfo_decimal3" \
    --dataset-type timeseries \
    --batch-size 256 \
    --epochs 200 \
    --lr 1e-4 \
    --lr-scheduler const-cooldown \
    --epochs-cooldown 20 \
    --lr-cooldown-power 1.0 \
    --lr-cooldown-end 1e-6 \
    --name "${EXP_NAME}" \
    --report-to wandb \
    --accum-freq 1 \
    --warmup 10 \
    --grad-clip-norm 1.0 \
    --workers 56 \
    --force-timeseries-seq-len 40 \
    --window_size 40 \
    --force_context_length 100 \
    --lock-text \
    --resume latest          # 自动从最新checkpoint恢复