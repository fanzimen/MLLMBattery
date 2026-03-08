CUDA_VISIBLE_DEVICES=0,1

# 例如格式：2025-09-26_15-58-30
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
#--lock-text \
    # --precision amp \
#    
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29501 src/open_clip_train/main.py \
    --model aaabatterytest \
    --train-data "../data_preprocess/datasoh/MIT/2017-05-12/2017-05-12_battery-1.csv" \
    --val-data "../data_preprocess/datasoh/MIT/2017-05-12/2017-05-12_battery-2.csv" \
    --train-description-data "../data_preprocess/full_descriptions_simple/MIT/2017-05-12/2017-05-12_battery-1.csv" \
    --val-description-data "../data_preprocess/full_descriptions_simple/MIT/2017-05-12/2017-05-12_battery-2.csv" \
    --dataset-type timeseries \
    --batch-size 32 \
    --epochs 80 \
    --lr 1e-4 \
    --name "battery_run_${NOW}" \
    --report-to wandb \
    --accum-freq 1 \
    --warmup 1000 \
    --grad-clip-norm 1.0 \
    --workers 8 \
    --force-timeseries-seq-len 40 \
    --window_size 40 \
