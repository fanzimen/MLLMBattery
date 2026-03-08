#!/bin/bash
# filepath: code/scripts/train_battery_soh.sh

# BatteryGPT 训练脚本 - 多数据集循环训练 + 自动多轮评估

set -e  # 遇到错误时停止


# ============ 配置区 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 消除警告
export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS=4

# SwanLab 配置
export SWANLAB_MODE=cloud

# 基础路径配置
CONFIG_FILE="config/batterygpt.yaml"
NUM_GPUS=4
DATA_BASE_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data"
OUTPUT_BASE_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt"
RESUME_CKPT="" 
# 定义多个训练集和测试集组合
FINETUNE_DATASETS=(
    "train_XJTU4_S:test_XJTU4_S"
    "train_XJTU6_S:test_XJTU6_S"
    "train_TJU_NCM_S:test_TJU_NCM_S"
    "train_XJTU3_S:test_XJTU3_S"
    # "train_10:test_10"
    # "train_mix:test_mix"
)

# 评估配置
EVAL_DEVICE="cuda:0"
EVAL_BATCH_SIZE=64
EVAL_STRIDE=1
EVAL_EPOCH_STEP=5
EVAL_NUM_GPUS=4 # 每隔多少个 Epoch 评估一次

# ============ 训练循环 ============
TOTAL_DATASETS=${#FINETUNE_DATASETS[@]}
CURRENT=0

echo "========================================"
echo "🚀 BatteryGPT 多数据集训练流程"
echo "   共 ${TOTAL_DATASETS} 个数据集"
echo "========================================"
echo ""

for dataset_pair in "${FINETUNE_DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # 解析训练集和测试集名称
    IFS=':' read -r train_name test_name <<< "$dataset_pair"
    
    echo ""
    echo "========================================"
    echo "📦 [$CURRENT/$TOTAL_DATASETS] 数据集: ${train_name}"
    echo "========================================"
    
    # 动态设置 Epochs
    if [[ "$train_name" == *"XJTU3"* ]]; then
        MAX_EPOCHS=50
        echo "⚙️  Dataset match XJTU3 -> Setting Epochs = 50"
    else
        MAX_EPOCHS=15
        echo "⚙️  Standard Dataset -> Setting Epochs = 15"
    fi
    
    train_folder="${DATA_BASE_DIR}/${train_name}"
    test_folder="${DATA_BASE_DIR}/${test_name}"
    
    # 创建实验名称
    exp_name="${train_name}_$(date +%Y%m%d_%H%M%S)"
    exp_output_dir="${OUTPUT_BASE_DIR}/run_${exp_name}"
    
    # ========== 训练阶段 ==========
    TRAIN_START=$(date +%s)
    
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=$((29500 + CURRENT)) \
        train_battery_soh.py \
        --config ${CONFIG_FILE} \
        --train_data_folder "${train_folder}" \
        --test_data_folder "${test_folder}" \
        --exp_name "${exp_name}" \
        --epochs ${MAX_EPOCHS} \
        ${RESUME_CKPT:+--resume_checkpoint "${RESUME_CKPT}"}
        # --keep_best_only \
        # --no_swanlab
    
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    echo "✅ 训练完成，耗时: ${TRAIN_DURATION} 秒"

    # ========== 验证阶段 (Multi-Epoch Evaluation) ==========
    echo ""
    echo "🔍 开始多轮评估 (Step=${EVAL_EPOCH_STEP}, Stride=${EVAL_STRIDE})..."
    
    # 查找所有 epoch 模型文件
    CHECKPOINT_FILES=($(find "${exp_output_dir}" -name "model_epoch_*.pt" | sort -V))
    
    count=0
    for CHECKPOINT_PATH in "${CHECKPOINT_FILES[@]}"; do
        # 提取 epoch 编号
        BASENAME=$(basename "${CHECKPOINT_PATH}")
        EPOCH_NUM=$(echo "${BASENAME}" | grep -oP 'epoch_\K\d+')
        
        # 1. 检查是否符合步长要求 (取模为0 或 等于最大Epoch)
        if (( EPOCH_NUM % EVAL_EPOCH_STEP != 0 )) && (( EPOCH_NUM != MAX_EPOCHS )); then
            continue
        fi
        
        count=$((count + 1))
        echo "   📊 [$count] 评估 Epoch ${EPOCH_NUM}: ${BASENAME}"
        
        # 设置输出目录
        EVAL_OUTPUT_DIR="${exp_output_dir}/evaluation_epoch_${EPOCH_NUM}"
        
        # 执行评估
        torchrun \
            --nproc_per_node=${EVAL_NUM_GPUS} \
            --master_port=$(( 29700 + CURRENT * 100 + EVAL_COUNT )) \
            evaluate_batterygpt.py \
            --config "${CONFIG_FILE}" \
            --checkpoint "${CHECKPOINT_PATH}" \
            --device "${EVAL_DEVICE}" \
            --output_dir "${EVAL_OUTPUT_DIR}" \
            --train_data_folder "${train_folder}" \
            --test_data_folder "${test_folder}" \
            --text_gen_batch_size ${EVAL_BATCH_SIZE} \
            --compute_uncertainty \
            --uncertainty_alpha 0.05 \
            --uncertainty_beta  1.0 \
    
    echo "✅ 数据集 ${train_name} 所有流程完成"

done
done