#!/bin/bash
set -u

# ============ GPU 配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 用几块改这里
NUM_GPUS=4                             # 与上面数量一致

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS=4

CONFIG_FILE="config/batterygpt.yaml"
# CHECKPOINT_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt/run_train_XJTU3_S_20260204_175526"
# CHECKPOINT_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt/run_train_XJTU4_S_20260304_123941"
CHECKPOINT_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt/run_train_TJU_NCM_S_20260204_164729"
# CHECKPOINT_DIR="/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt/run_train_XJTU6_S_20260204_160042"
# 数据目录
# TRAIN_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/train_XJTU3_S"
# TRAIN_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/train_XJTU4_S"
# TEST_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/test_XJTU3_S"
# TEST_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/test_XJTU4_S"

TRAIN_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/train_TJU_NCM_S"
TEST_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/test_TJU_NCM_S"
# TRAIN_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/train_XJTU6_S"
# TEST_FOLDER="/mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data/test_XJTU6_S"

# 评估 epoch 过滤
EPOCH_START="10"
EPOCH_END="15"
EPOCH_STEP=5

# 推理参数
TEXT_GEN_BATCH_SIZE=64   # 每块GPU的batch size
STRIDE=1
DEBUG_FLAG=""

TOTAL_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "========================================"
echo "🔍 BatteryGPT 批量评估 (${NUM_GPUS} GPU 并行)"
echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "  RANGE:          [${EPOCH_START:-MIN}, ${EPOCH_END:-MAX}], step=${EPOCH_STEP}"
echo "========================================"

[ ! -d "${CHECKPOINT_DIR}" ] && echo "❌ CHECKPOINT_DIR 不存在" && exit 1
[ ! -f "${CONFIG_FILE}" ]    && echo "❌ CONFIG_FILE 不存在"    && exit 1
[ ! -d "${TRAIN_FOLDER}" ]   && echo "❌ TRAIN_FOLDER 不存在"   && exit 1
[ ! -d "${TEST_FOLDER}" ]    && echo "❌ TEST_FOLDER 不存在"    && exit 1

mapfile -t CHECKPOINT_FILES < <(find "${CHECKPOINT_DIR}" -name "model_epoch_*.pt" | sort -V)
[ ${#CHECKPOINT_FILES[@]} -eq 0 ] && echo "❌ 没有找到 checkpoint" && exit 1

for CHECKPOINT_PATH in "${CHECKPOINT_FILES[@]}"; do
    BASENAME=$(basename "${CHECKPOINT_PATH}")
    EPOCH_NUM=$(echo "${BASENAME}" | grep -oP 'epoch_\K\d+')
    [ -z "${EPOCH_NUM}" ] && continue

    [ -n "${EPOCH_START}" ] && [ "${EPOCH_NUM}" -lt "${EPOCH_START}" ] && continue
    [ -n "${EPOCH_END}"   ] && [ "${EPOCH_NUM}" -gt "${EPOCH_END}"   ] && continue

    BASE_START=${EPOCH_START:-0}
    if [ "${EPOCH_STEP}" -gt 1 ]; then
        DELTA=$(( EPOCH_NUM - BASE_START ))
        if [ $(( DELTA % EPOCH_STEP )) -ne 0 ]; then
            if [ -n "${EPOCH_END}" ] && [ "${EPOCH_NUM}" -ne "${EPOCH_END}" ]; then
                continue
            fi
        fi
    fi

    TOTAL_COUNT=$(( TOTAL_COUNT + 1 ))
    OUTPUT_DIR="${CHECKPOINT_DIR}/evaluation_epoch_${EPOCH_NUM}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "----------------------------------------"
    echo "📊 [${TOTAL_COUNT}] Epoch ${EPOCH_NUM} — ${NUM_GPUS} GPUs"
    echo "  ckpt: ${BASENAME}"
    echo "  out : ${OUTPUT_DIR}"
    echo "----------------------------------------"

    EVAL_START=$(date +%s)

    # ── torchrun 多GPU启动 ──
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=$(( 29600 + TOTAL_COUNT )) \
        evaluate_batterygpt.py \
            --config        "${CONFIG_FILE}" \
            --checkpoint    "${CHECKPOINT_PATH}" \
            --output_dir    "${OUTPUT_DIR}" \
            --train_data_folder "${TRAIN_FOLDER}" \
            --test_data_folder  "${TEST_FOLDER}" \
            --text_gen_batch_size "${TEXT_GEN_BATCH_SIZE}" \
            --stride        "${STRIDE}" \
            --compute_uncertainty \
            --uncertainty_alpha 0.01 \
            --uncertainty_beta  1.0 \
            ${DEBUG_FLAG}

    EVAL_STATUS=$?
    EVAL_DURATION=$(( $(date +%s) - EVAL_START ))

    if [ ${EVAL_STATUS} -eq 0 ]; then
        SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
        echo "✅ Epoch ${EPOCH_NUM} 完成，耗时 ${EVAL_DURATION}s"
    else
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        echo "❌ Epoch ${EPOCH_NUM} 失败，exit_code=${EVAL_STATUS}"
    fi
done

echo ""
echo "========================================"
echo "📌 总任务=${TOTAL_COUNT}  成功=${SUCCESS_COUNT}  失败=${FAIL_COUNT}"
echo "========================================"
[ ${TOTAL_COUNT}  -eq 0 ] && exit 2
[ ${FAIL_COUNT}   -gt 0 ] && exit 3
exit 0