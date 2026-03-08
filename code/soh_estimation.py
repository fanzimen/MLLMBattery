"""
SOH 估计脚本 - 基于 CLIP 时序-文本对齐

核心思路：
1. 生成 SOH 候选文本库: ["SOH=1.000", "SOH=0.999", ..., "SOH=0.700"]
2. 对每个滑动窗口 (长度40, 步长1) 提取时序特征
3. 计算时序特征与所有 SOH 文本特征的余弦相似度
4. 选择相似度最高的 SOH 值作为估计值
5. 计算 MAE, RMSE 等误差指标
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# 将 src 目录添加到 python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'clbp/src'))
import open_clip

# --- [配置区] ---
# 1. 模型和权重路径
# PRETRAINED_PATH = '/mnt/disk1/fzm/codes/Time-LLM/clbp/logs/battery_moreinfo_fuse_1024_2026-01-04_20-06-19/checkpoints/epoch_best.pt'
PRETRAINED_PATH = '/mnt/disk1/fzm/codes/Time-LLM/clbp/logs/battery_moreinfo_fuse_1024_2026-01-13_20-54-57/checkpoints/epoch_best.pt'
MODEL_NAME = 'btr-1024-bert-sl40'
BATCH_SIZE_TS = 64     # 时序特征批量大小
BATCH_SIZE_TEXT = 64    # 文本特征批量大小
USE_AMP = True          # 使用混合精度

# 如果使用 Qwen 模型，取消下面的注释
# PRETRAINED_PATH = '/mnt/disk1/fzm/codes/Time-LLM/clbp/logs/battery_qwen_1024_2026-01-05_17-04-32/checkpoints/epoch_best.pt'
# PRETRAINED_PATH = '/mnt/disk1/fzm/codes/Time-LLM/clbp/logs/battery_qwen_1024_2026-01-12_15-11-23/checkpoints/epoch_best.pt'
# MODEL_NAME = 'btr-1024-qwen2.5'
# BATCH_SIZE_TS = 1
# BATCH_SIZE_TEXT = 1
# USE_AMP = False

# 2. 数据路径配置
TRAIN_DATA_DIR = 'data_preprocess/soh_data/train_10'  # 用于计算归一化参数
# TEST_DATA_DIR = 'data_preprocess/soh_data/test_10'    # 测试数据
TEST_DATA_DIR = 'data_preprocess/soh_data/train_10'    # 测试数据

# 3. SOH 候选值配置
SOH_MIN = 0.700       # 最小 SOH 值
SOH_MAX = 1.000       # 最大 SOH 值
SOH_STEP = 0.001      # SOH 步长 (精度到小数点后3位)

# 4. 窗口参数
WINDOW_SIZE = 40
STRIDE = 1            # 滑动步长

# 5. 输出配置
OUTPUT_BASE_DIR = 'soh_estimation_results'

# 6. 可视化配置
ENABLE_TSNE = True    # 是否绘制 t-SNE
TSNE_SAMPLES = 500    # t-SNE 采样数量 (太多会很慢)
# --- [配置区结束] ---


# ========== [模型配置] ==========
btr_1024_qwen_cfg = {
    "embed_dim": 1024,
    "timeseries_cfg": {
        "input_dim": 16,   
        "seq_len": 40,    
        "patch_size": 4,     
        "layers": 12,
        "width": 1024,
        "heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1
    },
    "text_cfg": {
        "hf_model_name": "pretrained_model/Qwen2.5-7B",
        "hf_tokenizer_name": "pretrained_model/Qwen2.5-7B",
        "hf_proj_type": "linear",
        "hf_model_pretrained": True,
        "width": 1024,
        "context_length": 100,
        "output_tokens": False,
    },
    "custom_text": False,
}

btr_1024_bert_sl40_cfg = {
    "embed_dim": 1024,
    "timeseries_cfg": {
        "input_dim": 16,   
        "seq_len": 40,
        "patch_size": 4,     
        "layers": 12,
        "width": 1024,
        "heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1
    },
    "text_cfg": {
        "hf_model_name": "/mnt/disk1/fzm/codes/MMLLM4Battery/clbp/src/pretrained_text/bert",
        "hf_tokenizer_name": "/mnt/disk1/fzm/codes/MMLLM4Battery/clbp/src/pretrained_text/bert",
        "hf_proj_type": "linear",
        "hf_model_pretrained": True,
        "width": 1024,
        "context_length": 100,
        "output_tokens": False,
    },
    "custom_text": False,
}

open_clip.factory._MODEL_CONFIGS['btr-1024-qwen2.5'] = btr_1024_qwen_cfg
open_clip.factory._MODEL_CONFIGS['btr-1024-bert-sl40'] = btr_1024_bert_sl40_cfg


# ========== [工具函数] ==========
def _list_csvs(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return sorted([str(x) for x in p.glob("*.csv")])
    elif p.is_file():
        return [str(p)]
    return []


def _clean_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    if arr.size > 0 and not np.isfinite(arr).all():
        for c in range(arr.shape[1]):
            col = arr[:, c]
            mask = np.isfinite(col)
            mean_val = col[mask].mean() if mask.any() else 0.0
            col[~mask] = mean_val
            arr[:, c] = col
    return arr


def compute_global_mean_std(soh_files: list[str]) -> tuple[np.ndarray, np.ndarray, list]:
    """计算全局均值和标准差"""
    print(f"📊 正在根据 {len(soh_files)} 个文件计算全局归一化参数...")
    if not soh_files:
        raise ValueError("No SOH csv files found.")
    
    first_df = pd.read_csv(soh_files[0])
    feature_columns = first_df.columns.drop(['soh', 'capacity'], errors='ignore').tolist()
    
    count = 0
    running_sum = np.zeros(len(feature_columns), dtype=np.float64)
    running_sumsq = np.zeros(len(feature_columns), dtype=np.float64)

    for fp in soh_files:
        try:
            df = pd.read_csv(fp)
            arr = _clean_features(df, feature_columns)
            if arr.size > 0:
                running_sum += arr.sum(axis=0, dtype=np.float64)
                running_sumsq += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
                count += arr.shape[0]
        except Exception as e:
            print(f"Warning: {fp}: {e}")
            continue
            
    mean = (running_sum / count).astype(np.float32)
    var = (running_sumsq / count) - np.square(mean, dtype=np.float64)
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std[std == 0] = 1.0
    
    return mean, std, feature_columns


def generate_soh_candidates(soh_min: float, soh_max: float, step: float) -> tuple[list[str], np.ndarray]:
    """
    生成 SOH 候选文本列表和对应的数值
    返回: (["SOH=1.000", "SOH=0.999", ...], [1.000, 0.999, ...])
    """
    soh_values = np.arange(soh_max, soh_min - step/2, -step)  # 从大到小
    soh_texts = [f"SOH={v:.3f}" for v in soh_values]
    return soh_texts, soh_values


# ========== [特征提取] ==========
@torch.no_grad()
def extract_timeseries_features_batched(model, ts_tensor, batch_size, device, use_amp=False):
    """批量提取时序特征"""
    num_samples = ts_tensor.shape[0]
    all_features = []
    
    for i in range(0, num_samples, batch_size):
        batch = ts_tensor[i:i+batch_size].to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                features = model.encode_timeseries(batch, normalize=True)
        else:
            features = model.encode_timeseries(batch, normalize=True)
        all_features.append(features.cpu())
        
    return torch.cat(all_features, dim=0)


@torch.no_grad()
def extract_text_features_batched(model, tokenizer, text_list, batch_size, device, use_amp=False):
    """批量提取文本特征"""
    num_samples = len(text_list)
    all_features = []
    
    for i in range(0, num_samples, batch_size):
        batch_texts = text_list[i:i+batch_size]
        tokens = tokenizer(batch_texts).to(device)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                features = model.encode_text(tokens, normalize=True)
        else:
            features = model.encode_text(tokens, normalize=True)
        all_features.append(features.cpu())
        del tokens
        
    return torch.cat(all_features, dim=0)


# ========== [SOH 估计核心函数] ==========
def estimate_soh_for_battery(
    soh_file_path: Path,
    model,
    tokenizer,
    device,
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    feature_columns: list,
    soh_text_features: torch.Tensor,  # 预计算的 SOH 文本特征 [N_soh, embed_dim]
    soh_values: np.ndarray,           # 对应的 SOH 数值 [N_soh]
    run_output_dir: Path
) -> dict:
    """
    对单个电池文件进行 SOH 估计
    """
    battery_name = soh_file_path.stem
    print(f"\n⚡ 正在处理: {battery_name}")
    
    # 1. 读取数据
    df = pd.read_csv(soh_file_path)
    features_np = _clean_features(df, feature_columns)
    soh_true = df['soh'].values  # 真实 SOH 序列
    
    total_length = len(features_np)
    num_windows = total_length - WINDOW_SIZE + 1
    
    if num_windows <= 0:
        raise ValueError(f"数据长度 ({total_length}) 小于窗口大小 ({WINDOW_SIZE})")
    
    print(f"   📏 数据长度: {total_length}, 窗口数: {num_windows}")
    
    # 2. 构建所有滑动窗口 (步长=1)
    windows = []
    window_end_indices = []  # 记录每个窗口对应的结束索引
    
    for start_idx in range(0, num_windows, STRIDE):
        end_idx = start_idx + WINDOW_SIZE
        window = features_np[start_idx:end_idx]
        windows.append(window)
        window_end_indices.append(end_idx - 1)  # 窗口最后一个时刻的索引
    
    # 转换为 Tensor 并标准化
    ts_batch = torch.from_numpy(np.stack(windows)).to(dtype=torch.float32)
    ts_batch = (ts_batch - mean_tensor.cpu()) / std_tensor.cpu()
    
    print(f"   🔄 提取 {len(windows)} 个窗口的时序特征...")
    
    # 3. 提取时序特征
    ts_features = extract_timeseries_features_batched(
        model, ts_batch, BATCH_SIZE_TS, device, use_amp=USE_AMP
    )
    ts_features = F.normalize(ts_features.float(), dim=-1)
    
    # 4. 计算与所有 SOH 文本的相似度，并选择最匹配的
    print(f"   🔍 计算相似度并匹配 SOH...")
    
    # 相似度矩阵: [num_windows, num_soh_candidates]
    similarity_matrix = (ts_features @ soh_text_features.T).numpy()
    
    # 找到每个窗口最匹配的 SOH 索引
    best_match_indices = np.argmax(similarity_matrix, axis=1)
    soh_estimated = soh_values[best_match_indices]
    
    # 5. 获取真实 SOH (窗口结束时刻)
    soh_true_at_windows = soh_true[window_end_indices]
    
    # 6. 计算误差指标
    errors = soh_estimated - soh_true_at_windows
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(np.abs(errors))
    
    # 7. 创建输出目录并保存结果
    output_dir = run_output_dir / battery_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存估计结果 CSV
    results_df = pd.DataFrame({
        'window_end_idx': window_end_indices,
        'soh_true': soh_true_at_windows,
        'soh_estimated': soh_estimated,
        'error': errors,
        'abs_error': np.abs(errors)
    })
    results_df.to_csv(output_dir / 'soh_estimation_results.csv', index=False)
    
    # 8. 绘制 SOH 估计曲线
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(window_end_indices, soh_true_at_windows, 'b-', label='True SOH', linewidth=1.5)
    plt.plot(window_end_indices, soh_estimated, 'r--', label='Estimated SOH', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('SOH')
    plt.title(f'{battery_name} - SOH Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(window_end_indices, errors, 'g-', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.fill_between(window_end_indices, errors, 0, alpha=0.3)
    plt.xlabel('Time Step')
    plt.ylabel('Error (Estimated - True)')
    plt.title(f'Estimation Error (MAE={mae:.4f}, RMSE={rmse:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'soh_estimation_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. 绘制误差分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Estimation Error')
    plt.ylabel('Frequency')
    plt.title(f'{battery_name} - Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🔥 [新增] 10. 绘制相似度热力图 (窗口 vs SOH 候选值)
    print(f"   🎨 绘制相似度热力图...")
    plot_similarity_heatmap(
        similarity_matrix,
        window_end_indices,
        soh_values,
        soh_true_at_windows,
        soh_estimated,
        output_dir,
        battery_name
    )
    
    # 11. t-SNE 可视化 (可选)
    if ENABLE_TSNE and len(ts_features) > 10:
        print(f"   🎨 绘制 t-SNE...")
        plot_tsne(
            ts_features.numpy(), 
            soh_true_at_windows, 
            output_dir / 'tsne_visualization.png',
            battery_name,
            max_samples=TSNE_SAMPLES
        )
    
    stats = {
        "battery_name": battery_name,
        "num_windows": len(windows),
        "mae": float(mae),
        "rmse": float(rmse),
        "max_error": float(max_error),
        "soh_true_min": float(soh_true_at_windows.min()),
        "soh_true_max": float(soh_true_at_windows.max()),
    }
    
    # 保存统计信息
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}, MaxError: {max_error:.4f}")
    
    return stats, ts_features.numpy(), soh_true_at_windows


# 🔥 [新增函数] 绘制相似度热力图
def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,  # [num_windows, num_soh_candidates]
    window_indices: list,
    soh_candidates: np.ndarray,
    soh_true: np.ndarray,
    soh_estimated: np.ndarray,
    output_dir: Path,
    battery_name: str,
    max_windows_to_plot: int = 50  # 最多绘制的窗口数
):
    """
    绘制退化片段与 SOH 候选值的匹配度热力图
    
    策略：
    1. 如果窗口数 > max_windows_to_plot，均匀采样
    2. 热力图横轴为 SOH 候选值，纵轴为时间步
    3. 在图上标注真实 SOH 和估计 SOH 的位置
    """
    num_windows = len(window_indices)
    
    # 采样窗口 (如果太多)
    if num_windows > max_windows_to_plot:
        sample_indices = np.linspace(0, num_windows - 1, max_windows_to_plot, dtype=int)
        similarity_sampled = similarity_matrix[sample_indices]
        window_indices_sampled = [window_indices[i] for i in sample_indices]
        soh_true_sampled = soh_true[sample_indices]
        soh_estimated_sampled = soh_estimated[sample_indices]
    else:
        similarity_sampled = similarity_matrix
        window_indices_sampled = window_indices
        soh_true_sampled = soh_true
        soh_estimated_sampled = soh_estimated
    
    # --- 绘图 1: 完整热力图 (所有 SOH 候选值) ---
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(
        similarity_sampled, 
        aspect='auto', 
        cmap='viridis',
        interpolation='nearest'
    )
    
    # 设置坐标轴
    ax.set_xlabel('SOH Candidates', fontsize=12)
    ax.set_ylabel('Time Step (Window End Index)', fontsize=12)
    ax.set_title(f'{battery_name} - Similarity Heatmap (Degradation vs SOH Candidates)', fontsize=14)
    
    # X 轴：每隔一定间隔显示 SOH 值
    num_soh = len(soh_candidates)
    x_tick_step = max(1, num_soh // 20)  # 显示约 20 个刻度
    x_ticks = np.arange(0, num_soh, x_tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{soh_candidates[i]:.2f}" for i in x_ticks], rotation=45, ha='right')
    
    # Y 轴：显示时间步
    y_tick_step = max(1, len(window_indices_sampled) // 20)
    y_ticks = np.arange(0, len(window_indices_sampled), y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([window_indices_sampled[i] for i in y_ticks])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # 🔥 在热力图上标注真实 SOH 和估计 SOH 的位置
    for i, (window_idx, true_val, est_val) in enumerate(zip(
        window_indices_sampled, soh_true_sampled, soh_estimated_sampled
    )):
        # 找到真实 SOH 在候选列表中最接近的索引
        true_soh_idx = np.argmin(np.abs(soh_candidates - true_val))
        est_soh_idx = np.argmin(np.abs(soh_candidates - est_val))
        
        # 标记真实 SOH (蓝色圆圈)
        ax.plot(true_soh_idx, i, 'o', color='blue', markersize=5, alpha=0.6, 
                markeredgecolor='white', markeredgewidth=0.5)
        
        # 标记估计 SOH (红色 X)
        ax.plot(est_soh_idx, i, 'x', color='red', markersize=6, alpha=0.8, 
                markeredgewidth=1.5)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='True SOH', markeredgecolor='white'),
        Line2D([0], [0], marker='x', color='red', markersize=8, 
               label='Estimated SOH', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 绘图 2: 局部放大热力图 (聚焦在真实 SOH 附近) ---
    # 计算 SOH 真实值的范围，聚焦在 ±0.05 范围内
    soh_true_min = max(SOH_MIN, soh_true.min() - 0.05)
    soh_true_max = min(SOH_MAX, soh_true.max() + 0.05)
    
    # 找到对应的候选索引范围
    focus_start_idx = np.searchsorted(soh_candidates[::-1], soh_true_max, side='left')
    focus_end_idx = np.searchsorted(soh_candidates[::-1], soh_true_min, side='right')
    focus_start_idx = len(soh_candidates) - focus_end_idx
    focus_end_idx = len(soh_candidates) - focus_start_idx
    
    if focus_end_idx > focus_start_idx:
        similarity_focused = similarity_sampled[:, focus_start_idx:focus_end_idx]
        soh_focused = soh_candidates[focus_start_idx:focus_end_idx]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(
            similarity_focused, 
            aspect='auto', 
            cmap='RdYlGn',  # 红黄绿渐变，更明显
            interpolation='nearest'
        )
        
        ax.set_xlabel('SOH Candidates (Zoomed)', fontsize=12)
        ax.set_ylabel('Time Step (Window End Index)', fontsize=12)
        ax.set_title(f'{battery_name} - Focused Heatmap (SOH: {soh_true_min:.3f} ~ {soh_true_max:.3f})', fontsize=14)
        
        # X 轴刻度
        x_tick_step = max(1, len(soh_focused) // 15)
        x_ticks = np.arange(0, len(soh_focused), x_tick_step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{soh_focused[i]:.3f}" for i in x_ticks], rotation=45, ha='right')
        
        # Y 轴刻度
        y_tick_step = max(1, len(window_indices_sampled) // 20)
        y_ticks = np.arange(0, len(window_indices_sampled), y_tick_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([window_indices_sampled[i] for i in y_ticks])
        
        cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        # 标注真实和估计 SOH
        for i, (true_val, est_val) in enumerate(zip(soh_true_sampled, soh_estimated_sampled)):
            # 映射到局部索引
            true_local_idx = np.argmin(np.abs(soh_focused - true_val))
            est_local_idx = np.argmin(np.abs(soh_focused - est_val))
            
            ax.plot(true_local_idx, i, 'o', color='blue', markersize=6, alpha=0.7, 
                    markeredgecolor='white', markeredgewidth=1)
            ax.plot(est_local_idx, i, 'x', color='red', markersize=7, alpha=0.9, 
                    markeredgewidth=2)
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_heatmap_focused.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"      ✅ 热力图已保存 (完整 + 聚焦)")



def plot_tsne(features: np.ndarray, soh_values: np.ndarray, save_path: Path, 
              title: str, max_samples: int = 500):
    """
    绘制 t-SNE 可视化，颜色按 SOH 值着色
    """
    from sklearn.manifold import TSNE
    
    # 采样以加速
    if len(features) > max_samples:
        indices = np.linspace(0, len(features) - 1, max_samples, dtype=int)
        features = features[indices]
        soh_values = soh_values[indices]
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=soh_values, 
        cmap='viridis',
        s=20,
        alpha=0.7
    )
    plt.colorbar(scatter, label='SOH')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'{title} - t-SNE Visualization (colored by SOH)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ========== [主函数] ==========
def main():
    print("=" * 60)
    print("🔋 SOH 估计程序 - 基于 CLIP 时序-文本对齐")
    print("=" * 60)
    
    # 1. 创建输出目录
    model_short = MODEL_NAME.replace('-', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(OUTPUT_BASE_DIR) / model_short / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果保存至: {run_output_dir}\n")
    
    # 2. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 设备: {device}")
    
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=None, 
        device=device,
        force_timeseries_seq_len=WINDOW_SIZE,
    )
    
    # 加载 checkpoint
    if PRETRAINED_PATH and os.path.isfile(PRETRAINED_PATH):
        print(f"⏳ 加载 Checkpoint: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if next(iter(state_dict.items()))[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 权重加载成功")
    else:
        print(f"❌ 找不到 Checkpoint: {PRETRAINED_PATH}")
        return

    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # 3. 计算全局归一化参数
    print("\n📊 计算全局统计量...")
    train_files = _list_csvs(TRAIN_DATA_DIR)
    mean, std, feature_columns = compute_global_mean_std(train_files)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device)
    print(f"✅ 统计完成，特征数: {len(feature_columns)}")

    # 4. 生成 SOH 候选文本并提取特征 (只需做一次)
    print(f"\n📝 生成 SOH 候选文本库 (范围: {SOH_MIN:.3f} ~ {SOH_MAX:.3f}, 步长: {SOH_STEP})...")
    soh_texts, soh_values = generate_soh_candidates(SOH_MIN, SOH_MAX, SOH_STEP)
    print(f"   共 {len(soh_texts)} 个候选 SOH 值")
    print(f"   示例: {soh_texts[:3]} ... {soh_texts[-3:]}")
    
    print(f"\n🔄 提取 SOH 文本特征...")
    soh_text_features = extract_text_features_batched(
        model, tokenizer, soh_texts, BATCH_SIZE_TEXT, device, use_amp=USE_AMP
    )
    soh_text_features = F.normalize(soh_text_features.float(), dim=-1)
    print(f"✅ SOH 文本特征维度: {soh_text_features.shape}")

    # 5. 获取测试文件列表
    test_files = sorted(list(Path(TEST_DATA_DIR).glob("*.csv")))
    print(f"\n📂 找到 {len(test_files)} 个测试文件")

    # 6. 批量处理
    all_results = []
    all_features_for_global_tsne = []
    all_soh_for_global_tsne = []
    
    for soh_path in test_files:
        try:
            stats, features, soh_values_at_windows = estimate_soh_for_battery(
                soh_path,
                model, tokenizer, device,
                mean_tensor, std_tensor, feature_columns,
                soh_text_features, soh_values,
                run_output_dir
            )
            all_results.append(stats)
            
            # 收集用于全局 t-SNE
            if ENABLE_TSNE:
                # 均匀采样以避免数据太多
                sample_indices = np.linspace(0, len(features) - 1, min(100, len(features)), dtype=int)
                all_features_for_global_tsne.append(features[sample_indices])
                all_soh_for_global_tsne.append(soh_values_at_windows[sample_indices])
                
        except Exception as e:
            print(f"❌ 处理 {soh_path.name} 时出错: {e}")
            continue

    # 7. 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        cols = ['battery_name', 'mae', 'rmse', 'max_error', 'num_windows', 'soh_true_min', 'soh_true_max']
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        
        summary_path = run_output_dir / 'SUMMARY_RESULTS.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # 全局 t-SNE (所有电池合并)
        if ENABLE_TSNE and all_features_for_global_tsne:
            print("\n🎨 绘制全局 t-SNE...")
            global_features = np.concatenate(all_features_for_global_tsne, axis=0)
            global_soh = np.concatenate(all_soh_for_global_tsne, axis=0)
            plot_tsne(
                global_features, global_soh,
                run_output_dir / 'global_tsne_visualization.png',
                'All Batteries',
                max_samples=1000
            )
        
        print(f"\n{'=' * 60}")
        print(f"✅ 批量处理完成！")
        print(f"📊 汇总结果: {summary_path}")
        print(f"📈 平均 MAE: {summary_df['mae'].mean():.4f}")
        print(f"📈 平均 RMSE: {summary_df['rmse'].mean():.4f}")
        print(f"{'=' * 60}")
        
        # 保存配置
        config = {
            "model_name": MODEL_NAME,
            "pretrained_path": PRETRAINED_PATH,
            "soh_min": SOH_MIN,
            "soh_max": SOH_MAX,
            "soh_step": SOH_STEP,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "num_soh_candidates": len(soh_texts),
            "timestamp": datetime.now().isoformat()
        }
        with open(run_output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    else:
        print("\n⚠️ 没有成功处理任何文件。")


if __name__ == "__main__":
    main()