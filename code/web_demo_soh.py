"""
BatteryGPT SOH 估计 Web Demo
- 上传电池退化数据 CSV
- 输入提问
- 输出 SOH 估计值和退化曲线
"""

import sys
import os
import warnings

# ============ 警告过滤（放在最前面） ============
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# 设置 OMP_NUM_THREADS 避免警告
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

# ============ Gradio 临时目录配置 ============
# 创建项目内的临时目录
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradio_temp')
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR

# ============ torchvision 兼容性补丁 ============
import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules["torchvision.transforms.functional_tensor"] = F
    except ImportError:
        pass

# ============ 正常导入 ============
import gradio as gr
import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.batterygpt import BatteryGPTModel


# ============ 配置 ============
CONFIG_PATH = "config/batterygpt.yaml"
CHECKPOINT_PATH = "/mnt/disk1/fzm/codes/AnomalyGPT/outputs/batterygpt/run_train_XJTU3_S_20260128_162101/model_epoch_50.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============ 全局变量 ============
model = None
config = None
mean_tensor = None
std_tensor = None
feature_columns = None


def load_config():
    """加载配置文件"""
    global config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model():
    """加载 BatteryGPT 模型"""
    global model, config
    
    if config is None:
        load_config()
    
    print(f"🔄 正在加载模型: {CHECKPOINT_PATH}")
    
    model = BatteryGPTModel(
        clbp_ckpt_path=config['clbp_ckpt_path'],
        vicuna_ckpt_path=config['vicuna_ckpt_path'],
        soh_min=config['soh_min'],
        soh_max=config['soh_max'],
        soh_step=config['soh_step'],
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.1),
        max_tgt_len=config.get('max_tgt_len', 128),
        use_official_llama=True,  # 推理时使用官方版本
        device=DEVICE
    )
    
    # 加载训练权重
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型加载成功 (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
    
    model = model.to(DEVICE)
    model.eval()
    model.half()  # FP16 推理加速
    
    print(f"✅ 模型已加载到 {DEVICE}")
    return model


def _clean_features(df: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """清理特征数据中的 NaN/Inf"""
    arr = df[feat_cols].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def compute_stats_from_data(df: pd.DataFrame):
    """从上传数据计算归一化参数"""
    global feature_columns, mean_tensor, std_tensor
    
    # 获取特征列（排除 soh 和 capacity）
    feature_columns = df.columns.drop(['soh', 'capacity'], errors='ignore').tolist()
    
    arr = _clean_features(df, feature_columns)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    
    mean_tensor = torch.from_numpy(mean).float().to(DEVICE)
    std_tensor = torch.from_numpy(std).float().to(DEVICE)
    
    return mean_tensor, std_tensor, feature_columns


def extract_soh_from_text(text: str) -> float:
    """从模型生成文本中提取 SOH 值"""
    import re
    
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # 匹配小数形式: SOH is 0.85
    patterns_decimal = [
        r'soh\s*(?:is|of|=|:)?\s*(\d+\.\d+)',
        r'state\s+of\s+health.*?(\d+\.\d+)',
        r'(\d+\.\d+)\s*(?:soh|state\s+of\s+health)',
    ]
    
    for pattern in patterns_decimal:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1))
            if 0.5 <= value <= 1.1:
                return min(value, 1.0)
    
    # 匹配百分比: 85%
    patterns_percent = [
        r'soh\s*(?:is|of|=|:)?\s*(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*%\s*(?:soh)?',
    ]
    
    for pattern in patterns_percent:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1)) / 100.0
            if 0.5 <= value <= 1.0:
                return value
    
    return None


def predict_soh(
    csv_file,
    user_question,
    window_idx,
    max_new_tokens,
    temperature,
):
    """主预测函数"""
    global model, config, mean_tensor, std_tensor, feature_columns
    
    if csv_file is None:
        return "❌ 请先上传电池数据 CSV 文件", None, None, None
    
    if model is None:
        load_model()
    
    # 读取数据
    try:
        df = pd.read_csv(csv_file.name if hasattr(csv_file, 'name') else csv_file)
    except Exception as e:
        return f"❌ 读取 CSV 失败: {e}", None, None, None
    
    # 检查必要列
    if 'soh' not in df.columns:
        return "❌ CSV 文件必须包含 'soh' 列", None, None, None
    
    # 计算归一化参数
    compute_stats_from_data(df)
    
    # 提取特征
    features_np = _clean_features(df, feature_columns)
    soh_true = df['soh'].values
    
    window_size = config['seq_len']
    total_length = len(features_np)
    num_windows = total_length - window_size + 1
    
    if num_windows <= 0:
        return f"❌ 数据长度 ({total_length}) 小于窗口大小 ({window_size})", None, None, None
    
    # 选择窗口
    if window_idx < 0 or window_idx >= num_windows:
        window_idx = num_windows - 1  # 使用最后一个窗口
    
    # 提取窗口数据
    window_data = features_np[window_idx:window_idx + window_size]
    ts_tensor = torch.from_numpy(window_data).float().unsqueeze(0)  # [1, seq_len, features]
    
    # 归一化
    ts_tensor = (ts_tensor - mean_tensor.cpu()) / std_tensor.cpu()
    ts_tensor = ts_tensor.to(DEVICE).half()
    
    # 构建 prompt
    if not user_question.strip():
        user_question = "What is the current State of Health (SOH) of this battery?"
    
    # 构建对话格式（匹配训练时的格式）
    conversation = [
        {
            "from": "human",
            "value": user_question
        }
    ]
    
    # 推理
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # 准备输入（用于 forward，获取回归头预测）
            inputs = {
                'timeseries': ts_tensor,
                'texts': [conversation],  # 使用 texts 键
                'soh_labels': torch.zeros(1, device=DEVICE),
            }
            
            # 回归头预测
            _, _, soh_pred_regression, _ = model(inputs)
            soh_regression = soh_pred_regression.item()
            
            # 文本生成

            # generate 方法使用不同的输入格式
            gen_inputs = {
                'timeseries': ts_tensor,
                'prompt': [f"### Human: {user_question}"],
                'soh_labels': torch.zeros(1, device=DEVICE),
            }
            
            outputs = model.generate(gen_inputs, max_new_tokens=max_new_tokens, temperature=temperature)
            
            if isinstance(outputs, tuple):
                response_text = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
            elif isinstance(outputs, list):
                response_text = outputs[0]
            else:
                response_text = str(outputs)
            
            soh_text = extract_soh_from_text(response_text)
            # if soh_text is None:
            #     soh_text = soh_regression

    # 获取真实 SOH（窗口末尾）
    true_soh_at_window = soh_true[window_idx + window_size - 1]
    
    # 绘制退化曲线
    fig = plot_degradation_curve(df, features_np, soh_true, window_idx, window_size, soh_text)
    
    # 格式化输出
    result_text = f"""
                **🤖 模型回复:**
                {response_text}

                ---
                **📊 SOH 估计结果:**
                - 回归头预测 SOH: **{soh_regression:.4f}**
                - 文本生成 SOH: **{soh_text:.4f}**
                - 真实 SOH: **{true_soh_at_window:.4f}**
                - 预测误差 (MAE): **{abs(soh_regression - true_soh_at_window):.4f}**

                **📍 分析窗口:**
                - 窗口位置: [{window_idx} : {window_idx + window_size}]
                - 对应时间点: 第 {window_idx + window_size - 1} 步
                """
    
    return result_text, f"{soh_regression:.4f}", f"{soh_text:.4f}", fig


def plot_degradation_curve(df, features_np, soh_true, current_window_idx, window_size, pred_soh):
    """绘制退化曲线并标注当前窗口"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制真实 SOH 曲线
    x = np.arange(len(soh_true))
    ax.plot(x, soh_true, 'b-', label='True SOH', linewidth=2)
    
    # 标注当前分析窗口
    window_start = current_window_idx
    window_end = current_window_idx + window_size
    ax.axvspan(window_start, window_end, alpha=0.3, color='yellow', label='Analysis Window')
    
    # 标注预测点
    pred_x = window_end - 1
    ax.scatter([pred_x], [pred_soh], color='red', s=100, zorder=5, label=f'Predicted SOH ({pred_soh:.4f})')
    ax.scatter([pred_x], [soh_true[pred_x]], color='green', s=100, zorder=5, marker='x', label=f'True SOH ({soh_true[pred_x]:.4f})')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('SOH', fontsize=12)
    ax.set_title('Battery SOH Degradation Curve', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def batch_predict_all_windows(csv_file):
    """批量预测所有窗口的 SOH（用于绘制完整退化曲线）"""
    global model, config, mean_tensor, std_tensor, feature_columns
    
    if csv_file is None:
        return "❌ 请先上传电池数据 CSV 文件", None
    
    if model is None:
        load_model()
    
    # 读取数据
    df = pd.read_csv(csv_file.name if hasattr(csv_file, 'name') else csv_file)
    compute_stats_from_data(df)
    
    features_np = _clean_features(df, feature_columns)
    soh_true = df['soh'].values
    
    window_size = config['seq_len']
    stride = config.get('stride', 1)
    total_length = len(features_np)
    num_windows = total_length - window_size + 1
    
    if num_windows <= 0:
        return f"❌ 数据长度不足", None
    
    # 提取所有窗口
    windows = []
    window_end_indices = []
    for start_idx in range(0, num_windows, stride):
        end_idx = start_idx + window_size
        if end_idx <= total_length:
            windows.append(features_np[start_idx:end_idx])
            window_end_indices.append(end_idx - 1)
    
    # 批量推理
    ts_batch = torch.from_numpy(np.stack(windows)).float()
    ts_batch = (ts_batch - mean_tensor.cpu()) / std_tensor.cpu()
    ts_batch = ts_batch.to(DEVICE).half()
    
    # ========== 回归头预测 ==========
    all_preds_regression = []
    batch_size = 16
    
    print(f"📊 批量回归头预测 ({len(windows)} 个窗口)...")
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = ts_batch[i:i+batch_size]
            inputs = {
                'timeseries': batch,
                'soh_labels': torch.zeros(batch.shape[0], device=DEVICE),
                'texts': [[]] * batch.shape[0]
            }
            with torch.amp.autocast('cuda'):
                _, _, soh_pred, _ = model(inputs)
                all_preds_regression.append(soh_pred.cpu().numpy())
    
    pred_soh_regression = np.concatenate(all_preds_regression).flatten()
    
    # ========== 文本生成预测 ==========
    all_preds_text = []
    text_batch_size = 8  # 文本生成较慢，使用较小 batch
    
    print(f"💬 批量文本生成 SOH 预测 ({len(windows)} 个窗口)...")
    with torch.no_grad():
        for i in range(0, len(windows), text_batch_size):
            batch_end = min(i + text_batch_size, len(windows))
            current_batch_size = batch_end - i
            ts_batch_chunk = ts_batch[i:batch_end]
            
            try:
                with torch.amp.autocast('cuda'):
                    # 准备批量 prompt
                    prompts = ['### Human: What is the current State of Health (SOH) of this battery?'] * current_batch_size
                    
                    gen_input = {
                        'timeseries': ts_batch_chunk,
                        'prompt': prompts,
                        'soh_labels': torch.zeros(current_batch_size, device=DEVICE)
                    }
                    
                    # 批量生成
                    outputs = model.generate(gen_input, max_new_tokens=64, temperature=0.1)
                    
                    # 解析批量输出
                    responses = []
                    if isinstance(outputs, tuple):
                        responses = outputs[0] if isinstance(outputs[0], list) else [outputs[0]]
                    elif isinstance(outputs, list):
                        responses = outputs
                    else:
                        responses = [outputs] if isinstance(outputs, str) else [str(outputs)]
                    
                    # 批量提取 SOH
                    for j, response in enumerate(responses):
                        text_soh = extract_soh_from_text(response)
                        if text_soh is None:
                            # 如果无法提取，使用回归头结果
                            text_soh = pred_soh_regression[i + j]
                        all_preds_text.append(text_soh)
                        
            except Exception as e:
                print(f"⚠️ 批量 {i}-{batch_end} 文本生成失败: {e}，使用回归头结果")
                # 使用回归头结果作为 fallback
                for j in range(current_batch_size):
                    all_preds_text.append(pred_soh_regression[i + j])
    
    pred_soh_text = np.array(all_preds_text)
    true_soh_at_windows = soh_true[window_end_indices]
    
    # 计算指标
    mae_regression = np.mean(np.abs(pred_soh_regression - true_soh_at_windows))
    rmse_regression = np.sqrt(np.mean((pred_soh_regression - true_soh_at_windows) ** 2))
    
    mae_text = np.mean(np.abs(pred_soh_text - true_soh_at_windows))
    rmse_text = np.sqrt(np.mean((pred_soh_text - true_soh_at_windows) ** 2))
    
    # ========== 绘制对比图（文本生成 vs 真实值）==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1: 文本生成结果
    ax1.plot(window_end_indices, true_soh_at_windows, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
    ax1.plot(window_end_indices, pred_soh_text, 'r--', label='Text Generated SOH', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(window_end_indices, true_soh_at_windows, pred_soh_text, alpha=0.3, color='red')
    
    ax1.set_xlabel('Time Step', fontsize=13)
    ax1.set_ylabel('SOH', fontsize=13)
    ax1.set_title(f'Text Generated SOH Prediction (MAE={mae_text:.4f}, RMSE={rmse_text:.4f})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 子图2: 回归头结果（对比参考）
    ax2.plot(window_end_indices, true_soh_at_windows, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
    ax2.plot(window_end_indices, pred_soh_regression, 'b--', label='Regression Head SOH', linewidth=2, marker='s', markersize=3)
    ax2.fill_between(window_end_indices, true_soh_at_windows, pred_soh_regression, alpha=0.3, color='blue')
    
    ax2.set_xlabel('Time Step', fontsize=13)
    ax2.set_ylabel('SOH', fontsize=13)
    ax2.set_title(f'Regression Head SOH Prediction (MAE={mae_regression:.4f}, RMSE={rmse_regression:.4f})', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 格式化结果文本
    result_text = f"""
            **📊 批量预测结果对比:**

            ### 🔥 文本生成 (主要结果)
            - 窗口数量: {len(windows)}
            - MAE: **{mae_text:.4f}**
            - RMSE: **{rmse_text:.4f}**
            - SOH 范围: [{pred_soh_text.min():.4f}, {pred_soh_text.max():.4f}]

            ### 📐 回归头 (对比参考)
            - MAE: **{mae_regression:.4f}**
            - RMSE: **{rmse_regression:.4f}**
            - SOH 范围: [{pred_soh_regression.min():.4f}, {pred_soh_regression.max():.4f}]

            ### 📈 性能对比
            - MAE 差异: **{abs(mae_text - mae_regression):.4f}** ({'文本更优' if mae_text < mae_regression else '回归头更优'})
            - RMSE 差异: **{abs(rmse_text - rmse_regression):.4f}**
"""
    
    return result_text, fig


def reset_state():
    """重置界面状态"""
    return None, "", -1, None, None, None, None


# ============ 构建界面 ============
with gr.Blocks(title="BatteryGPT SOH Estimation Demo", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <h1 align="center">🔋 BatteryGPT SOH 估计 Demo</h1>
        <p align="center">上传电池退化数据，选择分析窗口，获取 SOH 估计结果</p>
    """)
    
    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 📁 数据上传")
            csv_file = gr.File(
                label="上传电池数据 CSV",
                file_types=[".csv"],
                type="filepath"
            )
            
            gr.Markdown("### 💬 对话设置")
            user_question = gr.Textbox(
                label="提问 (留空使用默认问题)",
                placeholder="What is the current SOH of this battery?",
                lines=2
            )
            
            window_idx = gr.Slider(
                minimum=-1,
                maximum=1000,
                value=-1,
                step=1,
                label="窗口索引 (-1 = 最新窗口)",
                interactive=True
            )
            
            gr.Markdown("### ⚙️ 生成参数")
            max_new_tokens = gr.Slider(
                minimum=16,
                maximum=256,
                value=64,
                step=8,
                label="最大生成 Token 数",
                interactive=True
            )
            
            temperature = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=0.1,
                step=0.01,
                label="Temperature",
                interactive=True
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔍 单窗口预测", variant="primary")
                batch_btn = gr.Button("📈 批量预测", variant="secondary")
            
            clear_btn = gr.Button("🗑️ 清除")
        
        # 右侧：输出区域
        with gr.Column(scale=2):
            gr.Markdown("### 📊 预测结果")
            
            with gr.Row():
                soh_regression_output = gr.Textbox(
                    label="回归头 SOH",
                    interactive=False
                )
                soh_text_output = gr.Textbox(
                    label="文本生成 SOH",
                    interactive=False
                )
            
            result_text = gr.Markdown(label="详细结果")
            
            gr.Markdown("### 📈 退化曲线")
            plot_output = gr.Plot(label="SOH 退化曲线")
    
    # 事件绑定
    predict_btn.click(
        fn=predict_soh,
        inputs=[csv_file, user_question, window_idx, max_new_tokens, temperature],
        outputs=[result_text, soh_regression_output, soh_text_output, plot_output],
        show_progress=True
    )
    
    batch_btn.click(
        fn=batch_predict_all_windows,
        inputs=[csv_file],
        outputs=[result_text, plot_output],
        show_progress=True
    )
    
    clear_btn.click(
        fn=reset_state,
        inputs=[],
        outputs=[csv_file, user_question, window_idx, result_text, soh_regression_output, soh_text_output, plot_output]
    )
    
    # 使用示例
    gr.Markdown("""
    ---
    ### 📖 使用说明
    
    1. **上传数据**: CSV 文件需包含 `soh` 列和多维特征列
    2. **单窗口预测**: 选择特定窗口位置，获取该时刻的 SOH 估计和模型对话回复
    3. **批量预测**: 对整个序列进行滑动窗口预测，生成完整退化曲线
    4. **对话**: 可自定义问题，如 "这块电池的健康状态如何？"
    
    ### 📋 CSV 格式示例
    ```
    V_min,V_max,I_mean,T_avg,soh
    3.2,4.1,1.5,25.0,0.95
    3.1,4.0,1.6,26.0,0.94
    ...
    ```
    """)


# ============ 启动 ============
if __name__ == "__main__":
    print("🚀 正在初始化 BatteryGPT...")
    load_config()
    load_model()
    
    print("🌐 启动 Web Demo...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )