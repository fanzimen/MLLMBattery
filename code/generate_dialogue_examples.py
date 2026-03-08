"""
生成 BatteryGPT 对话数据示例
- 展示训练时使用的对话模板
- 输出多种格式（JSON, TXT, CSV）
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.battery_soh_dataset import (
    SOH_QUESTIONS_ENHANCED,
    SOH_QUESTIONS_ENHANCED_CN,
    SOH_QUESTIONS_BASIC,
    SOH_QUESTIONS_CN,
)


def print_template_overview():
    """打印所有对话模板概览"""
    print("\n" + "="*80)
    print("📋 BatteryGPT 对话模板概览")
    print("="*80)
    
    print("\n【增强版英文模板】(使用4个关键特征: current slope, current entropy, CV Q, CV time)")
    for i, t in enumerate(SOH_QUESTIONS_ENHANCED):
        print(f"  {i+1}. {t}")
    
    print("\n【增强版中文模板】")
    for i, t in enumerate(SOH_QUESTIONS_ENHANCED_CN):
        print(f"  {i+1}. {t}")
    
    print("\n【基础版英文模板】")
    for i, t in enumerate(SOH_QUESTIONS_BASIC):
        print(f"  {i+1}. {t}")
    
    print("\n【基础版中文模板】")
    for i, t in enumerate(SOH_QUESTIONS_CN):
        print(f"  {i+1}. {t}")


def generate_example_with_stats():
    """生成带统计特征的示例对话"""
    # 模拟统计特征（基于真实数据范围）
    example_stats = {
        'i_slope': -0.311,      # mA/step (已转换)
        'i_entropy': 2.094,     # 熵值
        'cv_q': 0.097,          # Ah
        'cv_time': 19.585,      # min (已转换)
    }
    
    battery_type = "NCM battery"
    soh = 0.918
    
    print("\n" + "="*80)
    print("📝 示例对话（使用模拟统计特征）")
    print("="*80)
    
    print(f"\n电池类型: {battery_type}")
    print(f"真实 SOH: {soh:.4f}")
    print(f"\n关键特征:")
    print(f"  current slope: {example_stats['i_slope']:.3f} A/step")
    print(f"  current entropy: {example_stats['i_entropy']:.3f}")
    print(f"  CV Q: {example_stats['cv_q']:.3f} Ah")
    print(f"  CV charge time: {example_stats['cv_time']:.3f} min")
    
    print("\n--- 增强版对话 (English) ---")
    prompt = SOH_QUESTIONS_ENHANCED[0].format(
        battery_type=battery_type,
        **example_stats
    )
    print(f"Human: {prompt}")
    print(f"GPT: SOH={soh:.3f}")
    
    print("\n--- 增强版对话 (中文) ---")
    prompt_cn = SOH_QUESTIONS_ENHANCED_CN[0].format(
        battery_type=battery_type,
        **example_stats
    )
    print(f"Human: {prompt_cn}")
    print(f"GPT: SOH={soh:.3f}")
    
    print("\n--- 基础版对话 ---")
    print(f"Human: {SOH_QUESTIONS_BASIC[0]}")
    print(f"GPT: SOH={soh:.3f}")


def identify_key_column_names(columns) -> dict:
    """识别关键特征列名"""
    key_cols = {
        'current_slope': None,
        'current_entropy': None,
        'cv_q': None,
        'cv_time': None,
    }
    
    for col in columns:
        col_lower = col.lower()
        
        # current slope
        if 'current' in col_lower and 'slope' in col_lower:
            key_cols['current_slope'] = col
        
        # current entropy
        if 'current' in col_lower and 'entropy' in col_lower:
            key_cols['current_entropy'] = col
        
        # CV Q
        if 'cv' in col_lower and 'q' in col_lower:
            key_cols['cv_q'] = col
        elif 'cv q' in col_lower:
            key_cols['cv_q'] = col
        
        # CV charge time
        if 'cv' in col_lower and ('time' in col_lower or 'charge' in col_lower):
            key_cols['cv_time'] = col
    
    return key_cols


def extract_key_features(df: pd.DataFrame, idx: int, key_col_names: dict) -> dict:
    """从DataFrame中提取关键特征并进行单位转换"""
    stats = {
        'i_slope': 0.0,
        'i_entropy': 0.0,
        'cv_q': 0.0,
        'cv_time': 0.0,
    }
    
    # 提取 current slope (单位转换: A -> mA)
    if key_col_names['current_slope'] and key_col_names['current_slope'] in df.columns:
        raw_value = float(df[key_col_names['current_slope']].iloc[idx])
        if abs(raw_value) < 0.001:
            stats['i_slope'] = raw_value * 1000  # A -> mA
        else:
            stats['i_slope'] = raw_value
    
    # 提取 current entropy
    if key_col_names['current_entropy'] and key_col_names['current_entropy'] in df.columns:
        stats['i_entropy'] = float(df[key_col_names['current_entropy']].iloc[idx])
    
    # 提取 CV Q
    if key_col_names['cv_q'] and key_col_names['cv_q'] in df.columns:
        stats['cv_q'] = float(df[key_col_names['cv_q']].iloc[idx])
    
    # 提取 CV charge time (单位转换: s -> min)
    if key_col_names['cv_time'] and key_col_names['cv_time'] in df.columns:
        raw_value = float(df[key_col_names['cv_time']].iloc[idx])
        if raw_value > 100:
            stats['cv_time'] = raw_value / 60  # s -> min
        else:
            stats['cv_time'] = raw_value
    
    return stats


def generate_from_real_data(data_folder: str, output_dir: str, num_samples: int = 20):
    """从真实数据生成对话示例"""
    print(f"\n📂 从真实数据生成示例: {data_folder}")
    
    EXCLUDE_COLS = {'soh', 'capacity', 'description', 'file_name', 
                    'battery_id', 'cycle', 'time', 'timestamp', 'date', 'index'}
    
    csv_files = sorted(Path(data_folder).glob("*.csv"))[:5]  # 取前5个文件
    
    if not csv_files:
        print("❌ 未找到 CSV 文件")
        return
    
    all_examples = []
    window_size = 40
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        battery_name = csv_file.stem
        
        # 检查必要列
        if 'soh' not in df.columns:
            print(f"⚠️ 跳过 {csv_file.name}: 缺少 soh 列")
            continue
        
        soh_values = df['soh'].values
        
        # 识别关键列
        key_col_names = identify_key_column_names(df.columns)
        
        # 检查是否有足够的关键特征
        missing_features = [k for k, v in key_col_names.items() if v is None]
        if missing_features:
            print(f"⚠️ {csv_file.name} 缺少特征: {missing_features}")
        
        # 推断电池类型
        filename_lower = csv_file.name.lower()
        if 'mit' in filename_lower:
            battery_type = "LFP/Graphite battery"
        elif 'xjtu' in filename_lower:
            battery_type = "NCM/Graphite battery"
        elif 'tju' in filename_lower:
            if 'ncm' in filename_lower:
                battery_type = "NCM battery"
            elif 'nca' in filename_lower:
                battery_type = "NCA battery"
            elif 'lfp' in filename_lower:
                battery_type = "LFP battery"
            else:
                battery_type = "Li-ion battery"
        else:
            battery_type = "Li-ion battery"
        
        # 采样窗口
        total_windows = len(df) - window_size + 1
        if total_windows <= 0:
            print(f"⚠️ {csv_file.name} 数据不足 ({len(df)} < {window_size})")
            continue
        
        sample_indices = np.linspace(0, total_windows - 1, min(num_samples, total_windows), dtype=int)
        
        for idx in sample_indices:
            window_end_idx = idx + window_size - 1
            current_soh = soh_values[window_end_idx]
            
            # 提取关键特征（窗口末尾）
            stats = extract_key_features(df, window_end_idx, key_col_names)
            
            # 生成对话
            try:
                prompt_en = SOH_QUESTIONS_ENHANCED[0].format(
                    battery_type=battery_type,
                    **stats
                )
                prompt_cn = SOH_QUESTIONS_ENHANCED_CN[0].format(
                    battery_type=battery_type,
                    **stats
                )
                
                all_examples.append({
                    'battery_id': battery_name,
                    'window_idx': idx,
                    'window_end_idx': window_end_idx,
                    'true_soh': float(current_soh),
                    'battery_type': battery_type,
                    'features': {
                        'current_slope_mA': stats['i_slope'],
                        'current_entropy': stats['i_entropy'],
                        'cv_q_Ah': stats['cv_q'],
                        'cv_time_min': stats['cv_time'],
                    },
                    'prompt_en': prompt_en,
                    'prompt_cn': prompt_cn,
                    'answer': f"SOH={current_soh:.3f}",
                })
            except KeyError as e:
                print(f"⚠️ 模板格式化错误: {e}")
                continue
    
    if not all_examples:
        print("❌ 没有生成任何示例")
        return
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, 'dialogue_examples.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    # 可读文本
    txt_path = os.path.join(output_dir, 'dialogue_examples.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BatteryGPT 对话数据示例\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据来源: {data_folder}\n")
        f.write(f"样本数量: {len(all_examples)}\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(all_examples):
            f.write(f"\n{'='*60}\n")
            f.write(f"Example {i+1}: {ex['battery_id']} (Window {ex['window_idx']})\n")
            f.write(f"{'='*60}\n")
            f.write(f"Battery Type: {ex['battery_type']}\n")
            f.write(f"True SOH: {ex['true_soh']:.4f}\n\n")
            
            f.write("--- 关键特征 (窗口末尾) ---\n")
            for k, v in ex['features'].items():
                f.write(f"  {k}: {v:.3f}\n")
            
            f.write("\n--- English Dialogue ---\n")
            f.write(f"Human: {ex['prompt_en']}\n")
            f.write(f"GPT: {ex['answer']}\n")
            
            f.write("\n--- 中文对话 ---\n")
            f.write(f"Human: {ex['prompt_cn']}\n")
            f.write(f"GPT: {ex['answer']}\n")
    
    # CSV (便于表格查看)
    csv_path = os.path.join(output_dir, 'dialogue_examples.csv')
    df_examples = pd.DataFrame([
        {
            'battery_id': ex['battery_id'],
            'window_idx': ex['window_idx'],
            'true_soh': ex['true_soh'],
            'battery_type': ex['battery_type'],
            'current_slope': ex['features']['current_slope_mA'],
            'current_entropy': ex['features']['current_entropy'],
            'cv_q': ex['features']['cv_q_Ah'],
            'cv_time': ex['features']['cv_time_min'],
            'prompt_en': ex['prompt_en'],
            'prompt_cn': ex['prompt_cn'],
        }
        for ex in all_examples
    ])
    df_examples.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 对话示例已保存:")
    print(f"   - {json_path}")
    print(f"   - {txt_path}")
    print(f"   - {csv_path}")
    
    # 打印前3个示例
    print(f"\n📝 前3个对话示例预览:")
    for i, ex in enumerate(all_examples[:3]):
        print(f"\n--- Example {i+1}: {ex['battery_id']} ---")
        print(f"Features: I_slope={ex['features']['current_slope_mA']:.3f}, "
              f"I_entropy={ex['features']['current_entropy']:.3f}, "
              f"CV_Q={ex['features']['cv_q_Ah']:.3f}, "
              f"CV_time={ex['features']['cv_time_min']:.3f}")
        print(f"Human: {ex['prompt_en'][:120]}...")
        print(f"GPT: {ex['answer']}")
    
    return all_examples


def main():
    parser = argparse.ArgumentParser(description='生成 BatteryGPT 对话示例')
    parser.add_argument('--data_folder', type=str, default=None, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./dialogue_examples', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=20, help='每个电池的样本数量')
    parser.add_argument('--show_templates', action='store_true', help='显示模板概览')
    args = parser.parse_args()
    
    # 显示模板概览
    if args.show_templates:
        print_template_overview()
    
    # 生成模拟示例
    generate_example_with_stats()
    
    # 从真实数据生成
    if args.data_folder and os.path.exists(args.data_folder):
        generate_from_real_data(
            data_folder=args.data_folder,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    else:
        print("\n💡 提示: 使用 --data_folder 指定数据目录可从真实数据生成示例")
        print("   例如: python generate_dialogue_examples.py --data_folder /mnt/disk1/fzm/codes/AnomalyGPT/data/soh_data --show_templates")


if __name__ == '__main__':
    main()