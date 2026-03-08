"""
检查 BatteryGPT 数据集中的文本长度
- 分析对话模板生成的提示词长度
- 检查描述文件的文本长度
- 统计 token 数量分布
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入对话模板
try:
    from datasets.battery_soh_dataset import (
        SOH_QUESTIONS_ENHANCED,
        SOH_QUESTIONS_ENHANCED_CN,
        SOH_QUESTIONS_BASIC,
        SOH_QUESTIONS_CN,
    )
except ImportError:
    print("⚠️ 无法导入模板，使用默认值")
    SOH_QUESTIONS_ENHANCED = ["Battery: {battery_type}. Features: slope={i_slope:.3f}, entropy={i_entropy:.3f}. Estimate SOH."]
    SOH_QUESTIONS_ENHANCED_CN = ["{battery_type} 电池。特征：斜率={i_slope:.3f}，熵={i_entropy:.3f}。估计SOH。"]
    SOH_QUESTIONS_BASIC = ["What is the current SOH?"]
    SOH_QUESTIONS_CN = ["当前SOH是多少？"]


def count_tokens(tokenizer, text):
    """统计文本的 token 数量"""
    tokens = tokenizer(text, add_special_tokens=False)
    return len(tokens['input_ids'])


def analyze_prompt_templates(tokenizer):
    """分析对话模板的长度"""
    print("\n" + "="*60)
    print("📝 对话模板长度分析")
    print("="*60)
    
    # 示例参数
    battery_type = "NCM/Graphite battery from XJTU dataset"
    stats = {
        'i_slope': 0.123,
        'i_entropy': 2.456,
        'cv_q': 1.234,
        'cv_time': 45.6
    }
    
    results = []
    
    # 英文增强模板
    print("\n[英文增强模板]")
    for i, template in enumerate(SOH_QUESTIONS_ENHANCED):
        prompt = template.format(battery_type=battery_type, **stats)
        num_tokens = count_tokens(tokenizer, prompt)
        results.append({
            'type': 'Enhanced (EN)',
            'template_id': i,
            'text': prompt,
            'num_tokens': num_tokens
        })
        print(f"  #{i}: {num_tokens} tokens")
        print(f"       \"{prompt[:80]}...\"")
    
    # 中文增强模板
    print("\n[中文增强模板]")
    for i, template in enumerate(SOH_QUESTIONS_ENHANCED_CN):
        try:
            prompt = template.format(battery_type=battery_type, **stats)
            num_tokens = count_tokens(tokenizer, prompt)
            results.append({
                'type': 'Enhanced (CN)',
                'template_id': i,
                'text': prompt,
                'num_tokens': num_tokens
            })
            print(f"  #{i}: {num_tokens} tokens")
            print(f"       \"{prompt[:80]}...\"")
        except KeyError as e:
            print(f"  #{i}: ⚠️ 模板格式错误 (缺少键: {e})")
    
    # 基础模板
    print("\n[基础模板]")
    for i, prompt in enumerate(SOH_QUESTIONS_BASIC):
        num_tokens = count_tokens(tokenizer, prompt)
        results.append({
            'type': 'Basic (EN)',
            'template_id': i,
            'text': prompt,
            'num_tokens': num_tokens
        })
        print(f"  #{i}: {num_tokens} tokens - \"{prompt}\"")
    
    for i, prompt in enumerate(SOH_QUESTIONS_CN):
        num_tokens = count_tokens(tokenizer, prompt)
        results.append({
            'type': 'Basic (CN)',
            'template_id': i,
            'text': prompt,
            'num_tokens': num_tokens
        })
        print(f"  #{i}: {num_tokens} tokens - \"{prompt}\"")
    
    # 统计
    df = pd.DataFrame(results)
    print(f"\n📊 模板统计:")
    print(f"   最小长度: {df['num_tokens'].min()} tokens")
    print(f"   最大长度: {df['num_tokens'].max()} tokens")
    print(f"   平均长度: {df['num_tokens'].mean():.1f} tokens")
    
    return df


def analyze_descriptions(tokenizer, description_folder):
    """分析电池描述文件的长度"""
    print("\n" + "="*60)
    print("📄 电池描述文件长度分析")
    print("="*60)
    
    if not description_folder or not os.path.exists(description_folder):
        print(f"⚠️ 描述文件夹不存在: {description_folder}")
        return None
    
    desc_files = list(Path(description_folder).glob("*.csv"))
    print(f"📂 找到 {len(desc_files)} 个描述文件")
    
    results = []
    
    for desc_file in tqdm(desc_files, desc="分析描述文件"):
        try:
            df = pd.read_csv(desc_file)
            if 'description' not in df.columns:
                continue
            
            # 分析第一条描述
            if len(df) > 0:
                desc_text = df['description'].iloc[0]
                if isinstance(desc_text, str):
                    num_tokens = count_tokens(tokenizer, desc_text)
                    results.append({
                        'file': desc_file.name,
                        'text': desc_text[:100] + "..." if len(desc_text) > 100 else desc_text,
                        'num_chars': len(desc_text),
                        'num_tokens': num_tokens
                    })
        except Exception as e:
            print(f"⚠️ 处理 {desc_file.name} 时出错: {e}")
    
    if not results:
        print("⚠️ 未找到有效的描述文本")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"\n📊 描述文本统计 (共 {len(df)} 个):")
    print(f"   最小长度: {df['num_tokens'].min()} tokens ({df['num_chars'].min()} chars)")
    print(f"   最大长度: {df['num_tokens'].max()} tokens ({df['num_chars'].max()} chars)")
    print(f"   平均长度: {df['num_tokens'].mean():.1f} tokens ({df['num_chars'].mean():.1f} chars)")
    print(f"   中位数:   {df['num_tokens'].median():.0f} tokens")
    
    # 显示最长的几个
    print(f"\n🔝 最长的 5 个描述:")
    top5 = df.nlargest(5, 'num_tokens')
    for idx, row in top5.iterrows():
        print(f"   {row['file']}: {row['num_tokens']} tokens")
        print(f"      \"{row['text']}\"")
    
    return df


def analyze_dataset_prompts(tokenizer, data_folder, num_samples=100):
    """分析实际数据集生成的完整对话长度"""
    print("\n" + "="*60)
    print(f"🔍 数据集实际对话长度分析 (采样 {num_samples} 个)")
    print("="*60)
    
    if not os.path.exists(data_folder):
        print(f"⚠️ 数据文件夹不存在: {data_folder}")
        return None
    
    csv_files = list(Path(data_folder).glob("*.csv"))
    if not csv_files:
        print(f"⚠️ 未找到 CSV 文件")
        return None
    
    print(f"📂 找到 {len(csv_files)} 个数据文件")
    
    results = []
    count = 0
    
    battery_type = "Li-ion battery"  # 简化
    
    for csv_file in csv_files:
        if count >= num_samples:
            break
        
        try:
            df = pd.read_csv(csv_file)
            if 'soh' not in df.columns:
                continue
            
            # 识别关键列
            key_cols = {}
            for col in df.columns:
                c = col.lower()
                if 'current' in c and 'slope' in c:
                    key_cols['i_slope'] = col
                if 'current' in c and 'entropy' in c:
                    key_cols['i_entropy'] = col
            
            # 采样几行
            sample_indices = np.linspace(0, len(df)-1, min(5, len(df)), dtype=int)
            
            for idx in sample_indices:
                if count >= num_samples:
                    break
                
                # 提取特征
                stats = {
                    'i_slope': float(df[key_cols['i_slope']].iloc[idx]) if 'i_slope' in key_cols else 0.0,
                    'i_entropy': float(df[key_cols['i_entropy']].iloc[idx]) if 'i_entropy' in key_cols else 0.0,
                    'cv_q': 0.0,
                    'cv_time': 0.0
                }
                
                soh = df['soh'].iloc[idx]
                
                # 生成问题 (使用第一个模板)
                question = SOH_QUESTIONS_ENHANCED[0].format(
                    battery_type=battery_type,
                    **stats
                )
                
                # 生成回答
                answer = f"SOH={soh:.3f}"
                
                # 完整对话
                full_conversation = f"### Human: {question}\n### Assistant: {answer}"
                
                num_tokens = count_tokens(tokenizer, full_conversation)
                
                results.append({
                    'file': csv_file.name,
                    'idx': idx,
                    'soh': soh,
                    'conversation': full_conversation[:150] + "...",
                    'num_tokens': num_tokens
                })
                
                count += 1
        
        except Exception as e:
            print(f"⚠️ 处理 {csv_file.name} 时出错: {e}")
    
    if not results:
        print("⚠️ 未找到有效的对话样本")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"\n📊 对话长度统计 (共 {len(df)} 个样本):")
    print(f"   最小长度: {df['num_tokens'].min()} tokens")
    print(f"   最大长度: {df['num_tokens'].max()} tokens")
    print(f"   平均长度: {df['num_tokens'].mean():.1f} tokens")
    print(f"   中位数:   {df['num_tokens'].median():.0f} tokens")
    
    # 分布
    print(f"\n📈 长度分布:")
    bins = [0, 50, 100, 150, 200, 300, 500, 1000, 2048]
    counts, _ = np.histogram(df['num_tokens'], bins=bins)
    for i in range(len(bins)-1):
        print(f"   {bins[i]:4d}-{bins[i+1]:4d} tokens: {counts[i]:3d} 个 ({counts[i]/len(df)*100:.1f}%)")
    
    # 显示几个示例
    print(f"\n📝 对话示例:")
    for idx, row in df.head(3).iterrows():
        print(f"\n   Sample {idx+1}: {row['num_tokens']} tokens")
        print(f"   {row['conversation']}")
    
    return df


def plot_length_distribution(df_prompts, df_descriptions, df_conversations, output_dir):
    """绘制长度分布图"""
    print("\n" + "="*60)
    print("📊 生成可视化图表...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 模板长度
    if df_prompts is not None:
        axes[0].hist(df_prompts['num_tokens'], bins=20, color='steelblue', alpha=0.7)
        axes[0].set_title('Prompt Templates')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Count')
        axes[0].axvline(df_prompts['num_tokens'].mean(), color='red', linestyle='--', label=f"Mean: {df_prompts['num_tokens'].mean():.1f}")
        axes[0].legend()
    
    # 描述长度
    if df_descriptions is not None:
        axes[1].hist(df_descriptions['num_tokens'], bins=20, color='coral', alpha=0.7)
        axes[1].set_title('Battery Descriptions')
        axes[1].set_xlabel('Tokens')
        axes[1].axvline(df_descriptions['num_tokens'].mean(), color='red', linestyle='--', label=f"Mean: {df_descriptions['num_tokens'].mean():.1f}")
        axes[1].legend()
    
    # 对话长度
    if df_conversations is not None:
        axes[2].hist(df_conversations['num_tokens'], bins=30, color='green', alpha=0.7)
        axes[2].set_title('Full Conversations')
        axes[2].set_xlabel('Tokens')
        axes[2].axvline(df_conversations['num_tokens'].mean(), color='red', linestyle='--', label=f"Mean: {df_conversations['num_tokens'].mean():.1f}")
        # 标记 2048 上限
        axes[2].axvline(2048, color='orange', linestyle='-.', linewidth=2, label='LLaMA Limit: 2048')
        axes[2].legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'text_length_distribution.png')
    plt.savefig(output_path, dpi=150)
    print(f"✅ 图表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='检查 BatteryGPT 数据集文本长度')
    parser.add_argument('--vicuna_path', type=str, 
                       default='/mnt/disk1/fzm/codes/AnomalyGPT/pretrained_ckpt/vicuna_ckpt/Vicuna-7b',
                       help='Vicuna tokenizer 路径')
    parser.add_argument('--description_folder', type=str,
                       default='data/full_descriptions_withmoreinfo_decimal3',
                       help='电池描述文件夹')
    parser.add_argument('--data_folder', type=str,
                       default='data/soh_data/train_XJTU4_S',
                       help='训练数据文件夹')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='数据集采样数量')
    
    args = parser.parse_args()
    
    # 加载 tokenizer
    print(f"🔧 加载 Tokenizer: {args.vicuna_path}")
    tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_path, use_fast=False)
    print(f"   词汇表大小: {len(tokenizer)}")
    
    # 分析各部分
    df_prompts = analyze_prompt_templates(tokenizer)
    df_descriptions = analyze_descriptions(tokenizer, args.description_folder)
    df_conversations = analyze_dataset_prompts(tokenizer, args.data_folder, args.num_samples)
    
    # 绘图
    plot_length_distribution(df_prompts, df_descriptions, df_conversations, args.output_dir)
    
    # 保存 CSV
    if df_prompts is not None:
        df_prompts.to_csv(os.path.join(args.output_dir, 'prompt_lengths.csv'), index=False)
    if df_descriptions is not None:
        df_descriptions.to_csv(os.path.join(args.output_dir, 'description_lengths.csv'), index=False)
    if df_conversations is not None:
        df_conversations.to_csv(os.path.join(args.output_dir, 'conversation_lengths.csv'), index=False)
    
    # 总结
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    
    print("\n📋 总结:")
    if df_prompts is not None:
        print(f"   提示模板:   平均 {df_prompts['num_tokens'].mean():.1f} tokens")
    if df_descriptions is not None:
        print(f"   电池描述:   平均 {df_descriptions['num_tokens'].mean():.1f} tokens")
    if df_conversations is not None:
        print(f"   完整对话:   平均 {df_conversations['num_tokens'].mean():.1f} tokens")
        print(f"   最大长度:   {df_conversations['num_tokens'].max()} tokens")
        
        # 风险评估
        max_len = df_conversations['num_tokens'].max()
        if max_len < 512:
            print(f"\n✅ 安全等级: 极安全 (最大长度 {max_len} << 2048)")
        elif max_len < 1024:
            print(f"\n✅ 安全等级: 安全 (最大长度 {max_len} < 2048)")
        elif max_len < 2048:
            print(f"\n⚠️  安全等级: 注意 (最大长度 {max_len} 接近 2048)")
        else:
            print(f"\n❌ 安全等级: 危险 (最大长度 {max_len} 超过 2048!)")


if __name__ == '__main__':
    main()