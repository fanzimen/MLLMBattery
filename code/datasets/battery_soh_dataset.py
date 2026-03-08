"""
Battery SOH Dataset for BatteryGPT Training
- 滑动窗口采样
- 自动计算特征与 SOH 相关性，动态选择高相关特征
- 生成包含特征值和斜率的对话文本
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter


# ============ 对话模板 (动态特征版) ============

# 动态特征问题模板 - 包含2个特征及其斜率
SOH_QUESTIONS_DYNAMIC = [
    "Battery: {battery_type}. {feat1_name}: {feat1_val:.4f}(slope: {feat1_slope:.4f}), {feat2_name}: {feat2_val:.4f}(slope: {feat2_slope:.4f}). Estimate SOH.",
    "For this {battery_type}, features: {feat1_name}={feat1_val:.4f}(slope: {feat1_slope:.4f}), {feat2_name}={feat2_val:.4f}(slope: {feat2_slope:.4f}). What is the SOH?",
    "{battery_type}. Key indicators: {feat1_name}={feat1_val:.4f}(slope: {feat1_slope:.4f}); {feat2_name}={feat2_val:.4f}(slope: {feat2_slope:.4f}). Current SOH?",
    "Analyzing {battery_type}: {feat1_name} is {feat1_val:.4f} with slope={feat1_slope:.4f}, {feat2_name} is {feat2_val:.4f} with slope={feat2_slope:.4f}. Estimate SOH.",
]

# 动态特征中文模板
SOH_QUESTIONS_DYNAMIC_CN = [
    "{battery_type}电池。{feat1_name}: {feat1_val:.4f} (斜率: {feat1_slope:.4f})，{feat2_name}: {feat2_val:.4f} (斜率: {feat2_slope:.4f})。请估计SOH。",
    "电池类型：{battery_type}。关键特征：{feat1_name}={feat1_val:.4f} (变化率: {feat1_slope:.4f})，{feat2_name}={feat2_val:.4f} (变化率: {feat2_slope:.4f})。SOH是多少？",
    "{battery_type}，{feat1_name}为{feat1_val:.4f}（趋势{feat1_slope:.4f}），{feat2_name}为{feat2_val:.4f}（趋势{feat2_slope:.4f}）。估计当前SOH。",
]

# 基础问题模板（备用）
SOH_QUESTIONS_BASIC = [
    "What is the current State of Health (SOH) of this battery?",
    "Can you estimate the SOH based on the given features?",
    "What is the estimated SOH value?",
]

SOH_QUESTIONS_CN = [
    "这块电池当前的健康状态（SOH）是多少？",
    "根据特征，你能估计SOH吗？",
    "这块电池还剩余多少容量？",
]


def compute_feature_correlations(df: pd.DataFrame, exclude_cols: set = None) -> dict:
    """
    计算单个 DataFrame 中所有特征与 SOH 的相关性
    
    Args:
        df: 数据 DataFrame
        exclude_cols: 排除的列名集合
    
    Returns:
        dict: {feature_name: correlation_value}
    """
    if exclude_cols is None:
        exclude_cols = {'soh', 'capacity', 'description', 'file_name', 
                        'battery_id', 'cycle', 'time', 'timestamp', 'date', 'index'}
    
    if 'soh' not in df.columns:
        return {}
    
    soh = df['soh'].values
    
    # 获取有效特征列
    feature_cols = [col for col in df.columns 
                   if col.lower() not in exclude_cols 
                   and col not in exclude_cols
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    correlations = {}
    for col in feature_cols:
        try:
            col_data = df[col].values.astype(float)
            
            # 清理 NaN/Inf
            valid_mask = np.isfinite(col_data) & np.isfinite(soh)
            if valid_mask.sum() < 10:
                continue
            
            # 计算 Pearson 相关系数
            corr, _ = stats.pearsonr(col_data[valid_mask], soh[valid_mask])
            if np.isfinite(corr):
                correlations[col] = abs(corr)  # 使用绝对值
        except Exception:
            continue
    
    return correlations


def select_top_shared_features(data_folder: str, top_k: int = 2, 
                               min_correlation: float = 0.1,
                               exclude_cols: set = None) -> list:
    """
    从多个 CSV 文件中选择共有的高相关性特征
    
    策略：
    1. 分别计算每个 CSV 文件中特征与 SOH 的相关性
    2. 对每个文件，取相关性排名前 N 的特征
    3. 统计所有文件中共同出现的高相关特征
    4. 返回出现次数最多且平均相关性最高的 top_k 个特征
    
    Args:
        data_folder: CSV 文件夹路径
        top_k: 选择的特征数量
        min_correlation: 最小相关性阈值
        exclude_cols: 排除的列名
    
    Returns:
        list: 选中的特征名列表
    """
    if exclude_cols is None:
        exclude_cols = {'soh', 'capacity', 'description', 'file_name', 
                        'battery_id', 'cycle', 'time', 'timestamp', 'date', 'index'}
    
    csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {data_folder}")
        return []
    
    print(f"\n📊 计算特征相关性 ({len(csv_files)} 个文件)...")
    
    # 收集每个文件的 top features
    file_top_features = []  # 每个文件的 top 特征列表
    all_correlations = {}   # 所有特征的相关性累积 {feat: [corr1, corr2, ...]}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(os.path.join(data_folder, csv_file))
            correlations = compute_feature_correlations(df, exclude_cols)
            
            if not correlations:
                continue
            
            # 排序，取 top 10 特征
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, corr in sorted_corr[:10] if corr >= min_correlation]
            file_top_features.append(set(top_features))
            
            # 累积相关性
            for feat, corr in correlations.items():
                if feat not in all_correlations:
                    all_correlations[feat] = []
                all_correlations[feat].append(corr)
            
        except Exception as e:
            print(f"  ⚠️ 跳过 {csv_file}: {e}")
            continue
    
    if not file_top_features:
        print("  ⚠️ 未能计算任何相关性")
        return []
    
    # 统计每个特征在多少个文件的 top 10 中出现
    feature_counts = Counter()
    for feat_set in file_top_features:
        for feat in feat_set:
            feature_counts[feat] += 1
    
    # 计算平均相关性
    feature_avg_corr = {}
    for feat, corrs in all_correlations.items():
        feature_avg_corr[feat] = np.mean(corrs)
    
    # 综合评分: 出现次数 * 平均相关性
    feature_scores = {}
    for feat, count in feature_counts.items():
        avg_corr = feature_avg_corr.get(feat, 0)
        # 归一化出现次数
        normalized_count = count / len(file_top_features)
        # 综合评分
        feature_scores[feat] = normalized_count * avg_corr
    
    # 排序选择 top_k
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, score in sorted_features[:top_k]]
    
    # 打印结果
    print(f"\n📋 特征相关性排名 (Top 10):")
    for i, (feat, score) in enumerate(sorted_features[:10]):
        count = feature_counts[feat]
        avg_corr = feature_avg_corr[feat]
        marker = " ✓" if feat in selected_features else ""
        print(f"   {i+1:2d}. {feat:40s} | 出现: {count:2d}/{len(file_top_features)} | "
              f"平均相关性: {avg_corr:.4f} | 评分: {score:.4f}{marker}")
    
    print(f"\n✅ 选中特征: {selected_features}")
    
    return selected_features


def check_prompt_length(tokenizer_path=None, verbose=True, selected_features=None):
    """
    检查当前模板的文本长度
    
    Args:
        tokenizer_path: Vicuna tokenizer 路径 (如果为 None，只计算字符数)
        verbose: 是否打印详细信息
        selected_features: 选中的特征名列表
    
    Returns:
        dict: 包含统计信息的字典
    """
    # 示例参数
    battery_type = "NCM/Graphite battery from XJTU dataset"
    
    # 使用动态特征名
    if selected_features and len(selected_features) >= 2:
        feat1_name, feat2_name = selected_features[0], selected_features[1]
    else:
        feat1_name, feat2_name = "feature_1", "feature_2"
    
    stats_example = {
        'feat1_name': feat1_name,
        'feat1_val': -0.1234,
        'feat1_slope': -0.0056,
        'feat2_name': feat2_name,
        'feat2_val': -2.3456,
        'feat2_slope': 0.0123,
    }
    soh = 0.856
    
    results = {
        'templates': [],
        'max_chars': 0,
        'max_tokens': 0,
        'avg_chars': 0,
        'avg_tokens': 0
    }
    
    # 加载 tokenizer (可选)
    tokenizer = None
    if tokenizer_path:
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            if verbose:
                print(f"🔧 已加载 Tokenizer: {tokenizer_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ 无法加载 Tokenizer: {e}")
                print("   将只统计字符数")
    
    if verbose:
        print("\n" + "="*60)
        print("📝 对话模板长度分析")
        print("="*60)
    
    all_lengths_chars = []
    all_lengths_tokens = []
    
    # 检查所有模板
    template_groups = [
        ("Dynamic (EN)", SOH_QUESTIONS_DYNAMIC),
        ("Dynamic (CN)", SOH_QUESTIONS_DYNAMIC_CN),
        ("Basic (EN)", SOH_QUESTIONS_BASIC),
        ("Basic (CN)", SOH_QUESTIONS_CN)
    ]
    
    for group_name, templates in template_groups:
        if verbose:
            print(f"\n[{group_name}]")
        
        for i, template in enumerate(templates):
            try:
                # 生成问题
                if '{' in template:
                    question = template.format(battery_type=battery_type, **stats_example)
                else:
                    question = template
                
                # 生成完整对话
                answer = f"SOH={soh:.3f}"
                full_conv = f"### Human: {question}\n### Assistant: {answer}"
                
                # 统计字符数
                num_chars = len(full_conv)
                all_lengths_chars.append(num_chars)
                
                # 统计 token 数
                num_tokens = 0
                if tokenizer:
                    tokens = tokenizer(full_conv, add_special_tokens=False)
                    num_tokens = len(tokens['input_ids'])
                    all_lengths_tokens.append(num_tokens)
                
                # 记录
                results['templates'].append({
                    'type': group_name,
                    'template_id': i,
                    'text': question,
                    'num_chars': num_chars,
                    'num_tokens': num_tokens if tokenizer else None
                })
                
                # 打印
                if verbose:
                    if tokenizer:
                        print(f"  #{i}: {num_chars} chars, {num_tokens} tokens")
                    else:
                        print(f"  #{i}: {num_chars} chars")
                    if len(question) <= 120:
                        print(f"       \"{question}\"")
                    else:
                        print(f"       \"{question}\"")
            
            except KeyError as e:
                if verbose:
                    print(f"  #{i}: ⚠️ 模板格式错误 (缺少键: {e})")
    
    # 统计
    results['max_chars'] = max(all_lengths_chars) if all_lengths_chars else 0
    results['avg_chars'] = np.mean(all_lengths_chars) if all_lengths_chars else 0
    
    if tokenizer and all_lengths_tokens:
        results['max_tokens'] = max(all_lengths_tokens)
        results['avg_tokens'] = np.mean(all_lengths_tokens)
    
    if verbose:
        print(f"\n📊 统计摘要:")
        print(f"   总模板数: {len(results['templates'])}")
        print(f"   字符数: 最大 {results['max_chars']}, 平均 {results['avg_chars']:.1f}")
        if tokenizer:
            print(f"   Token数: 最大 {results['max_tokens']}, 平均 {results['avg_tokens']:.1f}")
            
            # 安全评估
            max_len = results['max_tokens']
            if max_len < 512:
                print(f"\n✅ 安全等级: 极安全 (最大 {max_len} << 2048)")
            elif max_len < 1024:
                print(f"\n✅ 安全等级: 安全 (最大 {max_len} < 2048)")
            elif max_len < 2048:
                print(f"\n⚠️  安全等级: 注意 (最大 {max_len} 接近 2048)")
            else:
                print(f"\n❌ 安全等级: 危险 (最大 {max_len} 超过 2048!)")
    
    return results


class BatterySOHDataset(Dataset):
    """
    电池 SOH 数据集（动态特征版）
    - 从 CSV 文件读取多维特征
    - 自动计算特征与 SOH 相关性，选择 top 2 特征
    - 滑动窗口采样
    - 计算窗口内特征斜率
    - 生成包含特征值和斜率的对话
    """
    
    def __init__(
        self,
        data_folder: str,
        seq_len: int = 40,
        stride: int = 1,
        scaler: StandardScaler = None,
        train: bool = True,
        use_chinese: float = 0.0,
        use_dynamic_prompt: float = 0.8,
        description_folder: str = None,
        selected_features: list = None,
        feature_scaler: dict = None,
    ):
        """
        Args:
            data_folder: 包含 CSV 文件的文件夹路径
            seq_len: 滑动窗口长度
            stride: 滑动步长
            scaler: 标准化器 (训练集创建，测试集传入)
            train: 是否为训练集
            use_chinese: 使用中文对话的概率
            use_dynamic_prompt: 使用动态特征提示的概率
            description_folder: 包含电池描述的CSV文件夹（可选）
            selected_features: 预选的特征列表 (测试集需传入训练集的选择)
            feature_scaler: 特征归一化参数 (测试集需传入训练集的参数)
        """
        self.seq_len = seq_len
        self.stride = stride
        self.train = train
        self.use_chinese = use_chinese
        self.use_dynamic_prompt = use_dynamic_prompt
        
        # 定义排除列
        EXCLUDE_COLS = {'soh', 'capacity', 'description', 'file_name', 
                        'battery_id', 'cycle', 'time', 'timestamp', 'date', 'index'}
        self.exclude_cols = EXCLUDE_COLS
        
        # ===== 1. 选择高相关性特征 =====
        if train:
            # 训练集：计算相关性并选择 top 2 特征
            self.selected_features = select_top_shared_features(
                data_folder=data_folder,
                top_k=2,
                min_correlation=0.1,
                exclude_cols=EXCLUDE_COLS
            )
            if len(self.selected_features) < 2:
                print("⚠️ 未能选出足够的高相关特征，使用默认特征")
                self.selected_features = self._get_default_features(data_folder)
        else:
            # 测试集：使用传入的特征列表
            if selected_features is None or len(selected_features) < 2:
                raise ValueError("测试集需要传入训练集选择的 selected_features!")
            self.selected_features = selected_features
        
        print(f"\n🎯 使用特征: {self.selected_features}")
        
        # 读取所有 CSV 文件
        csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_folder}")
        
        print(f"[{'Train' if train else 'Test'}] Found {len(csv_files)} CSV files")
        
        # 加载描述文件（如果提供）
        self.description_map = {}
        if description_folder and os.path.exists(description_folder):
            self._load_descriptions(description_folder)
        
        # ===== 2. 收集所有数据 =====
        all_data = []
        all_soh = []
        all_dynamic_features = []  # 动态特征值和斜率
        all_battery_type = []
        all_file_indices = []
        all_sample_indices = []
        
        # 用于特征归一化的统计
        feat1_values = []
        feat2_values = []
        
        for file_idx, csv_file in enumerate(csv_files):
            df = pd.read_csv(os.path.join(data_folder, csv_file))
            
            # 获取特征列
            feature_cols = [col for col in df.columns 
                          if col.lower() not in EXCLUDE_COLS and col not in EXCLUDE_COLS]
            
            if not feature_cols or 'soh' not in df.columns:
                print(f"  ⚠️ Skipping {csv_file}: missing features or SOH")
                continue
            
            # 检查选中的特征是否存在
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                print(f"  ⚠️ Skipping {csv_file}: missing selected features {missing_features}")
                continue
            
            # 提取特征和标签
            features = self._clean_features(df, feature_cols)
            soh = df['soh'].values
            
            # 提取选中的两个特征的原始值
            feat1_col = self.selected_features[0]
            feat2_col = self.selected_features[1]
            feat1_raw = df[feat1_col].values.astype(float)
            feat2_raw = df[feat2_col].values.astype(float)
            
            # 获取电池类型描述
            battery_type = self._get_battery_type(csv_file, df)
            
            # 滑动窗口采样
            num_windows = (len(features) - seq_len) // stride + 1
            for i in range(num_windows):
                start = i * stride
                end = start + seq_len
                
                window_data = features[start:end]
                current_soh = soh[end - 1]
                
                # 提取窗口内的动态特征
                dynamic_feats = self._extract_dynamic_features(
                    feat1_raw[start:end], 
                    feat2_raw[start:end]
                )
                
                all_data.append(window_data)
                all_soh.append(current_soh)
                all_dynamic_features.append(dynamic_feats)
                all_battery_type.append(battery_type)
                all_file_indices.append(file_idx)
                all_sample_indices.append(end - 1)
                
                # 收集用于归一化
                feat1_values.append(dynamic_feats['feat1_val'])
                feat2_values.append(dynamic_feats['feat2_val'])
            
            print(f"  Loaded {csv_file}: {len(features)} rows -> {num_windows} windows")
        
        if not all_data:
            raise ValueError("No valid data found!")
        
        self.data = np.stack(all_data)
        self.soh = np.array(all_soh)
        self.dynamic_features = all_dynamic_features
        self.battery_type = all_battery_type
        self.file_indices = all_file_indices
        self.sample_indices = all_sample_indices
        self.feature_cols = feature_cols
        
        # ===== 3. 特征归一化参数 =====
        if train:
            self.feature_scaler = {
                'feat1_mean': np.mean(feat1_values),
                'feat1_std': np.std(feat1_values) + 1e-8,
                'feat2_mean': np.mean(feat2_values),
                'feat2_std': np.std(feat2_values) + 1e-8,
            }
            print(f"\n📐 特征统计:")
            print(f"   {self.selected_features[0]}: mean={self.feature_scaler['feat1_mean']:.4f}, std={self.feature_scaler['feat1_std']:.4f}")
            print(f"   {self.selected_features[1]}: mean={self.feature_scaler['feat2_mean']:.4f}, std={self.feature_scaler['feat2_std']:.4f}")
        else:
            if feature_scaler is None:
                raise ValueError("测试集需要传入训练集的 feature_scaler!")
            self.feature_scaler = feature_scaler
        
        # ===== 4. 标准化时序数据 =====
        if train:
            self.scaler = StandardScaler()
            flat_data = self.data.reshape(-1, self.data.shape[-1])
            self.scaler.fit(flat_data)
            print(f"  [Train] Fitted StandardScaler on {len(flat_data)} samples")
        else:
            if scaler is None:
                raise ValueError("Test dataset requires a scaler from training!")
            self.scaler = scaler
        
        # Apply scaling
        flat_data = self.data.reshape(-1, self.data.shape[-1])
        flat_data = self.scaler.transform(flat_data)
        self.data = flat_data.reshape(self.data.shape)
        
        print(f"  Total samples: {len(self.data)}, Feature dim: {self.data.shape[-1]}")
        print(f"  Dynamic prompt probability: {use_dynamic_prompt}")
    
    def _get_default_features(self, data_folder: str) -> list:
        """获取默认特征（当无法计算相关性时）"""
        csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
        if csv_files:
            df = pd.read_csv(os.path.join(data_folder, csv_files[0]))
            feature_cols = [col for col in df.columns 
                          if col.lower() not in self.exclude_cols 
                          and col not in self.exclude_cols
                          and pd.api.types.is_numeric_dtype(df[col])]
            if len(feature_cols) >= 2:
                return feature_cols[:2]
        return ['feature_0', 'feature_1']
    
    def _extract_dynamic_features(self, feat1_window: np.ndarray, 
                                   feat2_window: np.ndarray) -> dict:
        """
        提取窗口内的动态特征（值和斜率）
        
        Args:
            feat1_window: 特征1在窗口内的值序列
            feat2_window: 特征2在窗口内的值序列
        
        Returns:
            dict: 包含特征值和斜率
        """
        # 清理数据
        feat1_clean = self._clean_array(feat1_window)
        feat2_clean = self._clean_array(feat2_window)
        
        # 特征值：使用窗口末尾的值
        feat1_val = feat1_clean[-1]
        feat2_val = feat2_clean[-1]
        
        # 计算斜率：使用线性回归
        x = np.arange(len(feat1_clean))
        
        # 特征1斜率
        if np.std(feat1_clean) > 1e-10:
            slope1, _, _, _, _ = stats.linregress(x, feat1_clean)
        else:
            slope1 = 0.0
        
        # 特征2斜率
        if np.std(feat2_clean) > 1e-10:
            slope2, _, _, _, _ = stats.linregress(x, feat2_clean)
        else:
            slope2 = 0.0
        
        return {
            'feat1_val': float(feat1_val),
            'feat1_slope': float(slope1),
            'feat2_val': float(feat2_val),
            'feat2_slope': float(slope2),
        }
    
    def _clean_array(self, arr: np.ndarray) -> np.ndarray:
        """清理数组中的 NaN/Inf"""
        arr = arr.copy()
        mask = np.isfinite(arr)
        if not mask.all():
            if mask.any():
                arr[~mask] = arr[mask].mean()
            else:
                arr[:] = 0.0
        return arr
    
    def _load_descriptions(self, description_folder: str):
        """加载电池描述文件"""
        desc_files = [f for f in os.listdir(description_folder) if f.endswith('.csv')]
        for desc_file in desc_files:
            try:
                df = pd.read_csv(os.path.join(description_folder, desc_file))
                if 'description' in df.columns:
                    first_desc = df['description'].iloc[0] if len(df) > 0 else ""
                    battery_type = self._extract_battery_type_from_desc(first_desc)
                    self.description_map[desc_file] = battery_type
            except Exception as e:
                print(f"  ⚠️ Failed to load description from {desc_file}: {e}")
        
        print(f"  Loaded {len(self.description_map)} battery type descriptions")
    
    def _extract_battery_type_from_desc(self, description: str) -> str:
        """从描述中提取电池类型信息"""
        if not description:
            return "Unknown battery"
        
        parts = description.split('.')
        if parts:
            battery_type = parts[0].strip()
            if len(battery_type) > 50:
                battery_type = battery_type[:50] + "..."
            return battery_type
        
        return description[:50] if len(description) > 50 else description
    
    def _get_battery_type(self, csv_file: str, df: pd.DataFrame) -> str:
        """获取电池类型描述"""
        if csv_file in self.description_map:
            return self.description_map[csv_file]
        
        if 'description' in df.columns and len(df) > 0:
            desc = df['description'].iloc[0]
            if isinstance(desc, str) and desc:
                return self._extract_battery_type_from_desc(desc)
        
        return self._infer_battery_type_from_filename(csv_file)
    
    def _infer_battery_type_from_filename(self, filename: str) -> str:
        """从文件名推断电池类型"""
        filename_lower = filename.lower()
        
        if 'mit' in filename_lower:
            return "LFP/Graphite battery"
        elif 'xjtu' in filename_lower:
            return "NCM/Graphite battery"
        elif 'tju' in filename_lower:
            if 'ncm' in filename_lower:
                return "NCM battery"
            elif 'nca' in filename_lower:
                return "NCA battery"
            elif 'lfp' in filename_lower:
                return "LFP battery"
            return "Li-ion battery"
        elif 'calce' in filename_lower:
            return "Li-ion battery (CALCE)"
        
        return "Li-ion battery"
    
    def _clean_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """清洗特征数据"""
        numeric_cols = [col for col in feature_cols 
                       if pd.api.types.is_numeric_dtype(df[col])]
        arr = df[numeric_cols].to_numpy(dtype=np.float32)
        
        if not np.isfinite(arr).all():
            for c in range(arr.shape[1]):
                col = arr[:, c]
                mask = np.isfinite(col)
                if mask.any():
                    col[~mask] = col[mask].mean()
                else:
                    col[~mask] = 0.0
                arr[:, c] = col
        
        return arr
    
    def _format_feature_value(self, value: float, feature_name: str) -> float:
        """
        格式化特征值，确保数值在合理范围内显示
        """
        abs_val = abs(value)
        
        # 非常小的值：可能需要放大
        if abs_val < 0.0001 and abs_val > 0:
            return value * 1000  # 转换为 milli 单位
        
        # 非常大的值：可能需要缩小
        if abs_val > 10000:
            return value / 1000
        
        return value
    
    def _generate_conversation(self, soh: float, dynamic_feats: dict, 
                            battery_type: str) -> list:
        """
        生成对话（动态特征版 - 使用归一化特征值）
        
        Args:
            soh: 当前SOH值
            dynamic_feats: 动态特征字典 (包含原始值)
            battery_type: 电池类型描述
        """
        use_dynamic = random.random() < self.use_dynamic_prompt
        use_cn = random.random() < self.use_chinese
        
        if use_dynamic:
            # 🔥 关键修改: 归一化特征值并裁剪到合理范围
            feat1_normalized = (dynamic_feats['feat1_val'] - self.feature_scaler['feat1_mean']) / self.feature_scaler['feat1_std']
            feat2_normalized = (dynamic_feats['feat2_val'] - self.feature_scaler['feat2_mean']) / self.feature_scaler['feat2_std']
            
            # 裁剪到 [-3, 3] 标准差范围 (覆盖99.7%的正态分布数据)
            feat1_normalized = np.clip(feat1_normalized, -3.0, 3.0)
            feat2_normalized = np.clip(feat2_normalized, -3.0, 3.0)
            
            # 斜率保持原样 (已经是变化率,通常数值较小)
            feat1_slope = np.clip(dynamic_feats['feat1_slope'], -10.0, 10.0)  # 防止极端斜率
            feat2_slope = np.clip(dynamic_feats['feat2_slope'], -10.0, 10.0)
            
            if use_cn:
                template = random.choice(SOH_QUESTIONS_DYNAMIC_CN)
            else:
                template = random.choice(SOH_QUESTIONS_DYNAMIC)
            
            # 使用归一化值填充模板
            question = template.format(
                battery_type=battery_type,
                feat1_name=self.selected_features[0],
                feat1_val=feat1_normalized,      # 归一化值
                feat1_slope=feat1_slope,
                feat2_name=self.selected_features[1],
                feat2_val=feat2_normalized,      # 归一化值
                feat2_slope=feat2_slope,
            )
        else:
            if use_cn:
                question = random.choice(SOH_QUESTIONS_CN)
            else:
                question = random.choice(SOH_QUESTIONS_BASIC)
        
        answer = f"SOH={soh:.3f}"
        
        conversation = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]
        return conversation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            ts_data: (seq_len, num_features) 时序数据
            soh: float, SOH 标签
            conversation: list, 对话
        """
        ts_data = self.data[idx].astype(np.float32)
        soh = self.soh[idx]
        dynamic_feats = self.dynamic_features[idx]
        battery_type = self.battery_type[idx]
        
        conversation = self._generate_conversation(soh, dynamic_feats, battery_type)
        
        return ts_data, soh, conversation
    
    def collate_fn(self, batch):
        """自定义 collate 函数"""
        ts_data = torch.from_numpy(np.stack([item[0] for item in batch]))
        soh_labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        conversations = [item[2] for item in batch]
        
        return {
            'timeseries': ts_data,
            'soh_labels': soh_labels,
            'texts': conversations
        }
    
    def get_transfer_params(self) -> dict:
        """
        获取需要传递给测试集的参数
        
        Returns:
            dict: 包含 scaler, selected_features, feature_scaler
        """
        return {
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_scaler': self.feature_scaler,
        }


def create_train_test_datasets(train_folder: str, test_folder: str, 
                               seq_len: int = 40, stride: int = 1,
                               use_chinese: float = 0.0,
                               use_dynamic_prompt: float = 0.8,
                               description_folder: str = None):
    """
    便捷函数：创建训练集和测试集
    
    Args:
        train_folder: 训练数据文件夹
        test_folder: 测试数据文件夹
        seq_len: 滑动窗口长度
        stride: 滑动步长
        use_chinese: 中文模板概率
        use_dynamic_prompt: 动态模板概率
        description_folder: 描述文件夹
    
    Returns:
        train_dataset, test_dataset
    """
    print("="*60)
    print("📦 创建训练集")
    print("="*60)
    
    train_dataset = BatterySOHDataset(
        data_folder=train_folder,
        seq_len=seq_len,
        stride=stride,
        train=True,
        use_chinese=use_chinese,
        use_dynamic_prompt=use_dynamic_prompt,
        description_folder=description_folder,
    )
    
    # 获取传递参数
    transfer_params = train_dataset.get_transfer_params()
    
    print("\n" + "="*60)
    print("📦 创建测试集")
    print("="*60)
    
    test_dataset = BatterySOHDataset(
        data_folder=test_folder,
        seq_len=seq_len,
        stride=stride,
        train=False,
        use_chinese=use_chinese,
        use_dynamic_prompt=use_dynamic_prompt,
        description_folder=description_folder,
        scaler=transfer_params['scaler'],
        selected_features=transfer_params['selected_features'],
        feature_scaler=transfer_params['feature_scaler'],
    )
    
    return train_dataset, test_dataset


if __name__ == '__main__':
    """
    快速测试模板长度或数据集
    用法: 
        python battery_soh_dataset.py --check_length [--tokenizer_path PATH]
        python battery_soh_dataset.py --test_dataset --data_folder PATH
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='电池 SOH 数据集工具')
    parser.add_argument('--check_length', action='store_true',
                       help='检查模板长度')
    parser.add_argument('--test_dataset', action='store_true',
                       help='测试数据集加载')
    parser.add_argument('--tokenizer_path', type=str,
                       default='/mnt/disk1/fzm/codes/AnomalyGPT/pretrained_ckpt/vicuna_ckpt/Vicuna-7b',
                       help='Vicuna tokenizer 路径')
    parser.add_argument('--data_folder', type=str,
                       default=None,
                       help='数据文件夹路径')
    
    args = parser.parse_args()
    
    if args.check_length:
        # 检查模板长度
        results = check_prompt_length(
            tokenizer_path=args.tokenizer_path,
            verbose=True,
            selected_features=['current_slope', 'current_entropy']  # 示例特征
        )
    
    elif args.test_dataset and args.data_folder:
        # 测试数据集
        print("="*60)
        print("🧪 测试数据集加载")
        print("="*60)
        
        dataset = BatterySOHDataset(
            data_folder=args.data_folder,
            seq_len=40,
            stride=5,
            train=True,
            use_chinese=0.0,
            use_dynamic_prompt=0.9,
        )
        
        print(f"\n📊 数据集统计:")
        print(f"   样本数: {len(dataset)}")
        print(f"   选中特征: {dataset.selected_features}")
        
        # 查看几个样本
        print(f"\n📝 样本示例:")
        for i in range(min(3, len(dataset))):
            ts_data, soh, conv = dataset[i]
            print(f"\n   样本 {i}:")
            print(f"   SOH: {soh:.3f}")
            print(f"   问题: {conv[0]['value']}")
            print(f"   回答: {conv[1]['value']}")
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("💡 使用示例:")
        print("   检查模板长度: python battery_soh_dataset.py --check_length")
        print("   测试数据集:   python battery_soh_dataset.py --test_dataset --data_folder /path/to/data")