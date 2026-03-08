import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')




class Dataset_Battery_SOH(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_1',
                 target='soh', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, desc_path='data_preprocess/full_descriptions_simple_decimal3',
                 scaler=None):  # scaler 参数保留但不使用
        """
        电池 SOH 估计数据集 - 文件级滑动窗口采样
        Args:
            root_path: 数据根目录
            data_path: 子文件夹名
            desc_path: 文本描述文件目录
            size: [seq_len, label_len, pred_len]
        """
        if size == None:
            self.seq_len = 40
            self.label_len = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = 0
            self.pred_len = 1
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        
        self.root_path = root_path
        self.data_folder = data_path
        self.desc_path = desc_path
        self.external_scaler = scaler
        self.__read_data__()
        self.enc_in = self.data_list[0].shape[-1]  # 从第一个文件获取特征维度

    def __read_data__(self):
        """读取数据并生成文件级索引映射"""
        data_folder_path = os.path.join(self.root_path, self.data_folder)
        csv_files = sorted([f for f in os.listdir(data_folder_path) if f.endswith('.csv')])
        
        if not csv_files:
            raise ValueError(f"未在 {data_folder_path} 找到任何 CSV 文件")

        # 定义要排除的列
        EXCLUDE_COLS = {'soh', 'capacity', 'description', 'file_name', 'battery_id', 
                        'cycle', 'time', 'timestamp', 'date', 'index'}
        
        print(f"[{self.flag}] 找到 {len(csv_files)} 个 CSV 文件")
        
        # === [核心修复] 步骤 1: 根据 flag 决定是否拟合 scaler ===
        if self.scale and self.external_scaler is None and self.flag == 'train':
            # 仅训练集拟合新的 scaler
            self.scaler = StandardScaler()
            print(f"[{self.flag}] 正在拟合新的 StandardScaler...")
            
            all_train_data = []
            # 第一次遍历：收集所有特征数据用于拟合 scaler
            for csv_file in csv_files:
                df_raw = pd.read_csv(os.path.join(data_folder_path, csv_file))
                feature_cols = [col for col in df_raw.columns 
                            if col.lower() not in EXCLUDE_COLS and col not in EXCLUDE_COLS]
                if feature_cols:
                    all_train_data.append(self._clean_features(df_raw, feature_cols))
            
            if not all_train_data:
                raise ValueError(f"[{self.flag}] 没有找到任何有效数据")
            
            # 在训练数据上拟合 scaler
            self.scaler.fit(np.vstack(all_train_data))
            print(f"[{self.flag}] StandardScaler 拟合完成 (特征维度: {self.scaler.n_features_in_})")
            
        elif self.scale and self.external_scaler is not None:
            # 测试/验证集使用传入的 scaler
            self.scaler = self.external_scaler
            print(f"[{self.flag}] 使用外部传入的 StandardScaler (特征维度: {self.scaler.n_features_in_})")
            
        else:
            # 不进行标准化
            self.scaler = None
            print(f"[{self.flag}] 不使用标准化")

        # === 步骤 2: 逐文件读取并存储 ===
        self.data_list = []           # 每个文件的特征数据
        self.soh_list = []            # 每个文件的 SOH 标签
        self.desc_list = []           # 每个文件的描述列表
        self.file_names = []          # 文件名列表
        self.file_sample_counts = []  # 每个文件的样本数量
        
        for file_idx, csv_file in enumerate(csv_files):
            df_raw = pd.read_csv(os.path.join(data_folder_path, csv_file))
            
            # 读取描述
            if 'description' in df_raw.columns:
                file_descriptions = df_raw['description'].tolist()
            else:
                desc_csv_path = os.path.join(self.desc_path, csv_file)
                if os.path.exists(desc_csv_path):
                    file_descriptions = pd.read_csv(desc_csv_path)['description'].tolist()
                else:
                    file_descriptions = ["Battery state of health estimation task."] * len(df_raw)

            # 获取特征列
            feature_cols = [col for col in df_raw.columns 
                        if col.lower() not in EXCLUDE_COLS and col not in EXCLUDE_COLS]
            
            if not feature_cols:
                print(f"⚠️  文件 {csv_file} 没有特征列,跳过")
                continue
            
            data = self._clean_features(df_raw, feature_cols)
            
            if 'soh' not in df_raw.columns:
                print(f"⚠️  文件 {csv_file} 缺少 'soh' 列,跳过")
                continue
            
            soh = df_raw[['soh']].values
            
            # 清洗 SOH 数据
            soh_mask = np.isfinite(soh.flatten())
            if not soh_mask.all():
                mean_soh = soh[soh_mask].mean() if soh_mask.any() else 0.8
                soh[~soh_mask.reshape(-1, 1)] = mean_soh
            
            # 标准化特征 (使用 self.scaler.transform，而非 fit)
            if self.scale and self.scaler is not None:
                data = self.scaler.transform(data)
            
            # 计算该文件可生成的样本数量
            sample_count = max(0, len(data) - self.seq_len + 1)
            
            self.data_list.append(data)
            self.soh_list.append(soh)
            self.desc_list.append(file_descriptions)
            self.file_names.append(csv_file)
            self.file_sample_counts.append(sample_count)
            
            print(f"  加载 {csv_file}: {len(data)} 条原始样本 -> {sample_count} 个窗口样本")

        if not self.data_list:
            raise ValueError(f"[{self.flag}] 没有生成任何有效样本!")
        
        # === 步骤 3: 预计算累积索引 ===
        self.cumulative_counts = np.cumsum(self.file_sample_counts)
        
        total_samples = sum(self.file_sample_counts)
        print(f"[{self.flag}] 总窗口样本数: {total_samples}")
        print(f"[{self.flag}] 特征维度: {self.data_list[0].shape}")

    def _clean_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """清洗特征数据: 自动过滤非数值列,处理 -inf/inf/NaN"""
        numeric_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        
        if not numeric_cols:
            raise ValueError(f"没有找到任何数值类型的特征列!")
        
        arr = df[numeric_cols].to_numpy(dtype=np.float32, copy=True)
        
        if arr.size == 0:
            return arr
        
        # 处理非有限值
        finite = np.isfinite(arr)
        if not finite.all():
            for c in range(arr.shape[1]):
                col = arr[:, c]
                mask = np.isfinite(col)
                if mask.any():
                    mean_val = col[mask].mean()
                else:
                    mean_val = 0.0
                col[~mask] = mean_val
                arr[:, c] = col
        
        return arr
    
    def __getitem__(self, index):
        """
        文件级滑动窗口采样
        返回格式: (seq_x, seq_y, seq_x_mark, seq_y_mark, description, file_idx, sample_idx)
        """
        # === 步骤 1: 定位文件和文件内索引 ===
        file_idx = np.searchsorted(self.cumulative_counts, index + 1)
        if file_idx == 0:
            idx_in_file = index
        else:
            idx_in_file = index - self.cumulative_counts[file_idx - 1]
        
        # === 步骤 2: 从对应文件中提取窗口 ===
        data_seq = self.data_list[file_idx]
        soh_seq = self.soh_list[file_idx]
        desc_seq = self.desc_list[file_idx]
        
        s_begin = idx_in_file
        s_end = s_begin + self.seq_len

        seq_x = data_seq[s_begin:s_end]
        seq_y = soh_seq[s_end - 1:s_end]  # 取窗口末尾的 SOH
        
        # === 步骤 3: 获取描述 ===
        desc_idx = min(s_end - 1, len(desc_seq) - 1)
        description = desc_seq[desc_idx] if desc_idx >= 0 else "Battery state of health estimation task."
        
        # === 步骤 4: 伪造时间戳 (保持兼容性) ===
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        seq_y_mark = np.zeros((1, 1), dtype=np.float32)
        
        # === 步骤 5: 原始样本索引 (用于可视化) ===
        sample_idx = s_end - 1
        
        return (
            seq_x.astype(np.float32), 
            seq_y.astype(np.float32), 
            seq_x_mark, 
            seq_y_mark, 
            description,
            file_idx,
            sample_idx
        )

    def __len__(self):
        return sum(self.file_sample_counts)
    
    def inverse_transform(self, data):
        """逆标准化"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data