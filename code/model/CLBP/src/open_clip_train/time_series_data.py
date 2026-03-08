import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import json


class TimeSeriesTextDataset(Dataset):
    """时序数据-文本对数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 1000,
        input_dim: int = 10,
        transform=None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.transform = transform
        
        # 加载数据
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载时序数据和对应的文本描述"""
        # 这里需要根据您的数据格式进行调整
        # 假设数据格式为 JSON Lines，每行包含时序数据和文本描述
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def _preprocess_time_series(self, ts_data: np.ndarray) -> torch.Tensor:
        """预处理时序数据"""
        # 标准化
        ts_data = (ts_data - ts_data.mean(axis=0)) / (ts_data.std(axis=0) + 1e-8)
        
        # 截断或填充到固定长度
        if len(ts_data) > self.max_seq_len:
            ts_data = ts_data[:self.max_seq_len]
        elif len(ts_data) < self.max_seq_len:
            pad_len = self.max_seq_len - len(ts_data)
            ts_data = np.pad(ts_data, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        
        return torch.from_numpy(ts_data).float()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        # 处理时序数据
        ts_data = np.array(item['time_series'])  # 假设格式为 (seq_len, features)
        ts_tensor = self._preprocess_time_series(ts_data)
        
        # 处理文本数据
        text = item['text']
        text_tokens = self.tokenizer(text)
        
        if self.transform:
            ts_tensor = self.transform(ts_tensor)
            
        return ts_tensor, text_tokens


def get_time_series_data(args, preprocess_fns, epoch=0, tokenizer=None):
    """获取时序数据加载器"""
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    
    if args.train_data:
        train_dataset = TimeSeriesTextDataset(
            data_path=args.train_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            input_dim=args.ts_input_dim,
            transform=preprocess_train,
        )
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loader.num_samples = len(train_dataset)
        train_loader.num_batches = len(train_loader)
        data['train'] = train_loader
    
    if args.val_data:
        val_dataset = TimeSeriesTextDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            input_dim=args.ts_input_dim,
            transform=preprocess_val,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_loader.num_samples = len(val_dataset)
        val_loader.num_batches = len(val_loader)
        data['val'] = val_loader
    
    return data