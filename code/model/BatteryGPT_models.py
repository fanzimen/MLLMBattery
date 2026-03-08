"""
BatteryGPT 核心组件
- SOHPromptLearner: 将 SOH 分布转换为 Soft Prompts
- LinearLayer: 简单的投影层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    """简单的线性投影层，支持多层"""
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else output_dim
            layers.append(nn.Linear(in_d, output_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class SOHPromptLearner(nn.Module):
    """
    SOH Prompt Learner
    将 SOH 概率分布转换为可学习的 Soft Prompts，用于引导 LLM 生成。
    
    输入: SOH 分布 (batch, num_soh_bins)
    输出: Soft Prompts (batch, num_prompts, llm_embed_dim)
    """
    def __init__(
        self, 
        num_soh_bins: int = 301,      # SOH 候选值数量 (0.700 到 1.000，步长 0.001)
        llm_embed_dim: int = 4096,    # LLM 嵌入维度 (Vicuna-7B)
        num_prompts: int = 8,         # 生成的 Prompt Token 数量
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_soh_bins = num_soh_bins
        self.llm_embed_dim = llm_embed_dim
        self.num_prompts = num_prompts
        
        # 1D 卷积处理分布
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(num_prompts)  # (batch, 64, num_prompts)
        )
        
        # 投影到 LLM 嵌入空间
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_embed_dim)
        )
        
        # 可学习的基础 Prompts (类似于 Prefix Tuning)
        self.base_prompts = nn.Parameter(
            torch.randn(1, num_prompts, llm_embed_dim) * 0.02
        )
    
    def forward(self, soh_distribution: torch.Tensor) -> torch.Tensor:
        """
        Args:
            soh_distribution: (batch, num_soh_bins) SOH 概率分布
        Returns:
            prompts: (batch, num_prompts, llm_embed_dim)
        """
        batch_size = soh_distribution.shape[0]
        
        # (batch, num_soh_bins) -> (batch, 1, num_soh_bins)
        x = soh_distribution.unsqueeze(1)
        
        # 1D 卷积提取分布特征
        x = self.conv_layers(x)  # (batch, 64, num_prompts)
        x = x.permute(0, 2, 1)   # (batch, num_prompts, 64)
        
        # 投影到 LLM 空间
        dynamic_prompts = self.proj(x)  # (batch, num_prompts, llm_embed_dim)
        
        # 与基础 Prompts 结合
        prompts = self.base_prompts.expand(batch_size, -1, -1) + dynamic_prompts
        
        return prompts


class SOHRegressionHead(nn.Module):
    """
    SOH 回归头
    从时序特征直接预测 SOH 值，用于精度评估。
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # SOH ∈ [0, 1]
        )
    
    def forward(self, ts_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_features: (batch, input_dim)
        Returns:
            soh: (batch, 1)
        """
        return self.regressor(ts_features)