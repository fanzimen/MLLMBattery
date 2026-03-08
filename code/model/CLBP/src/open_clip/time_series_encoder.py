import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class PositionalEncoding(nn.Module):
    """位置编码用于时序数据"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformerEncoder(nn.Module):
    """多维时序数据的Transformer编码器"""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        output_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # 全局池化方式
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Layer normalization
        self.ln_final = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) 时序数据
            mask: (batch_size, seq_len) 可选的mask矩阵
        Returns:
            (batch_size, output_dim) 编码后的特征向量
        """
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 全局池化：平均池化所有时间步
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, d_model)
        
        # 输出投影
        x = self.output_projection(x)  # (batch_size, output_dim)
        
        # Layer normalization
        x = self.ln_final(x)
        
        return x


class TimeSeriesCNN(nn.Module):
    """基于CNN的时序编码器（备选方案）"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dims: list = [64, 128, 256, 512],
        kernel_sizes: list = [3, 3, 3, 3],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(hidden_dims[-1], output_dim)
        self.ln_final = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        # 转换为CNN格式: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = self.conv_layers(x)
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)
        
        # 输出投影
        x = self.output_projection(x)
        x = self.ln_final(x)
        
        return x