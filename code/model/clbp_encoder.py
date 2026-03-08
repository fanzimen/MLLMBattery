"""
CLBP Encoder Wrapper
封装预训练的 CLBP 模型，提供时序编码和文本编码接口。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 将 CLBP 的 src 目录添加到 Python 路径
# 请根据您的实际路径修改
CLBP_SRC_PATH = '/mnt/disk1/fzm/codes/Time-LLM/clbp/src'
if CLBP_SRC_PATH not in sys.path:
    sys.path.insert(0, CLBP_SRC_PATH)

import open_clip

# --- CLBP 模型配置 (与您训练时保持一致) ---
CLBP_MODEL_CONFIG = {
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
CLBP_MODEL_NAME = 'clbp-battery-bert'

# 注册配置
if CLBP_MODEL_NAME not in open_clip.factory._MODEL_CONFIGS:
    open_clip.factory._MODEL_CONFIGS[CLBP_MODEL_NAME] = CLBP_MODEL_CONFIG


class CLBPEncoder(nn.Module):
    """
    CLBP 编码器封装类
    - 加载预训练的 CLBP 模型
    - 提供时序编码和文本编码接口
    - 冻结所有参数
    """
    def __init__(self, ckpt_path: str, device: str = 'cuda', seq_len: int = 40):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.embed_dim = CLBP_MODEL_CONFIG['embed_dim']

        print(f"[CLBPEncoder] 正在加载 CLBP 模型: {ckpt_path}")
        
        # 创建模型结构
        self.model, _, _ = open_clip.create_model_and_transforms(
            CLBP_MODEL_NAME,
            pretrained=None,
            device='cpu',  # 先在 CPU 加载
            force_timeseries_seq_len=seq_len,
        )
        
        # 加载权重
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # 处理 'module.' 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[CLBPEncoder] 权重加载成功!")
        else:
            raise FileNotFoundError(f"CLBP checkpoint not found: {ckpt_path}")

        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 获取 tokenizer
        self.tokenizer = open_clip.get_tokenizer(CLBP_MODEL_NAME)
        
        print(f"[CLBPEncoder] 初始化完成, embed_dim={self.embed_dim}")

    @torch.no_grad()
    def encode_timeseries(self, ts_tensor: torch.Tensor) -> torch.Tensor:
        """
        编码时序数据
        Args:
            ts_tensor: (batch, seq_len, num_features) 标准化后的时序数据
        Returns:
            features: (batch, embed_dim) 归一化后的时序特征
        """
        ts_tensor = ts_tensor.to(self.device)
        features = self.model.encode_timeseries(ts_tensor, normalize=True)
        return features

    @torch.no_grad()
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        编码文本列表
        Args:
            texts: 文本字符串列表
        Returns:
            features: (num_texts, embed_dim) 归一化后的文本特征
        """
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens, normalize=True)
        return features
    
    def compute_soh_distribution(
        self, 
        ts_features: torch.Tensor, 
        soh_text_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        计算时序特征与 SOH 文本特征的相似度分布
        Args:
            ts_features: (batch, embed_dim) 时序特征
            soh_text_features: (num_soh_bins, embed_dim) SOH 文本特征
            temperature: softmax 温度
        Returns:
            soh_distribution: (batch, num_soh_bins) SOH 概率分布
        """
        # 计算余弦相似度
        similarity = ts_features @ soh_text_features.T  # (batch, num_soh_bins)
        # 转换为概率分布
        soh_distribution = F.softmax(similarity / temperature, dim=-1)
        return soh_distribution