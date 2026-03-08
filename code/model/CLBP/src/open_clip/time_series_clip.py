import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import torch.nn.functional as F


class TimeSeriesTextCLIP(nn.Module):
    """时序数据-文本对齐模型"""
    
    def __init__(
        self,
        time_series_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        output_dict: bool = False,
    ):
        super().__init__()
        self.time_series_encoder = time_series_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.output_dict = output_dict
        
        # 确保两个编码器输出维度一致
        self.embed_dim = embed_dim
        
    def lock_time_series_tower(self, unlocked_groups=0, freeze_bn_stats=True):
        """锁定时序编码器参数"""
        # 实现参数冻结逻辑
        for param in self.time_series_encoder.parameters():
            param.requires_grad = False
            
    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """锁定文本编码器参数"""
        # 复用原有文本编码器的锁定逻辑
        pass
        
    def set_grad_checkpointing(self, enable: bool = True):
        """设置梯度检查点"""
        if hasattr(self.time_series_encoder, 'set_grad_checkpointing'):
            self.time_series_encoder.set_grad_checkpointing(enable)
        if hasattr(self.text_encoder, 'set_grad_checkpointing'):
            self.text_encoder.set_grad_checkpointing(enable)
    
    def encode_time_series(self, time_series: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """编码时序数据"""
        features = self.time_series_encoder(time_series)
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """编码文本数据"""
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1) if normalize else features
    
    def forward(
        self, 
        time_series: Optional[torch.Tensor] = None, 
        text: Optional[torch.Tensor] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        
        time_series_features = self.encode_time_series(time_series, normalize=True) if time_series is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        
        if self.output_dict:
            return {
                "time_series_features": time_series_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        
        return time_series_features, text_features, self.logit_scale.exp()