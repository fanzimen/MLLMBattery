"""
TimeSeries Transformer Model
- 新增模块，用于处理多维时序数据
- 您需要根据您的具体数据格式和任务需求，重点关注和修改 `__init__` 中的输入层 (self.input_proj) 和位置编码 (self.positional_embedding)
"""
import torch
from torch import nn
from typing import Optional

from .transformer import Transformer


class TimeSeriesTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            seq_len: int = 256,
            dropout: float = 0.1,
    ):
        """
        初始化时序 Transformer 模型

        Args:
            input_dim (int): 输入时序数据的维度 (channels/features)。
            patch_size (int): 每个 "patch" 包含的时间步数。类似于 ViT 中的 patch size。
                             如果您的数据不需要 patch 化，可以设为 1。
            width (int): Transformer 内部的特征维度。
            layers (int): Transformer 层的数量。
            heads (int): 多头注意力机制的头数。
            mlp_ratio (float): MLP 层的扩展比例。
            output_dim (int): 最终输出特征的维度，应与文本塔的输出维度一致。
            act_layer (nn.Module): 激活函数。
            norm_layer (nn.Module): 归一化函数。
            seq_len (int): 输入时序数据的总长度。
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.width = width  #Transformer 内部的工作维度。这是一个超参数，决定了模型内部表示的丰富程度，通常是 512, 768 等。
        self.layers = layers  # Transformer 层的数量。更多的层可以捕捉更复杂的模式，但也增加了计算成本和过拟合风险。
        self.heads = heads  # 多头注意力机制的头数。更多的头可以让模型在不同的子空间中关注不同的信息。width 必须能被 heads 整除
        self.output_dim = output_dim # 最终输出特征的维度，应与文本塔的输出维度一致，以便进行对比学习。
        self.seq_len = seq_len # 输入时序数据的总长度
        
        # 计算 patch 数量
        self.num_patches = (seq_len + patch_size - 1) // patch_size  # 向上取整

        # --- [核心修改点 1] ---
        # 输入投影层 (Input Projection)
        # 将输入数据 (batch_size, seq_len, input_dim) 转换为 Transformer 需要的 (batch_size, num_patches, width)
        # 这里的实现方式有多种，取决于您的数据特性：
        # 1. 线性投影：如果 patch_size=1，可以直接用一个线性层。
        # 2. 卷积投影：类似于 ViT，可以用一个一维卷积层来做 "patch" 嵌入。
        #
        # 这里我们使用一维卷积作为示例，它能更好地捕捉局部时间依赖性。

        '''
        这是模型的第一步，也是至关重要的一步。它的作用是将输入的“时间片”转换（或称为“嵌入”）为 Transformer 能够处理的固定维度向量。
        为什么用 Conv1d? 将 kernel_size 和 stride 都设置为 patch_size 的一维卷积，可以完美地实现“打补丁”和“嵌入”两个操作。卷积核会在时序数据上以 patch_size 的步长滑动，每次处理一个 patch_size 长度的片段。
        维度变换: 它将一个形状为 (batch_size, input_dim, seq_len) 的输入，转换为 (batch_size, width, num_patches)。in_channels 对应 input_dim,out_channels 对应 Transformer 的工作维度 width。
        '''
        self.input_proj = nn.Conv1d(
            in_channels=input_dim,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token，用于聚合整个序列的信息
        '''
        这是直接从 ViT 借鉴来的一个关键组件。它是一个可学习的向量，其维度与 Transformer 的工作维度 width 相同。在数据送入 Transformer 之前，这个 class_embedding 会被拼接到所有 patch 嵌入序列的最前面。
        它的作用是作为一个“全局信息聚合器”。在经过多层 Transformer 的自注意力计算后，这个 class_embedding 对应的最终输出向量，就被认为是整个时序序列的全局表示
        '''
        # self.class_embedding = nn.Parameter(torch.randn(width))
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.02)
        
        # --- [核心修改点 2] ---
        # 位置编码 (Positional Embedding)
        # 为 class token 和每个 patch 提供位置信息
        # 使用更稳定的初始化方式
        # self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, width))
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, width) * 0.02)

        # --- [新增 2] --- 定义 Dropout 层
        self.pos_drop = nn.Dropout(p=dropout)
        # Transformer 主体
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        # # 最终的归一化和投影层
        # self.ln_post = norm_layer(width)
        # # --- [修正 3] ---
        # # 使用更稳定的初始化方式
        # # self.proj = nn.Parameter(torch.randn(width, output_dim))
        # self.proj = nn.Parameter(torch.randn(width, output_dim) * 0.02)

        
        self.ln_post = norm_layer(width)
        # --- [修正] ---
        # 当 width 和 output_dim 相同时，不再需要投影层
        if width == output_dim:
            self.proj = None
        else:
            self.proj = nn.Parameter(torch.randn(width, output_dim) * 0.02)


    def forward(self, x: torch.Tensor, return_sequence=False):
        """
        前向传播
        
        Args:
            x: [B, T, N] 多变量时序数据
            return_sequence: 如果为 True，返回所有 patch tokens；否则只返回 class token
        
        Returns:
            如果 return_sequence=True: [B, num_patches, width]
            如果 return_sequence=False: [B, width]
        """
        # 1. 输入投影（保持不变）
        x = x.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]
        x = self.input_proj(x)  # [B, N, T] -> [B, width, num_patches]
        x = x.permute(0, 2, 1)  # [B, width, num_patches] -> [B, num_patches, width]
        
        # 2. 添加 Class Token（保持不变）
        batch_size = x.shape[0]
        class_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)  # [B, num_patches+1, width]
        
        # 3. 添加位置编码（保持不变）
        x = x + self.positional_embedding.to(x.dtype)
        x = self.pos_drop(x)
        
        # 4. Transformer（保持不变）
        x = self.transformer(x)  # [B, num_patches+1, width]
        
        # --- [核心修改] ---
        # 5. 根据 return_sequence 决定返回格式
        if return_sequence:
            # 返回所有 patch tokens（不包括 class token）
            # 这对于需要序列表示的下游任务（如 TimeLLM）很有用
            # x = x[:, 1:, :]  # [B, num_patches, width]
            return x
        else:
            # 返回 class token（用于对比学习或分类）
            x = x[:, 0]  # [B, width]
            x = self.ln_post(x)
            if self.proj is not None:
                x = x @ self.proj
            return x

    def lock(self, unlocked_layers=0):
        """
        锁定模型参数，使其不可训练。可以只解锁最后几层。
        """
        for param in self.parameters():
            param.requires_grad = False
        
        if unlocked_layers > 0:
            # 解锁投影层和最后的 LayerNorm
            if self.proj is not None:
                self.proj.requires_grad = True
            for param in self.ln_post.parameters():
                param.requires_grad = True
            
            # 解锁最后 N 层的 Transformer block
            for i in range(1, unlocked_layers + 1):
                if i > len(self.transformer.resblocks):
                    break
                for param in self.transformer.resblocks[-i].parameters():
                    param.requires_grad = True
