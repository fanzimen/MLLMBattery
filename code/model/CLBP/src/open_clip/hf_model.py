""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType
import logging
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]

class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            # 关键修复：不再传 add_pooling_layer 给 AutoModel（Qwen2 等不支持）
            if pretrained:
                self.transformer = AutoModel.from_pretrained(model_name_or_path)
                # self.transformer.gradient_checkpointing_enable()
                logging.info(f"✅ Loaded pretrained model from {model_name_or_path}")
            else:
                self.transformer = AutoModel.from_config(self.config)
                logging.warning(f"⚠️ Initialized model randomly from config")
            # encoder-decoder 架构只取 encoder
            if getattr(self.config, "is_encoder_decoder", False):
                self.transformer = self.transformer.encoder
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
            if getattr(self.config, "is_encoder_decoder", False):
                self.transformer = self.transformer.encoder
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            # if model_name_or_path.startswith("Vicuna"):
            #     hidden_size1 =(d_model + output_dim) // 3
            #     hidden_size2 = (d_model + output_dim) * 2 // 3
            #     self.proj = nn.Sequential(
            #         nn.Linear(d_model, hidden_size1, bias=False),
            #         nn.GELU(),
            #         nn.Dropout(0.1),
            #         nn.Linear(hidden_size1, hidden_size2, bias=False),
            #         nn.GELU(),
            #         nn.Dropout(0.1),
            #         nn.Linear(hidden_size2, output_dim, bias=False),
            #     )
            # else:
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )
        try:
            param_dtype = next(self.transformer.parameters()).dtype
            if hasattr(self, "proj") and self.proj is not None:
                self.proj = self.proj.to(param_dtype)
        except StopIteration:
            pass
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
            # 对于 LLaMA,需要额外设置
            if hasattr(self.transformer.config, 'use_cache'):
                self.transformer.config.use_cache = False

            
# class HFTextEncoder(nn.Module):
#     """HuggingFace model adapter"""
#     output_tokens: torch.jit.Final[bool]

#     def __init__(
#             self,
#             model_name_or_path: str,
#             output_dim: int,
#             config: PretrainedConfig = None,
#             pooler_type: str = None,
#             proj_type: str = None,
#             pretrained: bool = True,
#             output_tokens: bool = False,
#     ):
#         super().__init__()
#         self.output_tokens = output_tokens
#         self.output_dim = output_dim

#         # TODO: find better way to get this information
#         uses_transformer_pooler = (pooler_type == "cls_pooler")

#         if transformers is None:
#             raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
#         if config is None:
#             self.config = AutoConfig.from_pretrained(model_name_or_path)
#             create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
#                 AutoModel.from_config, self.config)
#             # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
#             if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
#                 self.transformer = create_func(model_args)
#                 self.transformer = self.transformer.encoder
#             else:
#                 self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
#         else:
#             self.config = config
#             self.transformer = AutoModel.from_config(config)
#         if pooler_type is None:  # get default arch pooler
#             pooler_type = (arch_dict[self.config.model_type]["pooler"])

#         # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
#         self.vocab_size = getattr(self.config, 'vocab_size', 0)
#         self.context_length = getattr(self.config, 'max_position_embeddings', 0)

#         self.pooler = _POOLERS[pooler_type]()

#         d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
#         if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
#             self.proj = nn.Identity()
#         elif proj_type == 'linear':
#             self.proj = nn.Linear(d_model, output_dim, bias=False)
#         elif proj_type == 'mlp':
#             hidden_size = (d_model + output_dim) // 2
#             self.proj = nn.Sequential(
#                 nn.Linear(d_model, hidden_size, bias=False),
#                 nn.GELU(),
#                 nn.Linear(hidden_size, output_dim, bias=False),
#             )

    def forward(self, x: TensorType):
        # --- [修复] ---
        # 处理 pad_token_id 可能为 None 的情况
        pad_token_id = getattr(self.config, 'pad_token_id', None)
        if pad_token_id is None:
            # 如果没有设置 pad_token_id，使用 0 作为默认值
            pad_token_id = 0
        
        attn_mask = (x != pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
            if type(self.pooler) == ClsPooler 
            else out.last_hidden_state
        )
        
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
