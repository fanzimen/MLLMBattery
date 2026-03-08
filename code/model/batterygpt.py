"""
BatteryGPT 主模型
整合 CLBP 时序编码器、SOHPromptLearner 和 Vicuna LLM。

[修复] generate() return_scores=True 时：
  - outputs.sequences 在 inputs_embeds 模式下只含新生成token，不含prompt
  - 原代码 outputs.sequences[:, prompt_token_len:] 导致 generated_ids 为空 Tensor[1,0]
  - 修复：直接使用 outputs.sequences，无需切片
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import rnn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaTokenizer, StoppingCriteria, StoppingCriteriaList

from .clbp_encoder import CLBPEncoder
from .BatteryGPT_models import SOHPromptLearner, SOHRegressionHead, LinearLayer

USE_OFFICIAL_LLAMA = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_cache=True.*")


# ============ 辅助函数 ============

def build_one_instance(tokenizer, conversation):
    """构建单个对话实例的 input_ids 和 target_ids"""
    input_ids, target_ids = [], []
    for i, turn in enumerate(conversation):
        role = turn['from']
        text = turn['value']
        if role == 'human':
            text = text + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)
        elif role == 'gpt':
            text = text + '\n###'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += one_input_id
    return input_ids, target_ids


def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    """处理一个 batch 的对话"""
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

    input_ids = rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(
        batch_target_ids, batch_first=True, padding_value=-100)

    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    return input_ids, target_ids, attention_mask


class StoppingCriteriaSub(StoppingCriteria):
    """自定义停止条件"""
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        return stop_count >= self.ENCOUNTERS


# ============ 主模型 ============

PROMPT_START = '### Human: <TS>'


class BatteryGPTModel(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.args = args

        clbp_ckpt_path = args['clbp_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        self.max_tgt_len = args.get('max_tgt_len', 128)
        self.seq_len = args.get('seq_len', 40)
        use_official = args.get('use_official_llama', USE_OFFICIAL_LLAMA)

        self.soh_min = args.get('soh_min', 0.700)
        self.soh_max = args.get('soh_max', 1.000)
        self.soh_step = args.get('soh_step', 0.001)
        self.soh_temperature = args.get('soh_temperature', 0.07)

        self.lora_r = args.get('lora_r', 8)
        self.lora_alpha = args.get('lora_alpha', 16)
        self.lora_dropout = args.get('lora_dropout', 0.1)

        self.use_soh_loss = args.get('use_soh_loss', True)
        self.soh_loss_weight = args.get('soh_loss_weight', 1.0)

        self.device = torch.cuda.current_device()

        rank = dist.get_rank() if dist.is_initialized() else 0

        # ============ 1. CLBP 时序编码器 (冻结) ============
        if rank == 0:
            print(f'[BatteryGPT] 初始化 CLBP 编码器: {clbp_ckpt_path}')
        self.clbp_encoder = CLBPEncoder(
            ckpt_path=clbp_ckpt_path,
            device='cuda',
            seq_len=self.seq_len
        )
        self.ts_embed_dim = self.clbp_encoder.embed_dim  # 1024

        self._prepare_soh_text_features()

        # ============ 2. SOH Prompt Learner (可训练) ============
        if rank == 0:
            print('[BatteryGPT] 初始化 SOHPromptLearner')
        self.prompt_learner = SOHPromptLearner(
            num_soh_bins=self.num_soh_bins,
            llm_embed_dim=4096,
            num_prompts=8
        )

        # ============ 3. SOH 回归头 (可选) ============
        if self.use_soh_loss:
            self.soh_regression_head = SOHRegressionHead(
                input_dim=self.ts_embed_dim,
                hidden_dim=256
            )
            if rank == 0:
                print(f'[BatteryGPT] 初始化 SOH 回归头 (权重={self.soh_loss_weight})')
        else:
            self.soh_regression_head = None
            if rank == 0:
                print('[BatteryGPT] ⚠️ SOH Loss 已禁用')

        # ============ 4. 投影层: CLBP -> LLM ============
        self.ts_proj = nn.Linear(self.ts_embed_dim, 4096)

        # ============ 5. Vicuna LLM + LoRA ============
        if rank == 0:
            print(f"[BatteryGPT] 初始化 Vicuna LLM: {vicuna_ckpt_path}")

        if use_official:
            from transformers import LlamaForCausalLM
        else:
            from .modeling_llama import LlamaForCausalLM

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)

        if hasattr(self.llama_model, "gradient_checkpointing_enable"):
            self.llama_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        for param in self.llama_model.parameters():
            param.requires_grad = False

        if hasattr(self.llama_model, "enable_input_require_grads"):
            self.llama_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.llama_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)

        if rank == 0:
            self.llama_model.print_trainable_parameters()

        if hasattr(self.llama_model, 'base_model'):
            for param in self.llama_model.base_model.model.model.embed_tokens.parameters():
                param.requires_grad = False

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

        for param in self.prompt_learner.parameters():
            param.requires_grad = True
        if self.use_soh_loss and self.soh_regression_head is not None:
            for param in self.soh_regression_head.parameters():
                param.requires_grad = True
        for param in self.ts_proj.parameters():
            param.requires_grad = True

        if rank == 0:
            print('[BatteryGPT] 初始化完成!')

    def _prepare_soh_text_features(self):
        """预计算 SOH 候选文本的特征"""
        soh_values = np.arange(
            self.soh_max, self.soh_min - self.soh_step / 2, -self.soh_step)
        self.soh_values = soh_values
        self.num_soh_bins = len(soh_values)

        soh_texts = [f"SOH={v:.3f}" for v in soh_values]

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f'[BatteryGPT] 预计算 {self.num_soh_bins} 个 SOH 文本特征')

        with torch.no_grad():
            self.soh_text_features = self.clbp_encoder.encode_text(soh_texts)

        self.register_buffer('soh_text_features_buffer', self.soh_text_features)
        self.register_buffer(
            'soh_values_buffer', torch.from_numpy(soh_values).float())

    def encode_timeseries(self, ts_tensor: torch.Tensor):
        """
        编码时序数据
        Returns:
            ts_embeds:        (batch, 1, 4096)
            ts_features:      (batch, 1024)
            soh_distribution: (batch, num_soh_bins)
        """
        ts_features = self.clbp_encoder.encode_timeseries(ts_tensor)
        soh_distribution = self.clbp_encoder.compute_soh_distribution(
            ts_features,
            self.soh_text_features_buffer,
            temperature=self.soh_temperature
        )
        ts_embeds = self.ts_proj(ts_features).unsqueeze(1)
        return ts_embeds, ts_features, soh_distribution

    def prompt_wrap(self, ts_embeds, input_ids, target_ids, attention_mask, soh_prompts):
        """
        拼接序列：
        [BOS] + [### Human: <TS>] + [TS_Embed] + [</TS>] + [SOH_Prompts] + [Text]
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        batch_size = ts_embeds.shape[0]

        p_before_tokens = self.llama_tokenizer(
            PROMPT_START, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_tokens.input_ids).expand(batch_size, -1, -1)

        p_middle_tokens = self.llama_tokenizer(
            '</TS> ', return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_middle_embeds = self.llama_model.model.model.embed_tokens(
            p_middle_tokens.input_ids).expand(batch_size, -1, -1)

        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids)

        bos = torch.ones(
            [batch_size, 1], dtype=torch.long, device=self.device
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([
            bos_embeds, p_before_embeds, ts_embeds,
            p_middle_embeds, soh_prompts, p_after_embeds
        ], dim=1)

        prefix_len = (1 + p_before_embeds.size(1) + 1 +
                      p_middle_embeds.size(1) + soh_prompts.size(1))
        empty_targets = torch.full(
            [batch_size, prefix_len], -100, dtype=torch.long, device=self.device)
        targets = torch.cat([empty_targets, target_ids], dim=1)

        atts_prefix = torch.ones(
            [batch_size, prefix_len], dtype=torch.long, device=self.device)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)

        return inputs_embeds, targets, attention_mask

    def forward(self, inputs):
        """
        训练前向传播
        Returns: total_loss, gen_acc, soh_pred, soh_loss
        """
        ts_embeds, ts_features, soh_distribution = self.encode_timeseries(
            inputs['timeseries'])

        if self.use_soh_loss and self.soh_regression_head is not None:
            soh_pred = self.soh_regression_head(ts_features)
        else:
            soh_pred = torch.zeros(ts_embeds.shape[0], 1, device=self.device)

        soh_prompts = self.prompt_learner(soh_distribution)

        output_texts = inputs['texts']
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len)

        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            ts_embeds, input_ids, target_ids, attention_mask, soh_prompts)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
        )
        text_loss = outputs.loss

        soh_loss = torch.tensor(0.0, device=self.device)
        if (self.use_soh_loss and 'soh_labels' in inputs
                and self.soh_regression_head is not None):
            soh_labels = inputs['soh_labels'].to(self.device).view(-1, 1)
            soh_loss = F.mse_loss(soh_pred, soh_labels) * self.soh_loss_weight

        total_loss = text_loss + soh_loss

        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1e-8)

        return total_loss, gen_acc, soh_pred.detach(), soh_loss.detach()

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=64, temperature=0.1, top_p=0.9, return_scores=False):
        """
        推理生成

        Args:
            inputs: dict {'timeseries': Tensor(batch,seq,feat), 'prompt': str|List[str]}
            max_new_tokens: 最大新生成token数
            temperature: 采样温度
            top_p: nucleus sampling
            return_scores: 是否返回每步logits和generated_ids（用于LLM熵计算）

        Returns:
            return_scores=False:
                response(str|List[str]), soh_pred(float|ndarray), soh_dist(ndarray)
            return_scores=True:
                response, soh_pred, soh_dist,
                scores(tuple of Tensor[batch,vocab]),
                generated_ids(Tensor[batch, num_new_tokens])

        [修复说明]
            HuggingFace generate() 在 inputs_embeds 模式下：
              - return_dict_in_generate=False: 返回 Tensor[batch, num_new_tokens]
              - return_dict_in_generate=True:  outputs.sequences shape=[batch, num_new_tokens]
                                               (只含新生成token，不含prompt)
            原代码错误地做了 outputs.sequences[:, prompt_token_len:] 切片，
            导致 generated_ids 变成空 Tensor[batch, 0]，LLM熵无法计算。
            修复：直接使用 outputs.sequences，不做切片。
        """
        # 1. 编码时序
        ts_embeds, ts_features, soh_distribution = self.encode_timeseries(
            inputs['timeseries'])
        batch_size = ts_embeds.shape[0]

        # 2. 回归预测
        if self.use_soh_loss and self.soh_regression_head is not None:
            soh_pred_tensor = self.soh_regression_head(ts_features)
        else:
            soh_pred_tensor = torch.zeros(batch_size, 1, device=self.device)

        soh_pred = (soh_pred_tensor.item() if batch_size == 1
                    else soh_pred_tensor.squeeze().cpu().numpy())

        # 3. SOH Prompts
        soh_prompts = self.prompt_learner(soh_distribution)

        # 4. 文本输入
        prompts = inputs['prompt']
        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) != batch_size:
            raise ValueError(
                f"Prompt数量({len(prompts)}) != batch_size({batch_size})")

        texts = [p + '\n### Assistant:' for p in prompts]

        # 5. 构建 Embeddings
        p_after_tokens = self.llama_tokenizer(
            texts, return_tensors="pt",
            add_special_tokens=False, padding=True
        ).to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(
            p_after_tokens.input_ids)
        text_attention_mask = p_after_tokens.attention_mask

        def _expand(token_str):
            toks = self.llama_tokenizer(
                token_str, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            return self.llama_model.model.model.embed_tokens(
                toks.input_ids).expand(batch_size, -1, -1)

        p_before_embeds = _expand(PROMPT_START)
        p_middle_embeds = _expand('</TS> ')

        bos = torch.ones(
            [batch_size, 1], dtype=torch.long, device=self.device
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([
            bos_embeds, p_before_embeds, ts_embeds,
            p_middle_embeds, soh_prompts, p_after_embeds
        ], dim=1)

        # 6. Attention Mask
        prefix_len = (bos_embeds.shape[1] + p_before_embeds.shape[1] +
                      ts_embeds.shape[1] + p_middle_embeds.shape[1] +
                      soh_prompts.shape[1])
        prefix_mask = torch.ones(
            (batch_size, prefix_len), dtype=torch.long, device=self.device)
        attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

        # 7. 生成
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            output_scores=return_scores,
            return_dict_in_generate=return_scores,
        )

        # ============================================================
        # [修复] inputs_embeds 模式下 outputs.sequences 解包
        #
        # HuggingFace generate() 行为（inputs_embeds模式）：
        #   return_dict_in_generate=False:
        #     outputs → Tensor[batch, num_new_tokens]  (只含新生成token)
        #   return_dict_in_generate=True:
        #     outputs.sequences → Tensor[batch, num_new_tokens]  (同上，只含新token)
        #     outputs.scores   → tuple of Tensor[batch, vocab], len=num_new_tokens
        #
        # 注意：与 input_ids 模式不同！input_ids 模式下 sequences 含完整序列
        #       (prompt + 新token)，需要切片。
        #       inputs_embeds 模式下 sequences 只含新token，直接使用即可。
        #
        # 原代码 BUG：
        #   prompt_token_len = inputs_embeds.shape[1]  # e.g. 350
        #   generated_ids = outputs.sequences[:, 350:] # → Tensor[batch, 0] ← 空！
        #
        # 修复：直接使用 outputs.sequences
        # ============================================================
        if return_scores:
            gen_scores = outputs.scores       # tuple of Tensor[batch, vocab]
            generated_ids = outputs.sequences # Tensor[batch, num_new_tokens] ← 直接用
            raw_outputs = generated_ids
        else:
            gen_scores = None
            generated_ids = None
            raw_outputs = outputs             # Tensor[batch, num_new_tokens]

        # 8. 解码
        response_list = []
        for output in raw_outputs:
            output_text = self.llama_tokenizer.decode(
                output, skip_special_tokens=True)
            if '### Assistant:' in output_text:
                response = output_text.split('### Assistant:')[-1].strip()
                response = response.split('###')[0].strip()
            else:
                response = output_text.strip()
            response_list.append(response)

        # 9. 返回
        soh_dist_np = soh_distribution.cpu().numpy()
        if return_scores:
            if batch_size == 1:
                return (response_list[0], soh_pred, soh_dist_np,
                        gen_scores, generated_ids)
            return response_list, soh_pred, soh_dist_np, gen_scores, generated_ids
        else:
            if batch_size == 1:
                return response_list[0], soh_pred, soh_dist_np
            return response_list, soh_pred, soh_dist_np