"""
BatteryGPT 不确定性量化模块
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class UncertaintyResult:
    clbp_entropy_raw: torch.Tensor
    clbp_entropy_norm: torch.Tensor
    llm_entropy_per_step: torch.Tensor
    llm_entropy_numeric_steps: torch.Tensor
    combined_uncertainty: torch.Tensor
    num_numeric_steps: torch.Tensor


def get_numerical_token_ids(tokenizer) -> torch.Tensor:
    candidate_strings = (
        [str(i) for i in range(10)]
        + ["."]
        + [f"{i}" for i in range(10)]
        + [f"0.{i:02d}" for i in range(0, 100, 5)]
        + [f"0.{i}" for i in range(1000)]
    )

    ids = set()
    for s in candidate_strings:
        try:
            encoded = tokenizer.encode(s, add_special_tokens=False)
            ids.update(encoded)
        except Exception:
            pass

    if not ids:
        vocab = tokenizer.get_vocab()
        for token_str, tid in vocab.items():
            clean = token_str.replace("▁", "").replace("Ġ", "").strip()
            if clean.replace(".", "").replace("%", "").isdigit():
                ids.add(tid)

    return torch.tensor(sorted(ids), dtype=torch.long)


def _shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    safe_probs = probs.clamp(min=eps)
    return -(probs * safe_probs.log()).sum(dim=-1)


def compute_clbp_entropy(
    soh_distribution: torch.Tensor,
    soh_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_bins = soh_distribution.shape

    row_sum = soh_distribution.sum(dim=-1)
    already_prob = (row_sum - 1.0).abs().max() < 0.01

    if already_prob:
        probs = soh_distribution.float().clamp(min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        probs = F.softmax(soh_distribution.float(), dim=-1)

    entropy_raw = _shannon_entropy(probs)
    entropy_norm = entropy_raw / math.log(num_bins)
    return entropy_raw, entropy_norm


def compute_llm_numerical_entropy(
    scores: List[torch.Tensor],
    generated_ids: Optional[torch.Tensor],
    numerical_token_ids: torch.Tensor,
    min_numeric_prob_threshold: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not scores or len(scores) == 0:
        if generated_ids is not None:
            device = generated_ids.device
            B = generated_ids.shape[0]
        else:
            device = numerical_token_ids.device if numerical_token_ids.numel() > 0 else torch.device("cpu")
            B = 1
        return (
            torch.full((B, 1), -1.0, dtype=torch.float32, device=device),
            torch.full((B,), float("nan"), dtype=torch.float32, device=device),
            torch.zeros(B, dtype=torch.float32, device=device),
        )

    T = len(scores)
    device = scores[0].device
    B = scores[0].shape[0]

    numerical_token_ids = numerical_token_ids.to(device=device, dtype=torch.long)
    num_id_set = set(numerical_token_ids.detach().cpu().tolist())

    entropy_per_step = torch.full((B, T), -1.0, dtype=torch.float32, device=device)
    numeric_step_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for t, step_logits in enumerate(scores):
        step_logits = step_logits.float().to(device)
        probs = F.softmax(step_logits, dim=-1)

        if generated_ids is not None and t < generated_ids.shape[1]:
            gen_token_ids = generated_ids[:, t].to(device)
            is_numeric_step = torch.tensor(
                [int(tid.item()) in num_id_set for tid in gen_token_ids],
                dtype=torch.bool,
                device=device
            )
        else:
            if numerical_token_ids.numel() == 0:
                is_numeric_step = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                numeric_prob_sum = probs[:, numerical_token_ids].sum(dim=-1)
                is_numeric_step = numeric_prob_sum > min_numeric_prob_threshold

        step_entropy = _shannon_entropy(probs)

        entropy_per_step[:, t] = torch.where(
            is_numeric_step, step_entropy, torch.full_like(step_entropy, -1.0)
        )
        numeric_step_mask[:, t] = is_numeric_step

    num_numeric_steps = numeric_step_mask.sum(dim=-1).float()
    entropy_sum = (entropy_per_step * numeric_step_mask.float()).sum(dim=-1)
    entropy_numeric_mean = torch.where(
        num_numeric_steps > 0,
        entropy_sum / num_numeric_steps.clamp(min=1.0),
        torch.full_like(entropy_sum, float("nan"))
    )

    return entropy_per_step, entropy_numeric_mean, num_numeric_steps


def compute_generation_uncertainty(
    soh_distribution: torch.Tensor,
    soh_values: Optional[torch.Tensor],
    scores: Optional[List[torch.Tensor]],
    generated_ids: Optional[torch.Tensor],
    numerical_token_ids: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    llm_entropy_max_nats: Optional[float] = None,
) -> UncertaintyResult:
    device = soh_distribution.device

    clbp_entropy_raw, clbp_entropy_norm = compute_clbp_entropy(soh_distribution, soh_values)
    clbp_entropy_raw = clbp_entropy_raw.to(device)
    clbp_entropy_norm = clbp_entropy_norm.to(device)

    numerical_token_ids = numerical_token_ids.to(device=device, dtype=torch.long)

    if scores is not None and len(scores) > 0:
        entropy_per_step, llm_entropy_numeric, num_numeric_steps = compute_llm_numerical_entropy(
            scores=scores,
            generated_ids=generated_ids,
            numerical_token_ids=numerical_token_ids,
        )
        entropy_per_step = entropy_per_step.to(device)
        llm_entropy_numeric = llm_entropy_numeric.to(device)
        num_numeric_steps = num_numeric_steps.to(device)

        if llm_entropy_max_nats is None:
            llm_entropy_max_nats = math.log(scores[0].shape[-1])

        llm_entropy_norm = llm_entropy_numeric / llm_entropy_max_nats
        llm_entropy_norm_safe = torch.where(
            torch.isnan(llm_entropy_norm),
            torch.zeros_like(llm_entropy_norm, device=device),
            llm_entropy_norm.clamp(0.0, 1.0)
        )
    else:
        B = soh_distribution.shape[0]
        entropy_per_step = torch.full((B, 1), -1.0, dtype=torch.float32, device=device)
        llm_entropy_numeric = torch.full((B,), float("nan"), dtype=torch.float32, device=device)
        num_numeric_steps = torch.zeros(B, dtype=torch.float32, device=device)
        llm_entropy_norm_safe = torch.zeros(B, dtype=torch.float32, device=device)
        alpha, beta = 1.0, 0.0

    has_numeric = ~torch.isnan(llm_entropy_numeric)
    effective_alpha = torch.where(
        has_numeric,
        torch.full_like(clbp_entropy_norm, float(alpha), device=device),
        torch.ones_like(clbp_entropy_norm, device=device),
    )
    effective_beta = torch.where(
        has_numeric,
        torch.full_like(clbp_entropy_norm, float(beta), device=device),
        torch.zeros_like(clbp_entropy_norm, device=device),
    )

    combined = (
        effective_alpha * clbp_entropy_norm.clamp(0.0, 1.0)
        + effective_beta * llm_entropy_norm_safe
    )

    return UncertaintyResult(
        clbp_entropy_raw=clbp_entropy_raw.detach().cpu(),
        clbp_entropy_norm=clbp_entropy_norm.detach().cpu(),
        llm_entropy_per_step=entropy_per_step.detach().cpu(),
        llm_entropy_numeric_steps=llm_entropy_numeric.detach().cpu(),
        combined_uncertainty=combined.detach().cpu(),
        num_numeric_steps=num_numeric_steps.detach().cpu(),
    )