"""
BatteryGPT 评估脚本
[v3] 多GPU并行评估：按电池文件分片，Rank0汇总结果
"""
# ============ 警告过滤 ============
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

import sys
import os
import re
import math
import argparse
import yaml
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from scipy import stats

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules["torchvision.transforms.functional_tensor"] = F
    except ImportError:
        pass

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.batterygpt import BatteryGPTModel
from utils.uncertainty import (
    get_numerical_token_ids,
    compute_generation_uncertainty,
)

try:
    from datasets.battery_soh_dataset import (
        SOH_QUESTIONS_DYNAMIC,
        SOH_QUESTIONS_DYNAMIC_CN,
        SOH_QUESTIONS_BASIC,
        SOH_QUESTIONS_CN,
        select_top_shared_features
    )
except ImportError:
    tqdm.write("⚠️ Warning: Could not import templates from datasets.battery_soh_dataset.")
    SOH_QUESTIONS_DYNAMIC = [
        "Battery: {battery_type}. {feat1_name}: {feat1_val:.4f} (slope: {feat1_slope:.4f}), "
        "{feat2_name}: {feat2_val:.4f} (slope: {feat2_slope:.4f}). Estimate SOH."
    ]
    SOH_QUESTIONS_DYNAMIC_CN = [
        "{battery_type}。{feat1_name}: {feat1_val:.4f} (斜率: {feat1_slope:.4f})，"
        "{feat2_name}: {feat2_val:.4f} (斜率: {feat2_slope:.4f})。"
    ]
    SOH_QUESTIONS_BASIC = ["What is the current SOH?"]
    SOH_QUESTIONS_CN = ["当前SOH是多少？"]

    def select_top_shared_features(*args, **kwargs):
        return ['voltage', 'current']


# ============================================================
# 分布式工具函数
# ============================================================

def setup_distributed():
    """初始化分布式环境（torchrun 自动设置环境变量）"""
    if 'RANK' not in os.environ:
        # 单进程模式
        return 0, 1, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log_main(msg):
    """只在 Rank0 打印"""
    if is_main_process():
        tqdm.write(msg)


def gather_battery_results(local_results: dict, rank: int, world_size: int) -> dict:
    """
    将各 rank 的 battery_results dict 汇总到 rank0。
    使用 pickle 序列化 + broadcast_object_list。
    """
    if world_size == 1:
        return local_results

    # 每个 rank 把自己的结果放到列表里
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)

    if rank != 0:
        return {}

    merged = {}
    for res in all_results:
        if res:
            merged.update(res)
    return merged


def gather_array(local_arr: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """汇总各 rank 的 1D numpy array 到 rank0（顺序不保证，需配合索引）"""
    if world_size == 1:
        return local_arr

    all_arrays = [None] * world_size
    dist.all_gather_object(all_arrays, local_arr)
    if rank != 0:
        return np.array([])
    return np.concatenate([a for a in all_arrays if a is not None and len(a) > 0])


# ============================================================
# generate() 输出解析
# ============================================================

def _parse_generate_output_batch(gen_out):
    """
    批量解析 generate 输出
    返回:
      texts    : List[str]
      soh_dist : np.ndarray|None, shape [B, num_bins] 或 [num_bins]
      scores   : tuple|None
      gen_ids  : Tensor|None, shape [B, T] 或 [T]
    """
    if not isinstance(gen_out, (tuple, list)):
        return [str(gen_out)], None, None, None

    n = len(gen_out)
    text_obj = gen_out[0]
    if isinstance(text_obj, str):
        texts = [text_obj]
    elif isinstance(text_obj, list):
        texts = text_obj
    else:
        texts = [str(text_obj)]

    soh_dist = gen_out[2] if n >= 3 else None
    scores   = gen_out[3] if n >= 5 else None
    gen_ids  = gen_out[4] if n >= 5 else None

    if isinstance(soh_dist, torch.Tensor):
        soh_dist = soh_dist.detach().cpu().numpy()

    return texts, soh_dist, scores, gen_ids


# ============ 诊断函数 ============

def diagnose_generate_output(model_unwrapped, ts_single, prompt_single):
    test_input = {'timeseries': ts_single, 'prompt': prompt_single}
    tqdm.write("\n=== Diagnosing generate() output ===")
    try:
        out = model_unwrapped.generate(test_input, max_new_tokens=20, temperature=0.1)
        tqdm.write(f"Without return_scores: type={type(out)}, "
                   f"len={len(out) if isinstance(out, (list, tuple)) else 'N/A'}")
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                preview = (str(o)[:80] if not isinstance(o, torch.Tensor)
                           else f'Tensor{list(o.shape)}')
                tqdm.write(f"  [{i}]: type={type(o).__name__}, preview={preview}")
    except Exception as e:
        tqdm.write(f"  [No return_scores] Error: {e}")

    try:
        out2 = model_unwrapped.generate(
            test_input, max_new_tokens=20, temperature=0.1, return_scores=True)
        tqdm.write(f"\nWith return_scores=True: type={type(out2)}, "
                   f"len={len(out2) if isinstance(out2, (list, tuple)) else 'N/A'}")
        if isinstance(out2, (list, tuple)):
            for i, o in enumerate(out2):
                preview = (str(o)[:80] if not isinstance(o, torch.Tensor)
                           else f'Tensor{list(o.shape)}')
                tqdm.write(f"  [{i}]: type={type(o).__name__}, preview={preview}")
        if isinstance(out2, tuple) and len(out2) >= 5:
            gen_ids = out2[4]
            if isinstance(gen_ids, torch.Tensor) and gen_ids.shape[-1] > 0:
                tqdm.write(f"  ✅ LLM熵修复验证通过: generated_ids shape={list(gen_ids.shape)}")
            else:
                tqdm.write(f"  ❌ LLM熵修复未生效: generated_ids={gen_ids}")
    except Exception as e:
        tqdm.write(f"  [return_scores=True] Error: {e}")
    tqdm.write("=== End Diagnosis ===\n")


# ============ 不确定性计算 ============

def _compute_uncertainty_batch(
    soh_dists, scores, gen_ids,
    model_unwrapped, numerical_token_ids,
    device, alpha=0.5, beta=0.5
):
    """批量计算不确定性"""
    n             = len(soh_dists)
    clbp_list     = [float('nan')] * n
    llm_list      = [float('nan')] * n
    combined_list = [float('nan')] * n

    valid_idx = [i for i, sd in enumerate(soh_dists) if sd is not None]
    if not valid_idx:
        return clbp_list, llm_list, combined_list

    # ---- CLBP batch ----
    stacked   = np.stack([np.array(soh_dists[i]).reshape(-1) for i in valid_idx], axis=0)
    sd_tensor = torch.tensor(stacked, dtype=torch.float32, device=device)

    clbp_result = compute_generation_uncertainty(
        soh_distribution=sd_tensor,
        soh_values=getattr(model_unwrapped, 'soh_values_buffer', None),
        scores=None,
        generated_ids=None,
        numerical_token_ids=numerical_token_ids,
        alpha=1.0,
        beta=0.0,
    )
    clbp_vals = clbp_result.clbp_entropy_norm.detach().float().cpu().numpy().reshape(-1)
    for local_i, global_i in enumerate(valid_idx):
        clbp_list[global_i] = float(clbp_vals[local_i])

    # ---- LLM batch ----
    if beta > 0 and scores is not None and gen_ids is not None and len(scores) > 0:
        gi = gen_ids
        if isinstance(gi, torch.Tensor) and gi.dim() == 1:
            gi = gi.unsqueeze(0)

        if isinstance(gi, torch.Tensor) and gi.shape[-1] > 0:
            try:
                llm_result = compute_generation_uncertainty(
                    soh_distribution=sd_tensor,
                    soh_values=getattr(model_unwrapped, 'soh_values_buffer', None),
                    scores=scores,
                    generated_ids=gi,
                    numerical_token_ids=numerical_token_ids,
                    alpha=0.0,
                    beta=1.0,
                )
                raw    = llm_result.llm_entropy_numeric_steps
                raw_np = raw.detach().float().cpu().numpy().reshape(-1)

                vocab_size = scores[0].shape[-1]
                max_llm_H  = math.log(vocab_size + 1e-12)

                for local_i, global_i in enumerate(valid_idx):
                    if local_i < len(raw_np) and np.isfinite(raw_np[local_i]):
                        llm_list[global_i] = min(
                            float(raw_np[local_i]) / (max_llm_H + 1e-10), 1.0)
            except Exception as e:
                tqdm.write(f"⚠️ Batch LLM entropy failed: {e}")

    # ---- combine ----
    for i in valid_idx:
        c = clbp_list[i]
        l = llm_list[i]
        if np.isfinite(c) and np.isfinite(l):
            combined_list[i] = alpha * c + beta * l
        elif np.isfinite(c):
            combined_list[i] = c

    return clbp_list, llm_list, combined_list


# ============ Prompt 生成器 ============

class EvalPromptGenerator:
    def __init__(self, description_folder=None, use_chinese=False,
                 selected_features=None, feature_scaler=None):
        self.use_chinese = use_chinese
        self.description_map = {}
        if description_folder and os.path.exists(description_folder):
            self._load_descriptions(description_folder)

        self.selected_features = selected_features
        if not self.selected_features or len(self.selected_features) < 2:
            tqdm.write("⚠️ Warning: Less than 2 features selected.")
            self.selected_features = ['feature_0', 'feature_1']

        if not feature_scaler:
            raise ValueError("❌ feature_scaler is required for evaluation!")
        self.feature_scaler = feature_scaler

        if is_main_process():
            tqdm.write(f"\n📐 Evaluation Feature Scaler:")
            tqdm.write(f"   {self.selected_features[0]}: "
                       f"mean={self.feature_scaler['feat1_mean']:.4f}, "
                       f"std={self.feature_scaler['feat1_std']:.4f}")
            tqdm.write(f"   {self.selected_features[1]}: "
                       f"mean={self.feature_scaler['feat2_mean']:.4f}, "
                       f"std={self.feature_scaler['feat2_std']:.4f}")

        if self.use_chinese:
            self.template_dynamic = SOH_QUESTIONS_DYNAMIC_CN[0]
            self.template_basic   = SOH_QUESTIONS_CN[0]
        else:
            self.template_dynamic = SOH_QUESTIONS_DYNAMIC[0]
            self.template_basic   = SOH_QUESTIONS_BASIC[0]

    def _load_descriptions(self, description_folder):
        for desc_file in [f for f in os.listdir(description_folder) if f.endswith('.csv')]:
            try:
                df = pd.read_csv(os.path.join(description_folder, desc_file))
                if 'description' in df.columns and len(df) > 0:
                    self.description_map[desc_file] = self._clean_desc(
                        df['description'].iloc[0])
            except Exception:
                pass

    def _clean_desc(self, desc):
        if not isinstance(desc, str):
            return "Li-ion battery"
        parts = desc.split('.')
        return parts[0].strip() if parts else desc[:50]

    def infer_battery_type(self, filename):
        if filename in self.description_map:
            return self.description_map[filename]
        fname = filename.lower()
        if 'mit'  in fname: return "LFP/Graphite battery"
        if 'xjtu' in fname: return "NCM/Graphite battery"
        if 'tju'  in fname: return "Li-ion battery (TJU)"
        return "Li-ion battery"

    def calculate_dynamic_stats(self, df_window):
        stats_dict = {}
        x = np.arange(len(df_window))
        for feat_key, feat_name in [('feat1', self.selected_features[0]),
                                    ('feat2', self.selected_features[1])]:
            if feat_name in df_window.columns:
                data = df_window[feat_name].values.astype(float)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                stats_dict[f'{feat_key}_name'] = feat_name
                raw_val  = float(data[-1])
                norm_val = ((raw_val - self.feature_scaler[f'{feat_key}_mean']) /
                            self.feature_scaler[f'{feat_key}_std'])
                stats_dict[f'{feat_key}_val'] = float(np.clip(norm_val, -3.0, 3.0))
                if len(data) > 1 and np.std(data) > 1e-10:
                    try:
                        slope, *_ = stats.linregress(x, data)
                        slope = float(np.clip(
                            slope if np.isfinite(slope) else 0.0, -10.0, 10.0))
                    except Exception:
                        slope = 0.0
                else:
                    slope = 0.0
                stats_dict[f'{feat_key}_slope'] = slope
            else:
                stats_dict.update({
                    f'{feat_key}_name': feat_name,
                    f'{feat_key}_val':  0.0,
                    f'{feat_key}_slope': 0.0
                })
        for key, val in stats_dict.items():
            if isinstance(val, float) and not np.isfinite(val):
                stats_dict[key] = 0.0
        return stats_dict

    def generate_prompt(self, battery_type, stats_dict, use_dynamic=True):
        if use_dynamic:
            try:
                return self.template_dynamic.format(battery_type=battery_type, **stats_dict)
            except Exception as e:
                tqdm.write(f"⚠️ Prompt Error: {e}")
                return f"Estimate SOH for {battery_type}."
        return self.template_basic


# ============ 模型加载 ============

def load_model(config, checkpoint_path, device):
    log_main(f"Loading model from: {checkpoint_path}")
    model = BatteryGPTModel(
        clbp_ckpt_path=config['clbp_ckpt_path'],
        vicuna_ckpt_path=config['vicuna_ckpt_path'],
        soh_min=config['soh_min'],
        soh_max=config['soh_max'],
        soh_step=config['soh_step'],
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.1),
        max_tgt_len=config.get('max_tgt_len', 128),
        use_official_llama=True,
        device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing and is_main_process():
        tqdm.write(f"ℹ️  Missing keys: {len(missing)}")
    if unexpected and is_main_process():
        tqdm.write(f"⚠️  Unexpected keys: {len(unexpected)}")
    model = model.to(device)
    model.eval()
    return model


# ============ 工具函数 ============

def _clean_features(df, feature_cols):
    if all(c in df.columns for c in feature_cols):
        arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    else:
        arr_list = [df[c].values if c in df.columns else np.zeros(len(df))
                    for c in feature_cols]
        arr = np.stack(arr_list, axis=1).astype(np.float32)
    if arr.size > 0 and not np.isfinite(arr).all():
        for c in range(arr.shape[1]):
            col  = arr[:, c]
            mask = np.isfinite(col)
            col[~mask] = col[mask].mean() if mask.any() else 0.0
            arr[:, c]  = col
    return arr


def compute_global_mean_std(data_folder):
    cache_path = Path(data_folder) / '_norm_cache.npz'
    if cache_path.exists():
        log_main(f"📊 Loading normalization cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return (cache['mean'].astype(np.float32),
                cache['std'].astype(np.float32),
                list(cache['feature_columns']))

    csv_files = sorted(Path(data_folder).glob("*.csv"))
    log_main(f"📊 Calculating normalization stats from {len(csv_files)} files...")
    if not csv_files:
        raise ValueError(f"No CSV files in {data_folder}")

    first_df = pd.read_csv(csv_files[0])
    exclude  = {'soh', 'capacity', 'description', 'file_name', 'battery_id',
                'cycle', 'time', 'timestamp', 'date', 'index'}
    feature_columns = [c for c in first_df.columns if c not in exclude]

    count         = 0
    running_sum   = np.zeros(len(feature_columns), dtype=np.float64)
    running_sumsq = np.zeros(len(feature_columns), dtype=np.float64)

    for fp in csv_files:
        try:
            df  = pd.read_csv(fp)
            arr = _clean_features(df, feature_columns)
            if arr.size > 0:
                count         += arr.shape[0]
                running_sum   += arr.sum(axis=0)
                running_sumsq += (arr.astype(np.float64) ** 2).sum(axis=0)
        except Exception:
            pass

    mean = (running_sum / count).astype(np.float32)
    var  = (running_sumsq / count) - mean.astype(np.float64) ** 2
    std  = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std[std == 0] = 1.0

    # 只由 rank0 写缓存，避免多进程写冲突
    if is_main_process():
        np.savez(cache_path, mean=mean, std=std, feature_columns=feature_columns)
        log_main(f"💾 Saved normalization cache to {cache_path}")
    return mean, std, feature_columns


def compute_feature_scaler(data_folder, selected_features):
    feat_key   = '_'.join(selected_features[:2])
    cache_path = Path(data_folder) / f'_scaler_{feat_key}.yaml'
    if cache_path.exists():
        log_main(f"📐 Loading feature scaler cache from {cache_path}")
        with open(cache_path, 'r') as f:
            return yaml.safe_load(f)

    csv_files = sorted(Path(data_folder).glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files in {data_folder}")

    values = {0: [], 1: []}
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            for i, feat in enumerate(selected_features[:2]):
                if feat in df.columns:
                    v = df[feat].values.astype(float)
                    values[i].extend(v[np.isfinite(v)].tolist())
        except Exception as e:
            tqdm.write(f"⚠️ Failed to load {fp.name}: {e}")

    for i in range(2):
        if not values[i]:
            raise ValueError(f"No valid values for feature '{selected_features[i]}'")

    scaler = {
        'feat1_mean': float(np.mean(values[0])),
        'feat1_std':  float(np.std(values[0])) + 1e-8,
        'feat2_mean': float(np.mean(values[1])),
        'feat2_std':  float(np.std(values[1])) + 1e-8,
    }
    if is_main_process():
        with open(cache_path, 'w') as f:
            yaml.dump(scaler, f)
        log_main(f"💾 Saved feature scaler cache to {cache_path}")
    return scaler


def extract_soh_from_text(text):
    if not text or not isinstance(text, str):
        return None
    t = re.split(r'#{2,}|<\/s>|<eos>', text)[0]
    t = t.lower().strip()
    if not t:
        return None

    m = re.search(r'soh\s*[=:is]*\s*(\d+\.?\d*)', t)
    if m:
        v = float(m.group(1))
        return min(v / 100.0 if v > 1.0 else v, 1.0)

    m = re.search(r'(\d+\.?\d*)\s*%', t)
    if m:
        v = float(m.group(1))
        return min(v / 100.0 if v > 1.0 else v, 1.0)

    m = re.search(r'\b(0\.\d+)\b', t)
    if m:
        return float(m.group(1))

    return None


# ============ 核心评估函数（单 rank） ============

def _evaluate_battery_files(
    battery_files, model, device, config,
    mean_tensor, std_tensor, feature_columns,
    prompt_gen, numerical_token_ids, rank
):
    """
    对分配给本 rank 的电池文件列表进行评估。
    返回:
        battery_results : dict
        all_true        : np.ndarray
        all_text        : np.ndarray
        clbp_ent_arr    : np.ndarray
        llm_ent_arr     : np.ndarray
        combined_arr    : np.ndarray
        parse_failure   : int
    """
    text_batch_size = config.get('text_gen_batch_size', 32)
    compute_unc     = bool(config.get('compute_uncertainty', False))
    max_new_tokens  = int(config.get('max_new_tokens', 20))
    unc_alpha       = float(config.get('uncertainty_alpha', 0.5))
    unc_beta        = float(config.get('uncertainty_beta', 0.5))
    window_size     = config['seq_len']
    stride          = config.get('stride', 1)

    model_unwrapped = model.module if hasattr(model, 'module') else model
    diagnosis_done  = False
    parse_failure   = 0

    battery_results = {}
    all_true, all_text          = [], []
    all_clbp_ent, all_llm_ent  = [], []
    all_combined                = []

    # Rank0 显示进度条，其他 rank 静默
    bat_iter = tqdm(
        battery_files,
        desc=f"[GPU{rank}] Batteries",
        dynamic_ncols=True,
        position=rank,
        disable=(rank != 0),
    )

    for battery_file in bat_iter:
        battery_name = battery_file.stem

        try:
            df = pd.read_csv(battery_file)
        except Exception as e:
            tqdm.write(f"[GPU{rank}] Error reading {battery_file}: {e}")
            continue

        if 'soh' not in df.columns:
            continue

        features_np  = _clean_features(df, feature_columns)
        soh_values   = df['soh'].values
        battery_type = prompt_gen.infer_battery_type(battery_file.name)

        num_windows = (features_np.shape[0] - window_size) // stride + 1
        if num_windows <= 0:
            continue

        current_windows, current_prompts = [], []
        current_indices, current_sohs   = [], []

        bat_res = {
            'window_end_idx': [], 'true_soh': [],
            'pred_soh_text':  [], 'uncertainty': []
        }

        for i in range(num_windows):
            start = i * stride
            end   = start + window_size

            win_data  = features_np[start:end]
            true_soh  = float(soh_values[end - 1])
            df_window = df.iloc[start:end]
            dyn_stats = prompt_gen.calculate_dynamic_stats(df_window)
            prompt    = prompt_gen.generate_prompt(battery_type, dyn_stats, use_dynamic=True)

            current_windows.append(win_data)
            current_prompts.append(prompt)
            current_indices.append(end - 1)
            current_sohs.append(true_soh)

            if len(current_windows) >= text_batch_size or i == num_windows - 1:
                batch_len = len(current_windows)

                ts_batch = torch.tensor(
                    np.stack(current_windows), dtype=torch.float32, device=device)
                ts_batch = (ts_batch - mean_tensor) / std_tensor

                text_preds      = []
                generated_texts = []
                soh_dists       = []
                scores          = None
                gen_ids         = None

                if hasattr(model_unwrapped, 'generate'):
                    # 诊断只做一次（仅 rank0）
                    if not diagnosis_done and rank == 0:
                        diagnosis_done = True
                        diagnose_generate_output(
                            model_unwrapped, ts_batch[:1], [current_prompts[0]])

                    try:
                        batch_input = {
                            'timeseries': ts_batch,
                            'prompt':     current_prompts
                        }
                        gen_out = model_unwrapped.generate(
                            batch_input,
                            max_new_tokens=max_new_tokens,
                            temperature=0.1,
                            return_scores=compute_unc,
                        )
                        texts, soh_dist_batch, scores, gen_ids = \
                            _parse_generate_output_batch(gen_out)

                        # 对齐长度
                        if len(texts) < batch_len:
                            texts += [""] * (batch_len - len(texts))
                        elif len(texts) > batch_len:
                            texts = texts[:batch_len]

                        # 拆分 soh_dists
                        if soh_dist_batch is None:
                            soh_dists = [None] * batch_len
                        else:
                            arr = np.array(soh_dist_batch)
                            if arr.ndim == 1:
                                soh_dists = [arr.copy() for _ in range(batch_len)]
                            else:
                                soh_dists = [
                                    arr[k] if k < arr.shape[0] else None
                                    for k in range(batch_len)
                                ]

                        for text_str in texts:
                            generated_texts.append(text_str)
                            val = extract_soh_from_text(text_str)
                            if val is None:
                                parse_failure += 1
                                text_preds.append(float('nan'))
                            else:
                                text_preds.append(val)

                    except Exception as e:
                        tqdm.write(f"[GPU{rank}] ❌ Batch Gen Error: {e}")
                        text_preds      = [float('nan')] * batch_len
                        generated_texts = ["ERROR"]       * batch_len
                        soh_dists       = [None]          * batch_len
                        scores          = None
                        gen_ids         = None
                else:
                    text_preds      = [float('nan')] * batch_len
                    generated_texts = ["NO_GENERATE"]  * batch_len
                    soh_dists       = [None]           * batch_len

                # 不确定性
                if compute_unc:
                    batch_clbp, batch_llm, batch_comb = _compute_uncertainty_batch(
                        soh_dists, scores, gen_ids,
                        model_unwrapped, numerical_token_ids,
                        device, alpha=unc_alpha, beta=unc_beta
                    )
                else:
                    batch_clbp = [float('nan')] * batch_len
                    batch_llm  = [float('nan')] * batch_len
                    batch_comb = [float('nan')] * batch_len

                all_clbp_ent.extend(batch_clbp)
                all_llm_ent.extend(batch_llm)
                all_combined.extend(batch_comb)

                bat_res['window_end_idx'].extend(current_indices)
                bat_res['true_soh'].extend(current_sohs)
                bat_res['pred_soh_text'].extend(text_preds)
                bat_res['uncertainty'].extend(batch_comb)

                all_true.extend(current_sohs)
                all_text.extend(text_preds)

                current_windows, current_prompts = [], []
                current_indices, current_sohs   = [], []

        battery_results[battery_name] = bat_res

    return (
        battery_results,
        np.array(all_true,     dtype=np.float32),
        np.array(all_text,     dtype=np.float32),
        np.array(all_clbp_ent, dtype=np.float32),
        np.array(all_llm_ent,  dtype=np.float32),
        np.array(all_combined, dtype=np.float32),
        parse_failure
    )


# ============ 评估主逻辑 ============

def evaluate_soh_full(model, test_data_folder, device, output_dir, config,
                      cached_norm=None, rank=0, world_size=1):
    """
    多GPU并行评估入口。
    - 电池文件按 rank 分片
    - 各 rank 独立跑推理
    - Rank0 汇总、绘图、保存
    """
    text_batch_size = config.get('text_gen_batch_size', 32)
    compute_unc     = bool(config.get('compute_uncertainty', False))
    max_new_tokens  = int(config.get('max_new_tokens', 20))
    unc_alpha       = float(config.get('uncertainty_alpha', 0.5))
    unc_beta        = float(config.get('uncertainty_beta', 0.5))
    train_data_folder = config.get('train_data_folder') or test_data_folder

    log_main("\n" + "=" * 60)
    log_main(f"Full SOH Evaluation (Text Generation) — {world_size} GPU(s)")
    log_main("=" * 60)
    log_main(f"🚀 Batch={text_batch_size}, Stride={config.get('stride',1)}, "
             f"max_new_tokens={max_new_tokens}")
    log_main(f"🧪 Uncertainty: {'ON  α=' + str(unc_alpha) + ' β=' + str(unc_beta) if compute_unc else 'OFF'}")

    # ── 归一化参数（rank0 计算/读取，再广播） ──
    if cached_norm is not None:
        mean, std, feature_columns, selected_features, feature_scaler = cached_norm
        log_main("📦 Using cached normalization")
    else:
        # 只让 rank0 做 I/O，然后广播

        if rank == 0:
            mean, std, feature_columns = compute_global_mean_std(train_data_folder)
            log_main("\n📊 Selecting Top features...")
            try:
                selected_features = select_top_shared_features(train_data_folder, top_k=2)
            except Exception as e:
                log_main(f"⚠️ Feature selection failed: {e}. Using first two columns.")
                selected_features = feature_columns[:2]
            log_main(f"   Selected: {selected_features}")
            feature_scaler = compute_feature_scaler(train_data_folder, selected_features)
            norm_data = [mean, std, feature_columns, selected_features, feature_scaler]
        else:
            norm_data = [None]
        if world_size > 1:
            container = [norm_data if rank == 0 else None]
            dist.broadcast_object_list(container, src=0)
            mean, std, feature_columns, selected_features, feature_scaler = container[0]
        else:
            mean, std, feature_columns, selected_features, feature_scaler = norm_data

    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
    std_tensor  = torch.tensor(std,  dtype=torch.float32, device=device)

    prompt_gen = EvalPromptGenerator(
        description_folder=config.get('description_folder'),
        use_chinese=config.get('use_chinese', False),
        selected_features=selected_features,
        feature_scaler=feature_scaler
    )

    # ── 预计算 numerical_token_ids ──
    model_unwrapped = model.module if hasattr(model, 'module') else model
    if not hasattr(model_unwrapped, '_numerical_token_ids'):
        tok = getattr(model_unwrapped, "llama_tokenizer", None)
        if tok is not None:
            model_unwrapped._numerical_token_ids = get_numerical_token_ids(tok).to(device)
        else:
            model_unwrapped._numerical_token_ids = torch.tensor(
                [], dtype=torch.long, device=device)
    numerical_token_ids = model_unwrapped._numerical_token_ids

    # ── 按 rank 分片电池文件 ──
    all_test_files = sorted(Path(test_data_folder).glob("*.csv"))
    log_main(f"\n📂 Found {len(all_test_files)} test batteries, "
             f"distributing across {world_size} GPU(s)")

    # 轮询分配（保证均衡）
    local_files = [f for i, f in enumerate(all_test_files) if i % world_size == rank]
    log_main(f"   Rank{rank}: {len(local_files)} batteries")

    if dist.is_initialized():
        dist.barrier()  # 同步后再开始推理

    # ── 本 rank 推理 ──
    (local_bat_results,
     local_true, local_text,
     local_clbp, local_llm, local_comb,
     local_fail) = _evaluate_battery_files(
        local_files, model, device, config,
        mean_tensor, std_tensor, feature_columns,
        prompt_gen, numerical_token_ids, rank
    )

    if dist.is_initialized():
        dist.barrier()

    # ── Rank0 汇总 ──
    battery_results = gather_battery_results(local_bat_results, rank, world_size)
    all_true        = gather_array(local_true,  rank, world_size)
    all_text        = gather_array(local_text,  rank, world_size)
    clbp_ent_arr    = gather_array(local_clbp,  rank, world_size)
    llm_ent_arr     = gather_array(local_llm,   rank, world_size)
    combined_arr    = gather_array(local_comb,  rank, world_size)

    # 汇总 parse_failure
    if world_size > 1:
        fail_tensor = torch.tensor([local_fail], dtype=torch.long, device=device)
        dist.all_reduce(fail_tensor, op=dist.ReduceOp.SUM)
        parse_failure_total = int(fail_tensor.item())
    else:
        parse_failure_total = local_fail

    if rank != 0:
        norm_cache = (mean, std, feature_columns, selected_features, feature_scaler)
        return {}, {}, norm_cache

    # ============ Rank0：计算指标 ============
    if len(all_true) == 0:
        log_main("❌ No valid samples evaluated.")
        return {}, {}, None

    valid_mask         = np.isfinite(all_text)
    parse_success_rate = valid_mask.sum() / len(all_text) * 100
    valid_llm          = int(np.isfinite(llm_ent_arr).sum())
    total_samples      = len(all_true)

    if valid_mask.sum() > 0:
        mae_text  = float(np.mean(np.abs(all_true[valid_mask] - all_text[valid_mask])))
        rmse_text = float(np.sqrt(np.mean(
            (all_true[valid_mask] - all_text[valid_mask]) ** 2)))
        mape_text = float(np.mean(
            np.abs((all_true[valid_mask] - all_text[valid_mask])
                   / (all_true[valid_mask] + 1e-8))) * 100)
    else:
        mae_text = rmse_text = mape_text = float('nan')

    metrics    = {'text': {'mae': mae_text, 'rmse': rmse_text, 'mape': mape_text}}
    valid_clbp = int(np.isfinite(clbp_ent_arr).sum())
    valid_comb = int(np.isfinite(combined_arr).sum())

    log_main("\n" + "=" * 60)
    log_main("🔍 评估统计:")
    log_main(f"   总样本数    : {total_samples}")
    log_main(f"   文本解析成功: {valid_mask.sum()}/{total_samples} ({parse_success_rate:.1f}%)")
    log_main(f"   解析失败    : {parse_failure_total}")
    log_main(f"   真实SOH范围 : [{all_true.min():.3f}, {all_true.max():.3f}]")
    log_main(f"   预测SOH范围 : [{np.nanmin(all_text):.3f}, {np.nanmax(all_text):.3f}]")
    log_main(f"   不确定性有效: CLBP={valid_clbp}, LLM={valid_llm}, Combined={valid_comb}")
    log_main("=" * 60)
    log_main(f"\n📊 MAE={mae_text:.4f}, RMSE={rmse_text:.4f}, MAPE={mape_text:.2f}%")

    # 构建全局 sample_index_map（按电池名顺序拼接索引）
    sample_index_map = {}
    cursor = 0
    for bat_id in battery_results:
        n = len(battery_results[bat_id]['true_soh'])
        sample_index_map[bat_id] = list(range(cursor, cursor + n))
        cursor += n

    # 可视化
    plot_text_degradation(battery_results, output_dir, mae_text, rmse_text)
    plot_scatter(all_true, all_text, mae_text, rmse_text, output_dir)
    save_predictions_csv(all_true, all_text, output_dir,
                         clbp_ent=clbp_ent_arr,
                         llm_ent=llm_ent_arr,
                         combined_unc=combined_arr)

    if compute_unc:
        plot_uncertainty_distribution(
            all_true, combined_arr, clbp_ent_arr, llm_ent_arr,
            output_dir, llm_fixed=(valid_llm > 0))
        plot_degradation_with_uncertainty(
            battery_results, output_dir,
            list(clbp_ent_arr), list(llm_ent_arr), list(combined_arr),
            sample_index_map)

    norm_cache = (mean, std, feature_columns, selected_features, feature_scaler)
    return metrics, battery_results, norm_cache


# ============ 可视化模块 ============

def plot_text_degradation(battery_results, output_dir, mae, rmse):
    curves_dir = os.path.join(output_dir, 'text_curves')
    os.makedirs(curves_dir, exist_ok=True)
    num_batteries = len(battery_results)
    if num_batteries == 0:
        return

    for battery_id, data in battery_results.items():
        win_idx  = np.array(data['window_end_idx'])
        true_soh = np.array(data['true_soh'])
        pred_soh = np.array(data['pred_soh_text'])
        valid    = np.isfinite(pred_soh)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(win_idx, true_soh, 'k-', label='Ground Truth', linewidth=2)
        if valid.any():
            ax.plot(win_idx[valid], pred_soh[valid], 'r-',
                    label='BatteryGPT', linewidth=1.5, alpha=0.8)
        if (~valid).any():
            ax.scatter(win_idx[~valid], true_soh[~valid], c='gray',
                       s=10, alpha=0.5, label='Parse Failed', zorder=3)
        bat_mae = (float(np.mean(np.abs(true_soh[valid] - pred_soh[valid])))
                   if valid.any() else float('nan'))
        ax.set_title(f'{battery_id}  (MAE={bat_mae:.4f})', fontsize=13)
        ax.set_xlabel('Cycle/Step')
        ax.set_ylabel('SOH')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f'{battery_id}_text.png'), dpi=150)
        plt.close()

    n_cols = min(3, num_batteries)
    n_rows = (num_batteries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = (np.array([axes]).flatten() if num_batteries == 1
            else np.array(axes).flatten())

    for idx, (bat_id, data) in enumerate(battery_results.items()):
        ax       = axes[idx]
        win_idx  = np.array(data['window_end_idx'])
        true_soh = np.array(data['true_soh'])
        pred_soh = np.array(data['pred_soh_text'])
        valid    = np.isfinite(pred_soh)
        ax.plot(win_idx, true_soh, 'k-', linewidth=1.5)
        if valid.any():
            ax.plot(win_idx[valid], pred_soh[valid], 'r-', linewidth=1.2, alpha=0.8)
        ax.set_title(bat_id, fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(num_batteries, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Text Generation (MAE={mae:.4f}, RMSE={rmse:.4f})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_summary.png'), dpi=300)
    plt.close()
    log_main(f"✅ Saved text curves to {curves_dir}/")


def plot_scatter(true, text, mae, rmse, output_dir):
    if len(true) == 0:
        return
    valid = np.isfinite(text)
    if not valid.any():
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true[valid], text[valid], s=8, alpha=0.4, c='coral')
    vmin = min(float(true[valid].min()), float(text[valid].min()))
    vmax = max(float(true[valid].max()), float(text[valid].max()))
    ax.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='Ideal')
    ax.set_title(f'Text Generation\nMAE={mae:.4f}, RMSE={rmse:.4f}')
    ax.set_xlabel('True SOH')
    ax.set_ylabel('Predicted SOH')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_text.png'), dpi=300)
    plt.close()
    log_main("✅ Saved scatter plot")


def save_predictions_csv(true, text, output_dir,
                         clbp_ent=None, llm_ent=None, combined_unc=None):
    df = pd.DataFrame({'true_soh': true, 'pred_soh_text': text, 'err_text': text - true})
    if clbp_ent     is not None: df['clbp_entropy_norm']   = clbp_ent
    if llm_ent      is not None: df['llm_entropy_numeric'] = llm_ent
    if combined_unc is not None: df['combined_uncertainty'] = combined_unc
    save_path = os.path.join(output_dir, 'soh_predictions_full.csv')
    df.to_csv(save_path, index=False)
    log_main(f"✅ Saved predictions to {save_path}")


def plot_uncertainty_distribution(true_soh, combined_unc, clbp_ent, llm_ent,
                                   output_dir, llm_fixed=False):
    valid = np.isfinite(combined_unc)
    if valid.sum() == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(true_soh[valid], combined_unc[valid], s=8, alpha=0.4, c='steelblue')
    ax.set_xlabel('True SOH'); ax.set_ylabel('Combined Uncertainty')
    ax.set_title('Uncertainty vs True SOH'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    valid_clbp = np.isfinite(clbp_ent)
    if valid_clbp.any():
        ax.hist(clbp_ent[valid_clbp], bins=50, color='coral',
                edgecolor='white', linewidth=0.5)
    ax.set_title(f'CLBP Entropy  μ={np.nanmean(clbp_ent):.4f}  σ={np.nanstd(clbp_ent):.4f}')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    valid_llm = np.isfinite(llm_ent)
    if llm_fixed and valid_llm.any():
        ax.hist(llm_ent[valid_llm], bins=50, color='mediumseagreen',
                edgecolor='white', linewidth=0.5)
        ax.set_title(f'LLM Entropy  μ={np.nanmean(llm_ent):.4f}  '
                     f'σ={np.nanstd(llm_ent):.4f}  valid={valid_llm.sum()}')
    else:
        ax.text(0.5, 0.5, 'LLM Entropy — Not Available',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('LLM Numeric Token Entropy')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(combined_unc[valid], bins=50, color='mediumpurple',
            edgecolor='white', linewidth=0.5)
    ax.set_title(f'Combined  μ={np.nanmean(combined_unc):.4f}  σ={np.nanstd(combined_unc):.4f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_main(f"✅ Saved uncertainty analysis to {save_path}")


def plot_degradation_with_uncertainty(battery_results, output_dir,
                                       all_clbp_ent, all_llm_ent, all_combined,
                                       sample_index_map):
    curves_dir   = os.path.join(output_dir, 'uncertainty_curves')
    os.makedirs(curves_dir, exist_ok=True)
    combined_arr = np.array(all_combined, dtype=np.float32)

    for bat_id, data in battery_results.items():
        indices = sample_index_map.get(bat_id, [])
        if not indices:
            continue

        win_idx   = np.array(data['window_end_idx'])
        true_soh  = np.array(data['true_soh'])
        pred_text = np.array(data['pred_soh_text'])
        bat_unc   = combined_arr[indices] if len(indices) <= len(combined_arr) else \
                    np.full(len(indices), float('nan'))

        valid_pred = np.isfinite(pred_text)
        valid_unc  = np.isfinite(bat_unc)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        ax1.plot(win_idx, true_soh, 'k-', label='Ground Truth', linewidth=2, zorder=4)
        if valid_pred.any():
            ax1.plot(win_idx[valid_pred], pred_text[valid_pred],
                     color='steelblue', linewidth=1.5, label='BatteryGPT', zorder=3, alpha=0.9)

        if valid_pred.any() and valid_unc.any():
            unc_for_band = bat_unc.copy()
            unc_for_band[~valid_unc] = float(np.nanmean(bat_unc))
            unc_norm = ((unc_for_band - np.nanmin(unc_for_band)) /
                        (np.nanmax(unc_for_band) - np.nanmin(unc_for_band) + 1e-8))
            width     = 0.02 + 0.10 * unc_norm
            upper     = np.clip(pred_text + width, 0.0, 1.1)
            lower     = np.clip(pred_text - width, 0.0, 1.1)
            ax1.fill_between(win_idx, lower, upper, alpha=0.25, color='steelblue',
                             label='Uncertainty Band', zorder=2)
            threshold = float(np.nanpercentile(bat_unc[valid_unc], 75))
            high_mask = (bat_unc > threshold) & valid_pred
            if high_mask.any():
                ax1.scatter(win_idx[high_mask], pred_text[high_mask],
                            c='red', s=25, zorder=5, alpha=0.7, label='High Uncertainty')

        valid_both = valid_pred & np.isfinite(true_soh)
        bat_mae    = (float(np.mean(np.abs(true_soh[valid_both] - pred_text[valid_both])))
                      if valid_both.any() else float('nan'))
        ax1.set_title(f'{bat_id}  MAE={bat_mae:.4f}  '
                      f'(parsed={valid_pred.sum()}/{len(valid_pred)})', fontsize=12)
        ax1.set_ylabel('SOH')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.3)
        soh_min, soh_max = float(true_soh.min()), float(true_soh.max())
        ax1.set_ylim([max(0.0, soh_min - 0.08), min(1.1, soh_max + 0.08)])

        if valid_unc.any():
            ax2.fill_between(win_idx, 0, bat_unc, where=valid_unc,
                             color='coral', alpha=0.55, label='Combined Entropy')
            ax2.plot(win_idx[valid_unc], bat_unc[valid_unc], 'r-', linewidth=1.0, alpha=0.8)
            ax2.axhline(y=float(np.nanmean(bat_unc)), color='darkred',
                        linestyle='--', linewidth=1, alpha=0.6)
            ax2.set_ylabel('Uncertainty')
            ax2.legend(fontsize=8, loc='upper right')
        ax2.set_xlabel('Cycle / Step')
        ax2.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f'{bat_id}_uncertainty.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    log_main(f"✅ Saved uncertainty curves to {curves_dir}/")
    _plot_uncertainty_summary(battery_results, output_dir, sample_index_map, combined_arr)


def _plot_uncertainty_summary(battery_results, output_dir, sample_index_map, combined_arr):
    num_batteries = len(battery_results)
    if num_batteries == 0:
        return
    n_cols = min(3, num_batteries)
    n_rows = (num_batteries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = (np.array([axes]).flatten() if num_batteries == 1
            else np.array(axes).flatten())

    for idx, (bat_id, data) in enumerate(battery_results.items()):
        ax         = axes[idx]
        win_idx    = np.array(data['window_end_idx'])
        true_soh   = np.array(data['true_soh'])
        pred_text  = np.array(data['pred_soh_text'])
        valid_pred = np.isfinite(pred_text)

        indices = sample_index_map.get(bat_id, [])
        bat_unc = (combined_arr[indices] if len(indices) > 0
                   else np.full(len(pred_text), float('nan')))
        valid_unc = np.isfinite(bat_unc)

        ax.plot(win_idx, true_soh, 'k-', linewidth=1.5)
        if valid_pred.any():
            ax.plot(win_idx[valid_pred], pred_text[valid_pred],
                    color='steelblue', linewidth=1.2, alpha=0.85)
        if valid_pred.any() and valid_unc.any():
            unc_n = ((bat_unc - np.nanmin(bat_unc)) /
                     (np.nanmax(bat_unc) - np.nanmin(bat_unc) + 1e-8))
            width = 0.02 + 0.08 * unc_n
            ax.fill_between(win_idx,
                            np.clip(pred_text - width, 0.0, 1.1),
                            np.clip(pred_text + width, 0.0, 1.1),
                            alpha=0.25, color='steelblue')
        valid_both = valid_pred & np.isfinite(true_soh)
        bat_mae    = (float(np.mean(np.abs(true_soh[valid_both] - pred_text[valid_both])))
                      if valid_both.any() else float('nan'))
        ax.set_title(f'{bat_id}\nMAE={bat_mae:.4f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.1])

    for idx in range(num_batteries, len(axes)):
        axes[idx].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color='k', linewidth=1.5, label='Ground Truth'),
        plt.Line2D([0], [0], color='steelblue', linewidth=1.2, label='Prediction'),
        plt.Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.25, label='Uncertainty Band')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('SOH with Uncertainty Bands', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    log_main("✅ Saved uncertainty summary")


# ============ 报告 ============

def generate_report(metrics, battery_results, output_dir, config, world_size=1):
    if not metrics:
        return
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    text_m    = metrics.get('text', {})
    unc_alpha = config.get('uncertainty_alpha', 0.5)
    unc_beta  = config.get('uncertainty_beta', 0.5)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BatteryGPT 评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"评估时间       : {datetime.now()}\n")
        f.write(f"模型路径       : {config.get('checkpoint_path', 'N/A')}\n")
        f.write(f"GPU 数量       : {world_size}\n")
        f.write(f"窗口大小       : {config.get('seq_len')}, Stride={config.get('stride', 1)}\n")
        f.write(f"max_new_tokens : {config.get('max_new_tokens', 20)}\n")
        f.write(f"不确定性       : {config.get('compute_uncertainty', False)}\n")
        f.write(f"  CLBP熵 α     : {unc_alpha}\n")
        f.write(f"  LLM熵  β     : {unc_beta}\n\n")
        f.write("-" * 40 + "\n")
        f.write("文本生成:\n")
        f.write(f"  MAE:  {text_m.get('mae', float('nan')):.4f}\n")
        f.write(f"  RMSE: {text_m.get('rmse', float('nan')):.4f}\n")
        f.write(f"  MAPE: {text_m.get('mape', float('nan')):.2f}%\n\n")
        f.write("-" * 40 + "\n")
        f.write("按电池:\n")
        for bat_id, res in battery_results.items():
            t     = np.array(res['true_soh'])
            x     = np.array(res['pred_soh_text'])
            valid = np.isfinite(x)
            mae_x = (float(np.mean(np.abs(t[valid] - x[valid])))
                     if valid.any() else float('nan'))
            f.write(f"  {bat_id}: N={len(t)}, valid={valid.sum()}, "
                    f"MAE={mae_x:.4f}, SOH=[{t.min():.3f},{t.max():.3f}]\n")
    log_main(f"✅ Saved report to {report_path}")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="BatteryGPT Evaluation (Multi-GPU)")
    parser.add_argument('--config',              type=str,   required=True)
    parser.add_argument('--checkpoint',          type=str,   required=True)
    parser.add_argument('--output_dir',          type=str,   default=None)
    parser.add_argument('--device',              type=str,   default=None,
                        help='单卡模式下指定设备，多GPU模式由LOCAL_RANK决定，无需指定')
    parser.add_argument('--test_data_folder',    type=str,   default=None)
    parser.add_argument('--train_data_folder',   type=str,   default=None)
    parser.add_argument('--text_gen_batch_size', type=int,   default=32)
    parser.add_argument('--stride',              type=int,   default=1)
    parser.add_argument('--max_new_tokens',      type=int,   default=20)
    parser.add_argument('--debug',               action='store_true')
    parser.add_argument('--compute_uncertainty', action='store_true')
    parser.add_argument('--uncertainty_alpha',   type=float, default=None)
    parser.add_argument('--uncertainty_beta',    type=float, default=None)
    args = parser.parse_args()

    # ── 分布式初始化 ──
    rank, world_size, device = setup_distributed()

    # 单卡模式下允许 --device 覆盖
    if world_size == 1 and args.device is not None:
        device = torch.device(args.device)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.test_data_folder:  config['test_data_folder']  = args.test_data_folder
    if args.train_data_folder: config['train_data_folder'] = args.train_data_folder

    config['text_gen_batch_size'] = args.text_gen_batch_size
    config['stride']              = args.stride
    config['max_new_tokens']      = args.max_new_tokens
    config['checkpoint_path']     = args.checkpoint
    config['debug_mode']          = args.debug

    if args.compute_uncertainty:           config['compute_uncertainty'] = True
    if args.uncertainty_alpha is not None: config['uncertainty_alpha']   = args.uncertainty_alpha
    if args.uncertainty_beta  is not None: config['uncertainty_beta']    = args.uncertainty_beta

    config.setdefault('compute_uncertainty', False)
    config.setdefault('uncertainty_alpha',   0.5)
    config.setdefault('uncertainty_beta',    0.5)

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()  # 等 rank0 建好目录

    log_main(f"\n🖥️  Multi-GPU Eval: world_size={world_size}, rank={rank}, device={device}")

    # ── 加载模型（每个 rank 独立加载到自己的GPU） ──
    model = load_model(config, args.checkpoint, device)

    result = evaluate_soh_full(
        model=model,
        test_data_folder=config['test_data_folder'],
        device=device,
        output_dir=output_dir,
        config=config,
        cached_norm=None,
        rank=rank,
        world_size=world_size,
    )
    metrics, battery_results = result[0], result[1]

    if rank == 0:
        generate_report(metrics, battery_results, output_dir, config, world_size)
        log_main("\n" + "=" * 60)
        log_main("✅ Evaluation Complete!")
        log_main(f"   Output: {output_dir}")
        log_main("=" * 60)

    cleanup_distributed()


if __name__ == '__main__':
    main()