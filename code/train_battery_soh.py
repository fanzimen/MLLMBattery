"""
BatteryGPT 训练脚本 - 支持多GPU并行训练 + SwanLab 监控
"""
import sys
import os
import warnings

# ============ 警告过滤（放在最前面） ============
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# 设置 OMP_NUM_THREADS 避免警告
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import torchvision
# --- [兼容性补丁开始] ---
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules["torchvision.transforms.functional_tensor"] = F
    except ImportError:
        pass
    
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.batterygpt import BatteryGPTModel
from datasets.battery_soh_dataset import BatterySOHDataset


from utils.swanlab_logger import SwanLabLogger


def setup_logging(log_dir, rank):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_rank{rank}.log')
    
    # 清除已有的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, rank, 
                    swanlab_logger=None, use_amp=True, gradient_accumulation_steps=4):
    """训练一个 epoch"""

    model.train()
    total_loss = 0
    total_text_loss = 0  # [新增]
    total_soh_loss = 0   # [新增]
    total_acc = 0
    num_batches = 0
    
    # 用于收集 SOH 预测
    all_soh_preds = []
    all_soh_labels = []
    
    optimizer.zero_grad()
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    global_step = epoch * len(dataloader)
    
    for i, batch in enumerate(pbar):

        with autocast('cuda', enabled=use_amp):
            # [修改] forward 现在返回 4 个值
            loss, acc, soh_pred, soh_loss = model(batch)
            loss = loss / gradient_accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积
        if (i + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 累积指标
        loss_value = loss.item() * gradient_accumulation_steps
        soh_loss_value = soh_loss.item()  # [新增]
        
        # [新增] 计算文本损失（总损失 - SOH损失）
        text_loss_value = loss_value - soh_loss_value
        
        total_loss += loss_value
        total_text_loss += text_loss_value  # [新增]
        total_soh_loss += soh_loss_value    # [新增]
        total_acc += acc
        num_batches += 1
        
        # 收集 SOH 预测
        if 'soh_labels' in batch:
            all_soh_preds.append(soh_pred.detach().cpu().numpy())
            all_soh_labels.append(batch['soh_labels'].cpu().numpy())
        
        # 记录 batch 级别指标
        if swanlab_logger and (i + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            swanlab_logger.log_batch_metrics(
                loss=loss_value,
                text_loss=text_loss_value,  # [新增]
                soh_loss=soh_loss_value,    # [新增]
                acc=acc,
                soh_pred=soh_pred if 'soh_labels' in batch else None,
                soh_label=batch['soh_labels'] if 'soh_labels' in batch else None,
                iteration=global_step + i,
                lr=current_lr,
                mode='train'
            )
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'text_loss': f'{text_loss_value:.4f}',  # [新增]
                'soh_loss': f'{soh_loss_value:.4f}',    # [新增]
                'acc': f'{acc:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_text_loss = total_text_loss / num_batches  # [新增]
    avg_soh_loss = total_soh_loss / num_batches    # [新增]
    avg_acc = total_acc / num_batches
    
    if rank == 0:
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f} (Text={avg_text_loss:.4f}, SOH={avg_soh_loss:.4f}), Acc={avg_acc:.4f}")
    
    return avg_loss, avg_text_loss, avg_soh_loss, avg_acc, all_soh_preds, all_soh_labels  # [修改返回值]



@torch.no_grad()
def evaluate(model, dataloader, logger, rank, swanlab_logger=None, epoch=0):
    """评估"""
    model.eval()
    total_loss = 0
    total_text_loss = 0  # [新增]
    total_soh_loss = 0   # [新增]
    total_mae = 0
    num_batches = 0
    
    all_soh_preds = []
    all_soh_labels = []
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Evaluating")
    else:
        pbar = dataloader
    
    for batch in pbar:
        loss, acc, soh_pred, soh_loss = model(batch)  # [修改]
        
        # 计算 MAE
        soh_true = batch['soh_labels'].to(soh_pred.device)
        mae = torch.abs(soh_pred.squeeze() - soh_true).mean().item()
        
        loss_value = loss.item()
        soh_loss_value = soh_loss.item()  # [新增]
        text_loss_value = loss_value - soh_loss_value  # [新增]
        
        total_loss += loss_value
        total_text_loss += text_loss_value  # [新增]
        total_soh_loss += soh_loss_value    # [新增]
        total_mae += mae
        num_batches += 1
        
        # 收集预测值
        all_soh_preds.append(soh_pred.squeeze().cpu().numpy())
        all_soh_labels.append(soh_true.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_text_loss = total_text_loss / num_batches  # [新增]
    avg_soh_loss = total_soh_loss / num_batches    # [新增]
    avg_mae = total_mae / num_batches
    
    # 合并所有预测
    all_soh_preds = np.concatenate(all_soh_preds)
    all_soh_labels = np.concatenate(all_soh_labels)
    
    # 计算 RMSE 和 R²
    rmse = np.sqrt(np.mean((all_soh_preds - all_soh_labels) ** 2))
    ss_res = np.sum((all_soh_labels - all_soh_preds) ** 2)
    ss_tot = np.sum((all_soh_labels - all_soh_labels.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    if rank == 0:
        logger.info(f"Eval: Loss={avg_loss:.4f} (Text={avg_text_loss:.4f}, SOH={avg_soh_loss:.4f}), MAE={avg_mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # # 绘制预测对比图
        # if swanlab_logger:
        #     swanlab_logger.log_soh_prediction_plot(
        #         soh_true=all_soh_labels,
        #         soh_pred=all_soh_preds,
        #         epoch=epoch,
        #         split='val'
        #     )
    
    return avg_loss, avg_text_loss, avg_soh_loss, avg_mae, rmse, r2  # [修改返回值]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--local_rank', type=int, default=0, help='本地GPU编号')
    parser.add_argument('--no_swanlab', action='store_true', help='禁用 SwanLab 日志')
    
    # ========== 新增：支持命令行覆盖配置文件 ==========
    parser.add_argument('--train_data_folder', type=str, default=None, help='训练数据目录（覆盖配置文件）')
    parser.add_argument('--test_data_folder', type=str, default=None, help='测试数据目录（覆盖配置文件）')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称（覆盖默认时间戳）')
    parser.add_argument('--keep_best_only', action='store_true', help='只保留最佳模型，删除中间检查点')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（覆盖配置文件）')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='训练前加载指定checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数（覆盖配置文件）')
    args = parser.parse_args()
    
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # ========== 命令行参数覆盖配置文件 ==========
    if args.train_data_folder is not None:
        config['train_data_folder'] = args.train_data_folder
    if args.test_data_folder is not None:
        config['test_data_folder'] = args.test_data_folder
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    
    # 创建输出目录
    if args.exp_name:
        run_name = args.exp_name
    else:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = os.path.join(config['output_dir'], f'run_{run_name}')
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # 等待 rank 0 创建目录
    if world_size > 1:
        dist.barrier()
    
    # 设置日志
    logger = setup_logging(output_dir, rank)
    
    # ============ 初始化 SwanLab ============
    swanlab_logger = None
    if not args.no_swanlab and rank == 0:
        # try:
        swanlab_config = {
            # 数据集配置
            'train_data_folder': config['train_data_folder'],
            'test_data_folder': config['test_data_folder'],
            'seq_len': config['seq_len'],
            'stride': config['stride'],
            
            # 模型配置
            'soh_min': config['soh_min'],
            'soh_max': config['soh_max'],
            'soh_step': config['soh_step'],
            'lora_r': config.get('lora_r', 8),
            'lora_alpha': config.get('lora_alpha', 32),
            'lora_dropout': config.get('lora_dropout', 0.1),
            
            # 训练配置
            'batch_size': config['batch_size'],
            'num_epochs': config['num_epochs'],
            'learning_rate': config['learning_rate'],
            'use_amp': config.get('use_amp', True),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            
            # 系统配置
            'world_size': world_size,
            'num_workers': config.get('num_workers', 4),
            
            # 实验信息
            'experiment_name': run_name,
            'keep_best_only': args.keep_best_only,
        }
        
        swanlab_logger = SwanLabLogger(
            project_name="BatteryGPT-SOH-Estimation",
            experiment_name=run_name,
            config=swanlab_config,
            log_dir=output_dir,
            rank=rank,
            enabled=True
        )
        # except Exception as e:
        #     logger.warning(f"SwanLab initialization failed: {e}. Continuing without SwanLab.")
        #     swanlab_logger = None
    
    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Experiment name: {run_name}")
        logger.info(f"Train data: {config['train_data_folder']}")
        logger.info(f"Test data: {config['test_data_folder']}")
        logger.info(f"Keep best only: {args.keep_best_only}")
        logger.info(f"SwanLab enabled: {swanlab_logger is not None}")
    
    # 加载数据集
    if rank == 0:
        logger.info("Loading datasets...")

    train_dataset = BatterySOHDataset(
        data_folder=config['train_data_folder'],
        seq_len=config['seq_len'],
        stride=config['stride'],
        use_chinese=config.get('use_chinese', 0.0),
        use_dynamic_prompt=1,  # 80% 概率使用增强提示
        description_folder="/mnt/disk1/fzm/codes/AnomalyGPT/data/full_descriptions_withmoreinfo_decimal3",  # 可选

    )

    test_dataset = BatterySOHDataset(
        data_folder=config['test_data_folder'],
        seq_len=config['seq_len'],
        stride=config['stride'],
        scaler=train_dataset.scaler,
        use_chinese=config.get('use_chinese', 0.0),
        use_dynamic_prompt=1,  # 80% 概率使用增强提示
        description_folder="/mnt/disk1/fzm/codes/AnomalyGPT/data/full_descriptions_withmoreinfo_decimal3",  # 可选


    )
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )
    
    if rank == 0:
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        logger.info("Initializing model...")
    
    # 初始化模型
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
        use_official_llama=False,
        use_soh_loss=config.get('use_soh_loss', True),        # [新增]
        soh_loss_weight=config.get('soh_loss_weight', 1.0),  # [新增]
        device=device
    )
    
    start_epoch = 1
    if args.resume_checkpoint:
        if rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_checkpoint}...")
        ckpt = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            if missing_keys:
                logger.info(f"⚠️ Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.info(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        start_epoch = ckpt.get('epoch', 0) + 1

    
    model = model.to(device)
    

    
    # 包装为 DDP
    if world_size > 1:

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # 混合精度训练
    scaler = GradScaler('cuda') if config.get('use_amp', True) else None

    # 如果 checkpoint 包含优化器/调度器/AMP 状态，继续加载
    if args.resume_checkpoint:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])    
    # 训练循环
    best_mae = float('inf')
    best_text_loss = float('inf')  # [修改] 新增：最佳验证集损失
    best_path_loss = None         # [修改] 最佳 Loss 模型路径
    best_path_mae = None          # [修改] 最佳 MAE 模型路径
    
    # 用于追踪中间检查点（如果 keep_best_only=True）
    epoch_checkpoints = []
    
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        # 设置采样器的 epoch（用于 shuffle）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练 - [修改] 接收新的返回值
        train_loss, train_text_loss, train_soh_loss, train_acc, train_soh_preds, train_soh_labels = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, logger, rank,
            swanlab_logger=swanlab_logger,
            use_amp=config.get('use_amp', True),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4)
        )
        
        # 评估 - [修改] 接收新的返回值
        test_loss, test_text_loss, test_soh_loss, test_mae, test_rmse, test_r2 = evaluate(
            model, test_loader, logger, rank,
            swanlab_logger=swanlab_logger,
            epoch=epoch
        )
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录 epoch 级别指标
        if swanlab_logger:
            swanlab_logger.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_text_loss=train_text_loss,    # [新增]
                train_soh_loss=train_soh_loss,      # [新增]
                train_acc=train_acc,
                val_loss=test_loss,
                val_text_loss=test_text_loss,       # [新增]
                val_soh_loss=test_soh_loss,         # [新增]
                val_mae=test_mae,
                val_rmse=test_rmse,
                val_r2=test_r2,
                lr=current_lr,
            )
        
        # 更新学习率
        scheduler.step()
        

        if rank == 0:
            model_to_save = model.module if world_size > 1 else model
            
            # [关键修复] 只保存 requires_grad=True 的参数
            trainable_state = {}
            for name, param in model_to_save.named_parameters():
                if param.requires_grad:
                    trainable_state[name] = param.data.cpu()
            
            # 同时保存一些必要的 buffer
            for name, buffer in model_to_save.named_buffers():
                if 'soh_' in name:
                    trainable_state[name] = buffer.cpu()
            
            # 计算保存大小
            total_size_mb = sum(p.numel() * p.element_size() for p in trainable_state.values()) / 1e6
            logger.info(f"Saving checkpoint ({total_size_mb:.2f} MB)...")
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': trainable_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'train_loss': train_loss,
                'test_loss': test_loss,  # [新增]
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'config': config,
            }
            
            latest_path = os.path.join(output_dir, 'latest_model.pt')
            torch.save(save_dict, latest_path)
            
            # [修改] 判定最佳模型逻辑：优先考虑 Loss (文本+回归综合能力)
            is_best_loss = False
            is_best_mae = False
            
            # 策略1: 保存 text Loss 最低的模型 (best_model_loss.pt) - 推荐用于对话生成
            if test_text_loss < best_text_loss:
                best_text_loss = test_text_loss
                best_path_loss = os.path.join(output_dir, 'best_model.pt') # 默认 best 这是 loss 最好的
                torch.save(save_dict, best_path_loss)
                logger.info(f"🔥 New Best Model (Loss)! Val text Loss={test_text_loss:.4f} (MAE={test_mae:.4f})")
                is_best_loss = True
            
            # 策略2: 保存 MAE 最低的模型 (best_model_mae.pt) - 推荐用于纯回归任务
            if test_mae < best_mae:
                best_mae = test_mae
                best_path_mae = os.path.join(output_dir, 'best_model_mae.pt')
                torch.save(save_dict, best_path_mae)
                logger.info(f"🎯 New Best Model (MAE)! MAE={test_mae:.4f} (Loss={test_loss:.4f})")
                is_best_mae = True
            
            # 定期保存
            if not args.keep_best_only and epoch % config.get('save_every', 5) == 0:
                epoch_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
                torch.save(save_dict, epoch_path)
                epoch_checkpoints.append(epoch_path)
    
    # ========== 清理中间检查点 ==========
    if rank == 0 and args.keep_best_only:
        logger.info("Cleaning up intermediate checkpoints (keep_best_only=True)...")
        # 删除中间检查点
        for ckpt_path in epoch_checkpoints:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        
        logger.info(f"✅ Retained:\n   - Best Loss Model: {best_path_loss}\n   - Best MAE Model: {best_path_mae}")
    
    if rank == 0:
        logger.info(f"Training completed!")
        logger.info(f"   Best Val Loss: {best_text_loss:.4f}")
        logger.info(f"   Best Val MAE:  {best_mae:.4f}")
        
        # 记录最终总结
        if swanlab_logger:
            # swanlab_logger.log_text(
            #     key='training_summary',
            #     text=f"""
            #     Training Completed!
            #     - Total Epochs: {config['num_epochs']}
            #     - Best Validation MAE: {best_mae:.4f}
            #     - Output Directory: {output_dir}
            #     - Keep Best Only: {args.keep_best_only}
            #     """,
            #     step=config['num_epochs']
            # )
            swanlab_logger.finish()
    
    # 清理
    cleanup_distributed()


if __name__ == '__main__':
    main()