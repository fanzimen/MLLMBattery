"""
SwanLab 日志记录器 - 简化版
只记录 batch/epoch 级别的训练指标，分离文本损失和 SOH 损失
"""

import os
from typing import Dict, Optional, Any
import warnings

# 可选依赖
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    warnings.warn("swanlab not installed. Logging will be disabled.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Plot logging will be disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SwanLabLogger:
    """
    SwanLab 日志记录器（简化版）
    
    只记录以下指标：
    - Batch 级别：loss, text_loss, soh_loss, acc, mae, rmse, lr
    - Epoch 级别：train/val 平均指标
    - SOH 预测可视化图
    """
    
    def __init__(
        self,
        project_name: str = "BatteryGPT",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[str] = None,
        rank: int = 0,
        enabled: bool = True,
    ):
        """
        初始化 SwanLab 日志记录器
        
        Args:
            project_name: SwanLab 项目名称
            experiment_name: 实验名称
            config: 超参数配置字典
            log_dir: 本地日志目录
            rank: 分布式训练的 rank（只有 rank 0 记录）
            enabled: 是否启用日志
        """
        self.rank = rank
        self.enabled = enabled and (rank == 0) and SWANLAB_AVAILABLE
        self.run = None
        
        # 统计累积指标（用于 epoch 平均）
        self.epoch_train_metrics = {
            'loss': [], 'text_loss': [], 'soh_loss': [], 
            'acc': [], 'mae': [], 'rmse': []
        }
        self.epoch_val_metrics = {
            'loss': [], 'text_loss': [], 'soh_loss': [], 
            'mae': [], 'rmse': [], 'r2': []
        }
        
        if not self.enabled:
            if rank == 0:
                print("[SwanLab] Disabled (swanlab not available or rank != 0)")
            return
        
        # 检测运行模式
        swanlab_mode = os.environ.get('SWANLAB_MODE', 'local').lower()
        
        # 初始化 SwanLab
        try:
            init_kwargs = {
                'project': project_name,
                'experiment_name': experiment_name,
                'config': config or {},
                'logdir': log_dir,
            }
            
            if swanlab_mode == 'cloud':
                init_kwargs['mode'] = 'cloud'
                api_key = os.environ.get('SWANLAB_API_KEY')
                if api_key:
                    init_kwargs['api_key'] = api_key
                else:
                    print("[SwanLab] Warning: SWANLAB_MODE=cloud but SWANLAB_API_KEY not set")
            elif swanlab_mode == 'local':
                init_kwargs['mode'] = 'local'
            elif swanlab_mode == 'disabled':
                self.enabled = False
                print("[SwanLab] Disabled by SWANLAB_MODE=disabled")
                return
            else:
                init_kwargs['mode'] = 'local'
            
            self.run = swanlab.init(**init_kwargs)
            print(f"[SwanLab] ✅ Initialized in {swanlab_mode} mode")
            print(f"[SwanLab] Project: {project_name}, Experiment: {experiment_name}")
            
        except Exception as e:
            print(f"[SwanLab] ⚠️ Initialization failed: {e}")
            self.enabled = False
    
    def log_batch_metrics(
        self,
        loss: float,
        text_loss: float,
        soh_loss: float,
        acc: float,
        soh_pred: Optional[Any] = None,
        soh_label: Optional[Any] = None,
        iteration: int = 0,
        lr: float = 0.0,
        mode: str = 'train'
    ):
        """
        记录 batch 级别的指标
        
        Args:
            loss: 总损失（text_loss + soh_loss）
            text_loss: LLM 文本生成损失
            soh_loss: SOH 回归损失
            acc: Token 准确率
            soh_pred: SOH 预测值 (batch,) 或 (batch, 1)
            soh_label: SOH 真实值 (batch,)
            iteration: 当前迭代步数
            lr: 学习率
            mode: 'train' 或 'val'
        """
        if not self.enabled or self.run is None:
            return
        
        prefix = f"{mode}/batch"
        metrics = {
            f"{prefix}/loss": loss,
            f"{prefix}/text_loss": text_loss,
            f"{prefix}/soh_loss": soh_loss,
            f"{prefix}/acc": acc,
        }
        
        # 计算 SOH MAE 和 RMSE
        if soh_pred is not None and soh_label is not None and NUMPY_AVAILABLE:
            try:
                import torch
                if torch.is_tensor(soh_pred):
                    soh_pred = soh_pred.detach().cpu().numpy()
                if torch.is_tensor(soh_label):
                    soh_label = soh_label.detach().cpu().numpy()
                
                soh_pred = np.array(soh_pred).flatten()
                soh_label = np.array(soh_label).flatten()
                
                mae = np.mean(np.abs(soh_pred - soh_label))
                rmse = np.sqrt(np.mean((soh_pred - soh_label) ** 2))
                
                metrics[f"{prefix}/soh_mae"] = mae
                metrics[f"{prefix}/soh_rmse"] = rmse
                
                # 累积到 epoch 统计
                if mode == 'train':
                    self.epoch_train_metrics['mae'].append(mae)
                    self.epoch_train_metrics['rmse'].append(rmse)
            except Exception as e:
                pass
        
        # 记录学习率
        if mode == 'train':
            metrics['train/lr'] = lr
        
        # 累积到 epoch 统计
        if mode == 'train':
            self.epoch_train_metrics['loss'].append(loss)
            self.epoch_train_metrics['text_loss'].append(text_loss)
            self.epoch_train_metrics['soh_loss'].append(soh_loss)
            self.epoch_train_metrics['acc'].append(acc)
        
        # 记录到 SwanLab
        try:
            swanlab.log(metrics, step=iteration)
        except Exception as e:
            print(f"[SwanLab] Warning: Failed to log batch metrics: {e}")
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float = 0.0,
        train_text_loss: float = 0.0,
        train_soh_loss: float = 0.0,
        train_acc: float = 0.0,
        val_loss: float = 0.0,
        val_text_loss: float = 0.0,
        val_soh_loss: float = 0.0,
        val_mae: float = 0.0,
        val_rmse: float = 0.0,
        val_r2: float = 0.0,
        lr: float = 0.0,
    ):
        """
        记录 epoch 级别的平均指标
        
        Args:
            epoch: 当前 epoch
            train_loss: 训练总损失
            train_text_loss: 训练文本损失
            train_soh_loss: 训练 SOH 损失
            train_acc: 训练准确率
            val_loss: 验证总损失
            val_text_loss: 验证文本损失
            val_soh_loss: 验证 SOH 损失
            val_mae: 验证 MAE
            val_rmse: 验证 RMSE
            val_r2: 验证 R²
            lr: 当前学习率
        """
        if not self.enabled or self.run is None:
            return
        
        metrics = {
            # 训练指标
            'train/epoch/loss': train_loss,
            'train/epoch/text_loss': train_text_loss,
            'train/epoch/soh_loss': train_soh_loss,
            'train/epoch/acc': train_acc,
            'train/lr': lr,
            
            # 验证指标
            'val/epoch/loss': val_loss,
            'val/epoch/text_loss': val_text_loss,
            'val/epoch/soh_loss': val_soh_loss,
            'val/epoch/soh_mae': val_mae,
            'val/epoch/soh_rmse': val_rmse,
            'val/epoch/soh_r2': val_r2,
        }
        
        # 如果有累积的训练指标，计算平均值
        if len(self.epoch_train_metrics['mae']) > 0:
            metrics['train/epoch/soh_mae'] = np.mean(self.epoch_train_metrics['mae'])
            metrics['train/epoch/soh_rmse'] = np.mean(self.epoch_train_metrics['rmse'])
        
        # 记录到 SwanLab
        try:
            swanlab.log(metrics, step=epoch)
        except Exception as e:
            print(f"[SwanLab] Warning: Failed to log epoch metrics: {e}")
        
        # 重置 epoch 累积器
        for key in self.epoch_train_metrics:
            self.epoch_train_metrics[key].clear()
        for key in self.epoch_val_metrics:
            self.epoch_val_metrics[key].clear()
    
    # def log_soh_prediction_plot(
    #     self,
    #     soh_true: Any,
    #     soh_pred: Any,
    #     epoch: int,
    #     split: str = 'val'
    # ):
    #     """
    #     记录 SOH 预测散点图
        
    #     Args:
    #         soh_true: 真实 SOH 值 (N,)
    #         soh_pred: 预测 SOH 值 (N,)
    #         epoch: 当前 epoch
    #         split: 'train' 或 'val'
    #     """
    #     if not self.enabled or self.run is None or not MATPLOTLIB_AVAILABLE:
    #         return
        
    #     try:
    #         import torch
    #         if torch.is_tensor(soh_true):
    #             soh_true = soh_true.detach().cpu().numpy()
    #         if torch.is_tensor(soh_pred):
    #             soh_pred = soh_pred.detach().cpu().numpy()
            
    #         soh_true = np.array(soh_true).flatten()
    #         soh_pred = np.array(soh_pred).flatten()
            
    #         # 创建散点图
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.scatter(soh_true, soh_pred, alpha=0.5, s=20, c='blue', edgecolors='none')
            
    #         # 理想线 (y=x)
    #         min_val = min(soh_true.min(), soh_pred.min())
    #         max_val = max(soh_true.max(), soh_pred.max())
    #         ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
            
    #         # ±5% 误差带
    #         ax.fill_between(
    #             [min_val, max_val],
    #             [min_val * 0.95, max_val * 0.95],
    #             [min_val * 1.05, max_val * 1.05],
    #             alpha=0.2, color='green', label='±5% Error'
    #         )
            
    #         # 计算指标
    #         mae = np.mean(np.abs(soh_true - soh_pred))
    #         rmse = np.sqrt(np.mean((soh_true - soh_pred) ** 2))
            
    #         ax.set_xlabel('True SOH', fontsize=12)
    #         ax.set_ylabel('Predicted SOH', fontsize=12)
    #         ax.set_title(f'{split.capitalize()} - Epoch {epoch}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=14)
    #         ax.legend(loc='upper left')
    #         ax.grid(True, alpha=0.3)
    #         ax.set_aspect('equal')
            
    #         plt.tight_layout()
            
    #         # 记录到 SwanLab
    #         swanlab.log({f'{split}/soh_prediction_plot': swanlab.Image(fig)}, step=epoch)
    #         plt.close(fig)
            
    #     except Exception as e:
    #         print(f"[SwanLab] Warning: Failed to log SOH prediction plot: {e}")
    
    def finish(self):
        """结束 SwanLab 运行"""
        if self.enabled and self.run is not None:
            try:
                swanlab.finish()
                print("[SwanLab] ✅ Run finished")
            except Exception as e:
                print(f"[SwanLab] Warning: Failed to finish run: {e}")