import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np

from data_loader import create_data_loaders, UAVTrackingDataset
from models import create_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """统一的训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 train_dataset: UAVTrackingDataset,
                 val_dataset: UAVTrackingDataset,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            train_dataset: 训练数据集 (用于反归一化)
            val_dataset: 验证数据集
            config: 配置字典
            device: 计算设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # TensorBoard
        log_dir = Path(config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(log_dir / datetime.now().strftime('%Y%m%d_%H%M%S')))
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 结果统计
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 前向传播
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                          f"Loss: {avg_loss:.6f}")
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_y_pred = []
        all_y_true = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_y_pred.append(y_pred.cpu().numpy())
                all_y_true.append(y.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / num_batches
        
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        all_y_true = np.concatenate(all_y_true, axis=0)
        
        # 反归一化计算物理空间的误差
        if self.val_dataset.output_mean is not None:
            y_pred_denorm = all_y_pred * self.val_dataset.output_std + self.val_dataset.output_mean
            y_true_denorm = all_y_true * self.val_dataset.output_std + self.val_dataset.output_mean
        else:
            y_pred_denorm = all_y_pred
            y_true_denorm = all_y_true
        
        # BBox误差 (前4维)
        bbox_error = np.mean(np.abs(y_pred_denorm[:, :4] - y_true_denorm[:, :4]))
        
        # 运动误差 (后2维)
        motion_error = np.mean(np.abs(y_pred_denorm[:, 4:6] - y_true_denorm[:, 4:6]))
        
        metrics = {
            'loss': avg_loss,
            'bbox_mae': bbox_error,
            'motion_mae': motion_error
        }
        
        return metrics
    
    def train(self):
        """完整训练循环"""
        num_epochs = self.config.get('num_epochs', 100)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 记录结果
            self.results['train_losses'].append(train_metrics['loss'])
            self.results['val_losses'].append(val_metrics['loss'])
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/bbox_mae', val_metrics['bbox_mae'], epoch)
            self.writer.add_scalar('Metrics/motion_mae', val_metrics['motion_mae'], epoch)
            
            # 日志输出
            logger.info(f"Epoch {epoch+1} - "
                       f"Train Loss: {train_metrics['loss']:.6f}, "
                       f"Val Loss: {val_metrics['loss']:.6f}, "
                       f"BBox MAE: {val_metrics['bbox_mae']:.6f}, "
                       f"Motion MAE: {val_metrics['motion_mae']:.6f}")
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_path = self._save_checkpoint(epoch, val_metrics)
                logger.info(f"Best model saved! Loss: {self.best_val_loss:.6f}")
            
            # 学习率更新
            self.scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        logger.info("Training completed!")
        self._save_results()
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = True) -> Path:
        """保存模型检查点"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best:
            save_path = checkpoint_dir / f"best_model.pth"
        else:
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, save_path)
        
        logger.info(f"Checkpoint saved to {save_path}")
        return save_path
    
    def _save_results(self):
        """保存训练结果"""
        results_dir = Path(self.config.get('results_dir', 'results'))
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'training_results.json'
        
        # 转换numpy类型为Python原生类型
        results_dict = {
            'train_losses': [float(x) for x in self.results['train_losses']],
            'val_losses': [float(x) for x in self.results['val_losses']],
            'best_val_loss': float(self.best_val_loss),
            'config': self.config
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """主函数"""
    
    # 配置
    config = {
        # 数据配置
        'data_dir': './data',
        'train_meta': './data/train_files.txt',
        'val_meta': './data/val_files.txt',
        'batch_size': 32,
        'sequence_length': 4,  # 时序长度
        'num_workers': 4,
        
        # 模型配置
        'model_type': 'gru',  # 'gru', 'gru_attn', 'tcu'
        'input_dim': 24,
        'hidden_dim': 128,
        'output_dim': 6,
        'num_layers': 2,
        'dropout': 0.3,
        
        # 训练配置
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        
        # 目录配置
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
        'results_dir': './results'
    }
    
    # 创建目录
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['results_dir']).mkdir(exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 数据加载器
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        data_dir=config['data_dir'],
        train_meta=config['train_meta'],
        val_meta=config['val_meta'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    model = create_model(
        model_type=config['model_type'],
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    logger.info(f"Model: {config['model_type']}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()