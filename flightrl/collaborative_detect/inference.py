'''
Author: Lac_Creeper
Date: 2026-03-02 21:52:57 +0800
LastEditTime: 2026-03-02 21:52:59 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/collaborative_detect/inference.py
'''
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import logging

from models import create_model
from data_loader import UAVTrackingDataset

logger = logging.getLogger(__name__)


class Inferencer:
    """推理器"""
    
    def __init__(self,
                 model_path: str,
                 model_type: str = 'gru',
                 device: str = 'cuda'):
        """
        Args:
            model_path: 模型检查点路径
            model_type: 模型类型
            device: 计算设备
        """
        self.device = torch.device(device)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        self.model = create_model(
            model_type=model_type,
            input_dim=config.get('input_dim', 24),
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('output_dim', 6)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        单个样本推理
        
        Args:
            x: 输入特征 [24] 或 [T, 24]
        
        Returns:
            y: 预测输出 [6]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            y = self.model(x)
        
        return y.cpu().numpy()
    
    def predict_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """
        批量推理
        
        Args:
            x_batch: [B, 24] 或 [B, T, 24]
        
        Returns:
            y_batch: [B, 6]
        """
        if isinstance(x_batch, np.ndarray):
            x_batch = torch.from_numpy(x_batch).float()
        
        x_batch = x_batch.to(self.device)
        
        with torch.no_grad():
            y_batch = self.model(x_batch)
        
        return y_batch.cpu().numpy()


def evaluate_on_dataset(model_path: str,
                       test_meta: str,
                       data_dir: str,
                       model_type: str = 'gru') -> Dict[str, float]:
    """
    在测试集上评估模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer(model_path, model_type, str(device))
    
    # 加载测试数据集
    test_dataset = UAVTrackingDataset(
        data_dir=data_dir,
        meta_file=test_meta,
        normalize=True,
        augment=False
    )
    
    all_y_pred = []
    all_y_true = []
    
    logger.info(f"Evaluating on {len(test_dataset)} test samples...")
    
    for i in range(len(test_dataset)):
        x, y = test_dataset[i]
        x_np = x.cpu().numpy() if hasattr(x, 'cpu') else x
        
        y_pred = inferencer.predict(x_np)
        y_true = y.cpu().numpy() if hasattr(y, 'cpu') else y
        
        all_y_pred.append(y_pred[0])
        all_y_true.append(y_true)
    
    all_y_pred = np.stack(all_y_pred)
    all_y_true = np.stack(all_y_true)
    
    # 反归一化
    if test_dataset.output_mean is not None:
        y_pred = all_y_pred * test_dataset.output_std + test_dataset.output_mean
        y_true = all_y_true * test_dataset.output_std + test_dataset.output_mean
    else:
        y_pred = all_y_pred
        y_true = all_y_true
    
    # 计算指标
    results = {}
    
    # BBox误差 (像素)
    bbox_errors = np.abs(y_pred[:, :4] - y_true[:, :4])
    results['bbox_mae'] = np.mean(bbox_errors)
    results['bbox_rmse'] = np.sqrt(np.mean(bbox_errors ** 2))
    results['bbox_max_error'] = np.max(bbox_errors)
    
    # 运动误差 (像素/帧)
    motion_errors = np.abs(y_pred[:, 4:6] - y_true[:, 4:6])
    results['motion_mae'] = np.mean(motion_errors)
    results['motion_rmse'] = np.sqrt(np.mean(motion_errors ** 2))
    results['motion_max_error'] = np.max(motion_errors)
    
    # 总误差
    results['total_mae'] = np.mean(np.abs(y_pred - y_true))
    results['total_rmse'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    logger.info("Evaluation Results:")
    for key, val in results.items():
        logger.info(f"  {key}: {val:.6f}")
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 评估最佳模型
    results = evaluate_on_dataset(
        model_path='./checkpoints/best_model.pth',
        test_meta='./data/test_files.txt',
        data_dir='./data',
        model_type='gru'
    )