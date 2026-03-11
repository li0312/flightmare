'''
Author: Lac_Creeper
Date: 2026-03-02 06:28:36 +0800
LastEditTime: 2026-03-02 21:58:43 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/collaborative_detect/data_loader.py
'''
import numpy as np
import torch
import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class UAVTrackingDataset(Dataset):
    """
    双无人机追踪数据集加载器
    
    输入特征结构 (24维):
    - pose_b_a: [xyz(3), qw, qx, qy, qz(4)] = 7维
    - bbox_b: [umin, umax, vmin, vmax] = 4维
    - motion_b: [vx, vy] = 2维
    - pose_a_delta: [xyz(3), qw, qx, qy, qz(4)] = 7维
    - bbox_a_last: [umin, umax, vmin, vmax] = 4维
    
    输出特征结构 (6维):
    - bbox_a: [umin, umax, vmin, vmax] = 4维
    - motion_a: [vx, vy] = 2维
    """
    
    def __init__(self,
                 data_dir: str,
                 meta_file: str,
                 sequence_length: int = 1,
                 normalize: bool = True,
                 augment: bool = False):
        """
        Args:
            data_dir: 数据目录
            meta_file: 样本列表文件 (每行一个json文件路径)
            sequence_length: 时序长度 (1表示单帧，>1时会加载历史帧)
            normalize: 是否进行特征归一化
            augment: 是否进行数据增强
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.augment = augment
        
        # 读取样本列表
        self.samples = []
        with open(meta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(line)
        
        logger.info(f"Loaded {len(self.samples)} samples from {meta_file}")
        
        # 计算归一化统计（仅用于训练集）
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
        if self.normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """计算输入输出的均值和标准差"""
        logger.info("Computing normalization statistics...")
        
        inputs_list = []
        outputs_list = []
        
        for sample_path in self.samples[:min(len(self.samples), 100)]:  # 采样计算
            try:
                data = self._load_json(sample_path)
                x = self._extract_features(data)
                y = self._extract_labels(data)
                inputs_list.append(x)
                outputs_list.append(y)
            except Exception as e:
                logger.warning(f"Failed to load {sample_path}: {e}")
                continue
        
        if inputs_list:
            inputs = np.stack(inputs_list)
            outputs = np.stack(outputs_list)
            
            self.input_mean = inputs.mean(axis=0)
            self.input_std = inputs.std(axis=0) + 1e-8
            self.output_mean = outputs.mean(axis=0)
            self.output_std = outputs.std(axis=0) + 1e-8
            
            logger.info(f"Input stats - mean shape: {self.input_mean.shape}, std shape: {self.input_std.shape}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: 输入特征张量
                - 单帧模式: [24]
                - 时序模式: [T, 24]
            y: 输出标签张量 [6]
        """
        sample_path = self.samples[idx]
        
        try:
            data = self._load_json(sample_path)
        except Exception as e:
            logger.error(f"Failed to load {sample_path}: {e}")
            # 返回零张量
            if self.sequence_length == 1:
                return torch.zeros(24), torch.zeros(6)
            else:
                return torch.zeros(self.sequence_length, 24), torch.zeros(6)
        
        # 提取特征
        x = self._extract_features(data)
        y = self._extract_labels(data)
        
        # 数据增强（仅在训练时）
        if self.augment:
            x, y = self._augment_data(x, y)
        
        # 归一化
        if self.normalize and self.input_mean is not None:
            x = (x - self.input_mean) / self.input_std
            y = (y - self.output_mean) / self.output_std
        
        # 转为张量
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        # 构建时序序列（如果需要）
        if self.sequence_length > 1:
            x = self._build_sequence(x, idx)
        
        return x, y
    
    def _load_json(self, path: str) -> Dict:
        """加载JSON文件"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        """
        提取输入特征
        
        Returns:
            x: [24] 特征向量
        """
        pose_b_a = np.array(data['pose_b_a'], dtype=np.float32)      # [7]
        bbox_b = np.array(data['bbox_b'], dtype=np.float32)          # [4]
        motion_b = np.array(data['motion_b'], dtype=np.float32)      # [2]
        pose_a_delta = np.array(data['pose_a_delta'], dtype=np.float32)  # [7]
        bbox_a_last = np.array(data['bbox_a_last'], dtype=np.float32)    # [4]
        
        # 验证维度
        assert pose_b_a.shape == (7,), f"pose_b_a shape error: {pose_b_a.shape}"
        assert bbox_b.shape == (4,), f"bbox_b shape error: {bbox_b.shape}"
        assert motion_b.shape == (2,), f"motion_b shape error: {motion_b.shape}"
        assert pose_a_delta.shape == (7,), f"pose_a_delta shape error: {pose_a_delta.shape}"
        assert bbox_a_last.shape == (4,), f"bbox_a_last shape error: {bbox_a_last.shape}"
        
        x = np.concatenate([pose_b_a, bbox_b, motion_b, pose_a_delta, bbox_a_last])
        return x
    
    def _extract_labels(self, data: Dict) -> np.ndarray:
        """
        提取输出标签
        
        Returns:
            y: [6] 标签向量
        """
        bbox_a = np.array(data['bbox_a'], dtype=np.float32)      # [4]
        motion_a = np.array(data['motion_a'], dtype=np.float32)  # [2]
        
        assert bbox_a.shape == (4,), f"bbox_a shape error: {bbox_a.shape}"
        assert motion_a.shape == (2,), f"motion_a shape error: {motion_a.shape}"
        
        y = np.concatenate([bbox_a, motion_a])
        return y
    
    def _augment_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强：添加高斯噪声、旋转等
        """
        # 为pose添加微小噪声 (位置: ±0.02m, 角度: ±0.02rad)
        x_aug = x.copy()
        x_aug[0:3] += np.random.normal(0, 0.02, 3)      # pose_b_a xyz
        x_aug[7:11] += np.random.normal(0, 0.02, 4)     # pose_b_a quaternion
        x_aug[14:17] += np.random.normal(0, 0.02, 3)    # pose_a_delta xyz
        x_aug[17:21] += np.random.normal(0, 0.02, 4)    # pose_a_delta quaternion
        
        # 为bbox添加微小抖动 (±5像素)
        x_aug[7:11] += np.random.normal(0, 5, 4)        # bbox_b
        x_aug[21:25] += np.random.normal(0, 5, 4)       # bbox_a_last
        
        # 为运动方向添加噪声
        x_aug[11:13] += np.random.normal(0, 0.05, 2)    # motion_b
        
        # 同样增强输出
        y_aug = y.copy()
        y_aug[0:4] += np.random.normal(0, 5, 4)         # bbox_a
        y_aug[4:6] += np.random.normal(0, 0.05, 2)      # motion_a
        
        return x_aug, y_aug
    
    def _build_sequence(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        构建时序序列 (用于时序模型)
        简单方案：使用相邻样本构建
        """
        sequence = [x]
        
        # 向前查找历史帧
        for i in range(1, self.sequence_length):
            prev_idx = (idx - i) % len(self.samples)
            try:
                data = self._load_json(self.samples[prev_idx])
                x_prev = self._extract_features(data)
                if self.normalize and self.input_mean is not None:
                    x_prev = (x_prev - self.input_mean) / self.input_std
                x_prev = torch.from_numpy(x_prev).float()
                sequence.insert(0, x_prev)
            except:
                # 如果加载失败，使用零向量
                sequence.insert(0, torch.zeros_like(x))
        
        return torch.stack(sequence)
    
    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        """反归一化输出"""
        if self.normalize and self.output_mean is not None:
            return y * self.output_std + self.output_mean
        return y


def create_data_loaders(data_dir: str,
                       train_meta: str,
                       val_meta: str,
                       batch_size: int = 32,
                       sequence_length: int = 1,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    train_dataset = UAVTrackingDataset(
        data_dir=data_dir,
        meta_file=train_meta,
        sequence_length=sequence_length,
        normalize=True,
        augment=True
    )
    
    val_dataset = UAVTrackingDataset(
        data_dir=data_dir,
        meta_file=val_meta,
        sequence_length=sequence_length,
        normalize=True,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset