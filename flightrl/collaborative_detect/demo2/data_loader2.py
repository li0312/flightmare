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
                 sequence_length: int = 20,
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
        with open(meta_file, 'r') as f:
            self.seq_files = [line.strip() for line in f if line.split()]

        self.all_sequences = []
        logger.info(f"Loading {len(self.seq_files)} sequences into memory..")
        for path in self.seq_files:
            with open(path, 'r') as f:
                self.all_sequences.append(json.load(f))

        print(f"Loaded {len(self.all_sequences)} sequences successfully.")
        
        # 计算归一化统计（仅用于训练集）
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
        if self.normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """计算输入输出的均值和标准差"""
        logger.info(">>>Computing normalization statistics...")

        all_x, all_y = [], []
        for seq in self.all_sequences[:400]:
            for frame in seq:
                # x, y = self._extract_frame_data(frame)
                x = self._extract_features(frame)
                y = self._extract_labels(frame)
                all_x.append(x)
                all_y.append(y)
        
        all_x = np.stack(all_x)
        all_y = np.stack(all_y)
        self.input_mean = all_x.mean(axis=0)
        self.input_std = all_x.std(axis=0) + 1e-8
        self.output_mean = all_y.mean(axis=0)
        self.output_std = all_y.std(axis=0) + 1e-8

        logger.info("<<<Statistics computed successfully.")
        
    
    def __len__(self) -> int:
        return len(self.all_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: 输入特征张量
                - 单帧模式: [24]
                - 时序模式: [T, 24]
            y: 输出标签张量 [6]
        """
        full_seq = self.all_sequences[idx]
        seq_len_total = len(full_seq)

        if seq_len_total > self.sequence_length:
            start = np.random.randint(0, seq_len_total - self.sequence_length)
        else:
            start = 0

        x_seq, y_seq = [], []
        for i in range(start, start + self.sequence_length):
            frame = full_seq[min(i, seq_len_total - 1)]
            # x, y = self._extract_frame_data(frame)
            x = self._extract_features(frame)
            y = self._extract_labels(frame)

            if self.augment:
                x, y = self._augment_data(x, y)

            if self.normalize and self.input_mean is not None:
                x = (x - self.input_mean) / self.input_std
                y = (y - self.output_mean) / self.output_std

            x_seq.append(x)
            y_seq.append(y)

        return torch.from_numpy(np.stack(x_seq)), torch.from_numpy(np.stack(y_seq))        
    
    def _load_json(self, path: str) -> Dict:
        """加载JSON文件"""
        with open(path, 'r') as f:
            return json.load(f)
        
    def _extract_frame_data(self, frame: Dict) -> Tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([
            frame['pose_b_a'], frame['bbox_b'], frame['motion_b'],
            frame['pose_a_delta'], frame['bbox_a_last']
        ]).astype(np.float32)
        y = np.concatenate([
            frame['bbox_a'],
            frame['motion_a']
        ]).astype(np.float32)

        return x, y
    
    
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
        x_aug[3:7] += np.random.normal(0, 0.02, 4)     # pose_b_a quaternion
        x_aug[13:16] += np.random.normal(0, 0.02, 3)    # pose_a_delta xyz
        x_aug[16:20] += np.random.normal(0, 0.02, 4)    # pose_a_delta quaternion
        
        # 为bbox添加微小抖动 (±5像素)
        # x_aug[7:11] += np.random.normal(0, 5, 4)        # bbox_b
        # x_aug[20:24] += np.random.normal(0, 5, 4)       # bbox_a_last
        
        # 为运动方向添加噪声
        x_aug[11:13] += np.random.normal(0, 0.05, 2)    # motion_b
        
        # 同样增强输出
        y_aug = y.copy()
        # y_aug[0:4] += np.random.normal(0, 5, 4)         # bbox_a
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
        normalize=False,
        augment=True
    )
    
    val_dataset = UAVTrackingDataset(
        data_dir=data_dir,
        meta_file=val_meta,
        sequence_length=sequence_length,
        normalize=False,
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