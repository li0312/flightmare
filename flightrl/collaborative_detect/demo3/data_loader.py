import torch
import json
import os
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CollaborativeTrackingDataset(Dataset):
    def __init__(self, json_file_list, seq_len=5, 
                 noise_std=0.02, prob_short_drop=0.1, prob_long_drop=0.15):
        """
        参数:
        - json_file_list: 包含所有 json 文件路径的列表 (从 train_files.txt 读取)
        - seq_len: 时序窗口长度 T
        - img_w, img_h: 图像宽高，用于坐标归一化
        - noise_std: 归一化后的高斯噪声标准差
        - prob_short_drop: 单帧随机丢包概率
        - prob_long_drop: 长时连续建筑遮挡触发概率
        """
        self.seq_len = seq_len
        
        self.noise_std = noise_std
        self.prob_short_drop = prob_short_drop
        self.prob_long_drop = prob_long_drop
        
        # 构建索引：遍历所有文件，计算可切片的滑动窗口数量
        self.samples_index = []
        # 注意：为了加速，实际工程中可将 JSON 内容预读进 RAM。此处演示按需读取的逻辑
        for file_path in json_file_list:
            # 假设每个 json chunk 长度固定为 40
            chunk_len = 40 
            for start_idx in range(chunk_len - seq_len + 1):
                self.samples_index.append((file_path, start_idx))

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        file_path, start_idx = self.samples_index[idx]
        
        # 读取 JSON 文件
        with open(file_path, 'r') as f:
            chunk_data = json.load(f)
            
        # 提取窗口内的 T 帧序列
        seq_data = chunk_data[start_idx : start_idx + self.seq_len]
        
        # 初始化张量容器
        ego_obs_seq = torch.zeros((self.seq_len, 6))
        peer_obs_seq = torch.zeros((self.seq_len, 6))
        ego_pose_seq = torch.zeros((self.seq_len, 7))
        peer_pose_seq = torch.zeros((self.seq_len, 7))
        
        for t, step_data in enumerate(seq_data):
            # 1. 组装 6D 观测 [xmin, ymin, xmax, ymax, vx, vy]
            ego_bbox = step_data["bbox_a_last"]
            peer_bbox = step_data["bbox_b"]
            ego_obs = np.array([ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3], step_data["motion_a_last"][0], step_data["motion_a_last"][1]])
            peer_obs = np.array([peer_bbox[0], peer_bbox[1], peer_bbox[2], peer_bbox[3], step_data["motion_b"][0], step_data["motion_b"][1]])
            # ego_obs[0] += np.random.normal(0, 2/960)
            # ego_obs[1] += np.random.normal(0, 2/540)
            # ego_obs[2] += np.random.normal(0, 2/960)
            # ego_obs[3] += np.random.normal(0, 2/540)
            # ego_obs[4] += np.random.normal(0, 0.03)
            # ego_obs[5] += np.random.normal(0, 0.03)

            ego_obs[0] += np.random.normal(0, 20/960)
            ego_obs[1] += np.random.normal(0, 20/540)
            ego_obs[2] += np.random.normal(0, 20/960)
            ego_obs[3] += np.random.normal(0, 20/540)
            ego_obs[4] += np.random.normal(0, 0.12)
            ego_obs[5] += np.random.normal(0, 0.12)

            # ego_obs[0] += np.random.normal(0, 8/960)
            # ego_obs[1] += np.random.normal(0, 8/540)
            # ego_obs[2] += np.random.normal(0, 8/960)
            # ego_obs[3] += np.random.normal(0, 8/540)
            # ego_obs[4] += np.random.normal(0, 0.09)
            # ego_obs[5] += np.random.normal(0, 0.09)
            # peer_obs[0] += np.random.normal(0, 8/960)
            # peer_obs[1] += np.random.normal(0, 8/540)
            # peer_obs[2] += np.random.normal(0, 8/960)
            # peer_obs[3] += np.random.normal(0, 8/540)

            
            
            # 2. 组装 7D 位姿 [px, py, pz, qw, qx, qy, qz]
            ego_pose = step_data["pose_a_delta"]
            peer_pose = step_data["pose_b_a"]
            
            # 归一化检测框
            ego_obs_seq[t] = torch.tensor(ego_obs)
            peer_obs_seq[t] = torch.tensor(peer_obs)
            ego_pose_seq[t] = torch.tensor(ego_pose)
            peer_pose_seq[t] = torch.tensor(peer_pose)
            
            # 保存第 T 帧的绝对纯净 Ground Truth 作为目标标签
            if t == self.seq_len - 1:
                gt_bbox = step_data["bbox_a"] # 必须是绝对真值
                gt_obs = [gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], step_data["motion_a"][0], step_data["motion_a"][1]]
                gt_target = torch.tensor(gt_obs)

        # ==================== 在线数据增强 (Data Augmentation) ====================
        
        # 1. 注入观测噪声
        # noise = torch.randn_like(peer_obs_seq[:, :4]) * self.noise_std
        # peer_obs_seq[:, :4] += noise
        
        # 2. 随机长时遮挡 (Block Dropout)
        if random.random() < self.prob_long_drop:
            blank_len = random.randint(0, self.seq_len)
            blank_start = random.randint(0, self.seq_len - blank_len)
            peer_obs_seq[blank_start : blank_start + blank_len, :] = -1.0
            
        # 3. 随机单帧闪烁丢包
        mask = torch.rand(self.seq_len) < self.prob_short_drop
        peer_obs_seq[mask, :] = -1.0
        
        # 4. 判断最后一帧（当前时刻 $t$）僚机是否失效，生成门控惩罚 Mask
        peer_mask = 0.0
        if (peer_obs_seq[-1, 0] == -1.0).all():
            peer_mask = 1.0
            
        return ego_obs_seq, peer_obs_seq, ego_pose_seq, peer_pose_seq, gt_target, torch.tensor([peer_mask])