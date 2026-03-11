'''
Author: Lac_Creeper
Date: 2026-03-02 05:46:23 +0800
LastEditTime: 2026-03-02 21:58:15 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /src/flightmare/flightrl/collaborative_detect/models.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GRUTracker(nn.Module):
    """
    基础GRU时序跟踪器
    
    架构:
    输入 [B, T, 24] -> GRU -> [B, T, hidden_dim]
                      -> 取最后时刻 [B, hidden_dim]
                      -> FC -> 输出 [B, 6]
    """
    
    def __init__(self,
                 input_dim: int = 24,
                 hidden_dim: int = 128,
                 output_dim: int = 6,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 24] 或 [B, 24] (单帧情况)
        
        Returns:
            y: [B, 6]
        """
        # 处理单帧输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 24] -> [B, 1, 24]
        
        batch_size, seq_len, _ = x.shape
        
        # 投影到隐藏维度
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # GRU编码
        gru_out, _ = self.gru(x)  # [B, T, gru_output_dim]
        
        # 取最后一帧
        last_output = gru_out[:, -1, :]  # [B, gru_output_dim]
        
        # 输出预测
        y = self.output_layer(last_output)  # [B, 6]
        
        return y


class GRUCrossAttentionTracker(nn.Module):
    """
    GRU + 交叉注意力跟踪器
    
    架构:
    输入 [B, T, 24] -> GRU -> [B, T, hidden_dim]
                    -> 分解为Query (UAV_A最后时刻) 和 Key/Value (UAV_B信息)
                    -> Cross-Attention
                    -> FC -> 输出 [B, 6]
    
    特点:
    - 使用交叉注意力融合UAV_B和UAV_A的特征
    - Query来自UAV_A的状态
    - Key/Value来自UAV_B的观测和历史
    """
    
    def __init__(self,
                 input_dim: int = 24,
                 hidden_dim: int = 128,
                 output_dim: int = 6,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU编码器
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 特征分解 (拆分成UAV_A和UAV_B部分)
        # 输入结构: [pose_b_a(7), bbox_b(4), motion_b(2), pose_a_delta(7), bbox_a_last(4)]
        # UAV_B部分: 0:13 (7+4+2)
        # UAV_A部分: 13:24 (7+4)
        self.uav_b_dim = 13
        self.uav_a_dim = 11
        
        # UAV_B特征投影
        self.uav_b_proj = nn.Linear(self.uav_b_dim, hidden_dim)
        
        # UAV_A特征投影
        self.uav_a_proj = nn.Linear(self.uav_a_dim, hidden_dim)
        
        # 交叉注意力模块
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 24] 或 [B, 24]
        
        Returns:
            y: [B, 6]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # 投影到隐藏维度
        x_proj = self.input_proj(x)  # [B, T, hidden_dim]
        
        # GRU编码
        gru_out, _ = self.gru(x_proj)  # [B, T, hidden_dim]
        
        # 拆分UAV_B和UAV_A部分
        uav_b_feat = x[:, :, :self.uav_b_dim]  # [B, T, 13]
        uav_a_feat = x[:, :, self.uav_b_dim:]   # [B, T, 11]
        
        # 投影
        uav_b_proj = self.uav_b_proj(uav_b_feat)  # [B, T, hidden_dim]
        uav_a_proj = self.uav_a_proj(uav_a_feat)  # [B, T, hidden_dim]
        
        # 交叉注意力: UAV_A最后时刻作为Query，UAV_B全时序作为Key/Value
        query = uav_a_proj[:, -1:, :]  # [B, 1, hidden_dim]
        key = uav_b_proj                # [B, T, hidden_dim]
        value = uav_b_proj              # [B, T, hidden_dim]
        
        attn_out, attn_weights = self.cross_attention(query, key, value)
        # attn_out: [B, 1, hidden_dim]
        
        # 融合GRU特征和注意力特征
        gru_last = gru_out[:, -1:, :]  # [B, 1, hidden_dim]
        fused = torch.cat([gru_last, attn_out], dim=-1)  # [B, 1, 2*hidden_dim]
        fused = self.fusion_layer(fused.squeeze(1))  # [B, hidden_dim]
        
        # 输出预测
        y = self.output_layer(fused)  # [B, 6]
        
        return y


class TCUTracker(nn.Module):
    """
    时序卷积单元 (Temporal Convolutional Unit) 跟踪器
    
    架构:
    输入 [B, T, 24] -> 投影 -> [B, hidden_dim, T]
                    -> 多层1D卷积 (因果卷积)
                    -> [B, hidden_dim, T]
                    -> 取最后时刻 [B, hidden_dim]
                    -> FC -> 输出 [B, 6]
    
    特点:
    - 使用因果卷积确保不泄露未来信息
    - 支持残差连接
    - 较GRU更轻量，并行性更好
    """
    
    def __init__(self,
                 input_dim: int = 24,
                 hidden_dim: int = 128,
                 output_dim: int = 6,
                 num_levels: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 时序卷积层 (因果卷积)
        self.tcn_layers = nn.ModuleList()
        self.tcn_bns = nn.ModuleList()
        self.tcn_dropouts = nn.ModuleList()
        
        for level in range(num_levels):
            dilation = 2 ** level
            padding = (kernel_size - 1) * dilation
            
            conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
            
            # 因果卷积：移除右侧填充
            self.tcn_layers.append(conv)
            self.tcn_bns.append(nn.BatchNorm1d(hidden_dim))
            self.tcn_dropouts.append(nn.Dropout(dropout))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 24] 或 [B, 24]
        
        Returns:
            y: [B, 6]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # 投影到隐藏维度并转置以适应Conv1d
        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)   # [B, hidden_dim, T]
        
        # 时序卷积层 (带残差连接)
        residual = x
        for conv, bn, dropout in zip(self.tcn_layers, self.tcn_bns, self.tcn_dropouts):
            x = conv(x)
            
            # 移除右侧填充以实现因果性
            x = x[:, :, :seq_len]
            
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # 残差连接
            x = x + residual
            residual = x
        
        # 取最后一帧
        last_feat = x[:, :, -1]  # [B, hidden_dim]
        
        # 输出预测
        y = self.output_layer(last_feat)  # [B, 6]
        
        return y


def create_model(model_type: str = 'gru',
                 input_dim: int = 24,
                 hidden_dim: int = 128,
                 output_dim: int = 6,
                 **kwargs) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: 'gru', 'gru_attn', 或 'tcu'
        input_dim: 输入维度
        hidden_dim: 隐藏维度
        output_dim: 输出维度
        **kwargs: 其他模型参数
    
    Returns:
        model: 构建的模型实例
    """
    if model_type == 'gru':
        return GRUTracker(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == 'gru_attn':
        return GRUCrossAttentionTracker(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == 'tcu':
        return TCUTracker(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")