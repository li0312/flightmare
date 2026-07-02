'''
Author: Lac_Creeper
Date: 2026-03-23 21:52:57 +0800
LastEditTime: 2026-03-25 04:19:49 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo1/models1.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_ciou(pred_boxes, target_boxes, eps=1e-7):
    """
    计算 CIoU (Complete IoU)
    参数:
        pred_boxes: 预测框 [N, 4], 格式为 [umin, umax, vmin, vmax]
        target_boxes: 真实框 [N, 4], 格式同上
    """
    # 分离坐标
    p_umin, p_umax, p_vmin, p_vmax = pred_boxes.unbind(-1)
    t_umin, t_umax, t_vmin, t_vmax = target_boxes.unbind(-1)

    # 1. 计算宽高
    p_w, p_h = (p_umax - p_umin).clamp(min=0), (p_vmax - p_vmin).clamp(min=0)
    t_w, t_h = (t_umax - t_umin).clamp(min=0), (t_vmax - t_vmin).clamp(min=0)

    # 2. 计算交集区域 (Intersection)
    inter_umin = torch.max(p_umin, t_umin)
    inter_umax = torch.min(p_umax, t_umax)
    inter_vmin = torch.max(p_vmin, t_vmin)
    inter_vmax = torch.min(p_vmax, t_vmax)
    
    inter_area = (inter_umax - inter_umin).clamp(min=0) * \
                 (inter_vmax - inter_vmin).clamp(min=0)

    # 3. 计算并集区域 (Union)
    p_area = p_w * p_h
    t_area = t_w * t_h
    union_area = p_area + t_area - inter_area + eps

    # 4. 计算 IoU
    iou = inter_area / union_area

    # 5. 计算最小外接矩形 (Enclosing Box)
    enclose_umin = torch.min(p_umin, t_umin)
    enclose_umax = torch.max(p_umax, t_umax)
    enclose_vmin = torch.min(p_vmin, t_vmin)
    enclose_vmax = torch.max(p_vmax, t_vmax)
    
    # 外接矩形对角线距离平方 c^2
    cw = enclose_umax - enclose_umin
    ch = enclose_vmax - enclose_vmin
    c2 = cw**2 + ch**2 + eps

    # 6. 计算中心点距离平方 rho^2
    p_center_u, p_center_v = (p_umin + p_umax) / 2, (p_vmin + p_vmax) / 2
    t_center_u, t_center_v = (t_umin + t_umax) / 2, (t_vmin + t_vmax) / 2
    rho2 = (p_center_u - t_center_u)**2 + (p_center_v - t_center_v)**2

    # 7. 计算 CIoU 惩罚项 (v 和 alpha)
    # v 衡量宽高比的一致性
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(t_w / (t_h + eps)) - torch.atan(p_w / (p_h + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # 最终 CIoU = IoU - (Distance_Term + Aspect_Ratio_Term)
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou.clamp(min=-1.0, max=1.0)


def to_bbox6(x):
    center_u = x[..., 0]
    center_v = x[..., 1]
    height = x[..., 2]
    motion_x = x[..., 3]
    motion_y = x[..., 4]
    width = height / 2
    umin = center_u - width / 2
    umax = center_u + width / 2
    vmin = center_v - height / 2
    vmax = center_v + height / 2
    return torch.stack([umin, umax, vmin, vmax, motion_x, motion_y], axis=-1)



class CollaborativeTrackingLoss(nn.Module):
    def __init__(self, ciou_weight=1.0, dir_weight=2.0, bbox_weight=1.0, dot_weight=1.0):
        super().__init__()
        self.ciou_weight = ciou_weight
        self.dir_weight = dir_weight
        self.bbox_weight = bbox_weight
        self.dot_weight = dot_weight

    def forward(self, pred, target):
        """
        pred/target: [Batch, Seq, 6]
        index 0-3: bbox [umin, umax, vmin, vmax]
        index 4-5: motion [vx, vy]
        """
        pred = to_bbox6(pred)
        target = to_bbox6(target)
        loss_bbox = F.mse_loss(pred[..., :4], target[..., :4])
        # 展平数据以便计算
        pred_flat = pred.view(-1, 6)
        target_flat = target.view(-1, 6)
        
        # --- 1. CIoU BBox Loss ---
        pred_boxes = pred_flat[:, :4]
        target_boxes = target_flat[:, :4]
        ciou_val = calculate_ciou(pred_boxes, target_boxes)
        loss_ciou = 1.0 - ciou_val.mean()

        # --- 2. Direction Loss (带归一化) ---
        pred_dir = pred_flat[:, 4:]
        target_dir = target_flat[:, 4:]
        
        # 归一化预测向量和目标向量
        loss_dir = F.mse_loss(pred_dir, target_dir)
        pred_dir_norm = pred_dir / (torch.norm(pred_dir, p=2, dim=-1, keepdim=True) + 1e-8)
        target_dir_norm = target_dir / (torch.norm(target_dir, p=2, dim=-1, keepdim=True) + 1e-8)
        loss_dot = 1.0 - F.cosine_similarity(pred_dir_norm, target_dir_norm, dim=-1).mean()

        # 总损失
        total_loss = self.ciou_weight * loss_ciou + self.dir_weight * loss_dir + self.bbox_weight * loss_bbox + self.dot_weight * loss_dot

        loss_items = {
            "total_loss": total_loss.item(),
            "loss_ciou": loss_ciou.item(),
            "loss_dir": loss_dir.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_dot": loss_dot.item()
        }

        return total_loss, loss_items



class DirectionNormLoss(nn.Module):
    """
    针对检测框的 MSE 损失和针对方向向量的归一化损失
    """
    def __init__(self, bbox_weight=1.0, dir_weight=5.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.dir_weight = dir_weight

    def forward(self, pred, target):
        # pred/target shape: [B, T, 6]
        # index 0:4 为 bbox (umin, umax, vmin, vmax)
        # index 4:6 为 motion (vx, vy)
        
        # 1. BBox Loss
        loss_bbox = F.mse_loss(pred[..., :3], target[..., :3])
        
        # 2. Direction Loss (归一化后再计算误差)
        pred_dir = pred[..., 3:]
        target_dir = target[..., 3:]
        
        eps = 1e-8
        # 对预测和目标方向向量进行 L2 归一化
        # pred_dir_norm = pred_dir / (torch.norm(pred_dir, p=2, dim=-1, keepdim=True) + eps)
        # target_dir_norm = target_dir / (torch.norm(target_dir, p=2, dim=-1, keepdim=True) + eps)
        pred_dir_norm = pred_dir
        target_dir_norm = target_dir
        
        # 计算归一化后的 MSE (保证模型只学习方向，不被模长干扰)
        loss_dir = F.mse_loss(pred_dir_norm, target_dir_norm)
        
        return self.bbox_weight * loss_bbox + self.dir_weight * loss_dir

class LSTMNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=5, num_layers=2):
        super(LSTMNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层
        # batch_first=True 匹配 [Batch, Seq, Feature] 的输入格式
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出层：映射到 6 维 (bbox 4 + motion 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: [B, T, 24]
        # h0, c0 默认为 0
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 我们预测整个序列的输出，以匹配 data_loader 的 y [B, T, 6]
        predictions = self.fc(lstm_out)
        
        return predictions
    

# --- 辅助组件：位置编码 (Transformer 必需) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        return x + self.pe[:, :x.size(1), :]

# --- 1. CNN 模型 (Temporal CNN) ---
class TemporalCNNNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=5):
        super().__init__()
        # 使用因果卷积 (Causal Conv) 确保不看到未来信息
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2) # padding=(k-1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, 24] -> [B, 24, T]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))[..., :-2] # 裁剪多余的 padding 保持因果性
        x = F.relu(self.conv2(x))[..., :-2]
        # -> [B, T, hidden]
        x = x.transpose(1, 2)
        return self.fc(x)

# --- 2. Cross-Attention 模型 ---
class CrossAttentionNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=5):
        super().__init__()
        self.embed_dim = hidden_dim
        
        # 1. 投影层：将原始输入映射到几何空间
        # 视角 A 的查询流：包含自身位姿变化和上一帧位置
        self.q_proj = nn.Sequential(
            nn.Linear(7, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 视角 B 的参考流：包含相对位姿和 B 的观察
        self.k_proj = nn.Sequential(
            nn.Linear(7, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 视角 B 的特征流：包含运动信息
        self.v_proj = nn.Sequential(
            nn.Linear(2, hidden_dim), # motion_b(2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Cross-Attention 层
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 3. 残差输出层
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5) 
        )

    def forward(self, x):
        # x shape: [B, T, 24]
        # 提取各个部分
        p_b_a = x[..., 0:4]
        bb_b = x[..., 4:7]
        mo_b = x[..., 7:9]
        p_a_d = x[..., 9:13]
        bb_a_l = x[..., 13:16]
        
        # 生成 Q, K, V
        # Q 模拟“单应矩阵请求”：基于 A 的变化寻找目标
        query = self.q_proj(torch.cat([p_a_d, bb_a_l], dim=-1)) 
        # K 模拟“单应矩阵对齐”：将 B 的观察转换到 A 的参考系
        key = self.k_proj(torch.cat([p_b_a, bb_b], dim=-1))
        # V 是实际携带的运动信息
        value = self.v_proj(mo_b)
        
        # 执行 Cross-Attention
        # 注意：这里我们加入一个因果 Mask，如果是时序序列
        attn_out, _ = self.attn(query, key, value)
        
        # 残差预测
        delta = self.output_head(attn_out)
        
        # 仿照单应矩阵的最后一步：基准位置 + 神经网络修正
        # bbox_a = bbox_a_last + delta
        final_bbox = bb_a_l + delta[..., :3] * 0.1
        final_motion = delta[..., 3:5]
        
        # return delta
        return torch.cat([final_bbox, final_motion], dim=-1)

# --- 3. Transformer 模型 ---
class TransformerNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=5, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, 24]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # 为 Transformer 生成 Mask，防止看到未来
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.transformer(x, mask=mask)
        return self.fc(x)