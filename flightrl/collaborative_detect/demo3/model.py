import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.ops as ops

class BaselineLoss(nn.Module):
    def __init__(self, lambda_bbox=1.0, lambda_vel=0.5, lambda_ego=0.2, lambda_gate=0.6):
        super().__init__()
        self.l1 = lambda_bbox
        self.l2 = lambda_vel
        self.l3 = lambda_ego
        self.l4 = lambda_gate
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def calc_iou_loss(self, pred_box, gt_box, eps=1e-7):
        p_umin, p_umax, p_vmin, p_vmax = pred_box.unbind(-1)
        t_umin, t_umax, t_vmin, t_vmax = gt_box.unbind(-1)
       
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
        return 1.0 - iou  # Loss 形式，越小越好

    def calc_diou_loss(self, pred_box, gt_box, eps=1e-7):
        """
        简化版 DIoU 计算 (假设输入为 [B, 4] -> tl_x, tl_y, br_x, br_y)
        实际论文工程中建议使用 torchvision.ops.distance_box_iou
        """
        """
        计算 CIoU (Complete IoU)
        参数:
            pred_boxes: 预测框 [N, 4], 格式为 [umin, umax, vmin, vmax]
            target_boxes: 真实框 [N, 4], 格式同上
        """
        # 分离坐标
        p_umin, p_umax, p_vmin, p_vmax = pred_box.unbind(-1)
        t_umin, t_umax, t_vmin, t_vmax = gt_box.unbind(-1)

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
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)  # 确保 CIoU 在合理范围内
        return 1.0 - ciou  # Loss 形式，越小越好

    def forward(self, pred_final, pred_ego_only, alpha, target, peer_mask):
        """
        pred_final: [B, 6] 最终输出 (tl, br, vel)
        pred_ego_only: [B, 6] 仅靠本机自运动预测的输出
        alpha: [B, 1] 僚机信息采纳度
        target: [B, 6] 真实的 Ground Truth
        peer_mask: [B, 1] 布尔/浮点张量，1.0 表示该样本的僚机数据被刻意遮挡(丢弃)了，0.0表示正常
        """
        # 拆解目标
        pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
        target_box, target_vel = target[:, :4], target[:, 4:]
        
        # 1. 最终输出的 BBox Loss (Smooth L1 + DIoU)
        # loss_bbox = self.smooth_l1(pred_box, target_box)
        # loss_bbox = 200 * self.smooth_l1(pred_box, target_box) + self.calc_diou_loss(pred_box, target_box).mean()
        scale = torch.tensor([960.0, 960.0, 540.0, 540.0], device=pred_box.device)
        offset = torch.tensor([480.0, 480.0, 270.0, 270.0], device=pred_box.device)

        pred_box_scaled = pred_box * scale + offset
        gt_box_scaled = target_box * scale + offset
        loss_bbox = self.smooth_l1(pred_box_scaled, gt_box_scaled) + self.calc_iou_loss(pred_box_scaled, gt_box_scaled).mean()
        
        # 2. 最终输出的 Velocity Loss (MSE)
        loss_vel = self.mse(pred_vel, target_vel)
        # loss_vel += 1.0 - F.cosine_similarity(pred_vel, target_vel, dim=-1).mean()  # 加入余弦相似度损失，鼓励方向一致
        
        # 3. 本机推演辅助 Loss (让 Ego 分支能够独立行走)
        loss_ego_aux = self.smooth_l1(pred_ego_only, target)
        
        # 4. 门控正则化 Loss (如果该帧被 Mask 掉了，强行把 alpha 压到 0)
        # alpha 是 Sigmoid 输出，取值 0~1
        expected_alpha = 1.0 - peer_mask
        loss_gate = self.bce(alpha, expected_alpha)
        
        # 总 Loss 加权
        total_loss = (self.l1 * loss_bbox + 
                      self.l2 * loss_vel)
                      
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_vel': loss_vel.item(),
            'loss_ego': loss_ego_aux.item(),
            'loss_gate': loss_gate.item()
        }
    
class CollaborativeTrackingLoss(nn.Module):
    def __init__(self, lambda_bbox=1.0, lambda_vel=0.5, lambda_ego=0.2, lambda_gate=0.6):
        super().__init__()
        self.l1 = lambda_bbox
        self.l2 = lambda_vel
        self.l3 = lambda_ego
        self.l4 = lambda_gate
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def calc_iou_loss(self, pred_box, gt_box, eps=1e-7):
        p_umin, p_umax, p_vmin, p_vmax = pred_box.unbind(-1)
        t_umin, t_umax, t_vmin, t_vmax = gt_box.unbind(-1)
       
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
        return 1.0 - iou  # Loss 形式，越小越好

        



    def calc_diou_loss(self, pred_box, gt_box, eps=1e-7):
        """
        简化版 DIoU 计算 (假设输入为 [B, 4] -> tl_x, tl_y, br_x, br_y)
        实际论文工程中建议使用 torchvision.ops.distance_box_iou
        """
        """
        计算 CIoU (Complete IoU)
        参数:
            pred_boxes: 预测框 [N, 4], 格式为 [umin, umax, vmin, vmax]
            target_boxes: 真实框 [N, 4], 格式同上
        """
        # 分离坐标
        p_umin, p_umax, p_vmin, p_vmax = pred_box.unbind(-1)
        t_umin, t_umax, t_vmin, t_vmax = gt_box.unbind(-1)
        # p_umin = (p_umin + 0.5) * 960
        # p_umax = (p_umax + 0.5) * 960
        # p_vmin = (p_vmin + 0.5) * 540
        # p_vmax = (p_vmax + 0.5) * 540
        # t_umin = (t_umin + 0.5) * 960
        # t_umax = (t_umax + 0.5) * 960
        # t_vmin = (t_vmin + 0.5) * 540
        # t_vmax = (t_vmax + 0.5) * 540


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
        ciou = torch.clamp(ciou, min=-1.0, max=1.0)  # 确保 CIoU 在合理范围内
        return 1.0 - ciou  # Loss 形式，越小越好

    def forward(self, pred_final, pred_ego_only, alpha, target, peer_mask):
        """
        pred_final: [B, 6] 最终输出 (tl, br, vel)
        pred_ego_only: [B, 6] 仅靠本机自运动预测的输出
        alpha: [B, 1] 僚机信息采纳度
        target: [B, 6] 真实的 Ground Truth
        peer_mask: [B, 1] 布尔/浮点张量，1.0 表示该样本的僚机数据被刻意遮挡(丢弃)了，0.0表示正常
        """
        # 拆解目标
        pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
        target_box, target_vel = target[:, :4], target[:, 4:]
        
        # 1. 最终输出的 BBox Loss (Smooth L1 + DIoU)
        # loss_bbox = self.smooth_l1(pred_box, target_box)
        scale = torch.tensor([960.0, 960.0, 540.0, 540.0], device=pred_box.device)
        offset = torch.tensor([480.0, 480.0, 270.0, 270.0], device=pred_box.device)

        pred_box_scaled = pred_box * scale + offset
        gt_box_scaled = target_box * scale + offset
        loss_bbox = self.smooth_l1(pred_box_scaled, gt_box_scaled) + self.calc_iou_loss(pred_box_scaled, gt_box_scaled).mean()
        # loss_bbox = 200 * self.smooth_l1(pred_box, target_box) + self.calc_diou_loss(pred_box, target_box).mean()
        
        # 2. 最终输出的 Velocity Loss (MSE)
        loss_vel = self.mse(pred_vel, target_vel)
        # loss_vel += 1.0 - F.cosine_similarity(pred_vel, target_vel, dim=-1).mean()  # 加入余弦相似度损失，鼓励方向一致
        
        # 3. 本机推演辅助 Loss (让 Ego 分支能够独立行走)
        loss_ego_aux = self.smooth_l1(pred_ego_only, target)
        
        # 4. 门控正则化 Loss (如果该帧被 Mask 掉了，强行把 alpha 压到 0)
        # alpha 是 Sigmoid 输出，取值 0~1
        expected_alpha = 1.0 - peer_mask
        loss_gate = self.bce(alpha, expected_alpha)
        
        # 总 Loss 加权
        total_loss = (self.l1 * loss_bbox + 
                      self.l2 * loss_vel + 
                      self.l3 * loss_ego_aux + 
                      self.l4 * loss_gate)
                      
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_vel': loss_vel.item(),
            'loss_ego': loss_ego_aux.item(),
            'loss_gate': loss_gate.item()
        }




# =====================================================================
# 1. 所提算法: 门控时空交叉注意力网络 (Proposed ST-CAG)
# [包含完整的时间位置编码、双分支残差、自适应门控]
# =====================================================================
class SpatioTemporalFusionNet(nn.Module):
    def __init__(self, seq_len=5, hidden_dim=128, num_heads=4):
        super().__init__()
        self.seq_len = seq_len
        input_dim = 6 + 7 
        
        self.time_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        self.q_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.k_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.v_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.ego_motion_decoder = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 6))
        self.peer_correction_decoder = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 6))
        self.gating = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # nn.init.constant_(self.gating[0].bias, 2.0)

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        ego_input = torch.cat([ego_obs, ego_pose], dim=-1)     
        peer_input = torch.cat([peer_obs, peer_pose], dim=-1) 
        
        Q = self.q_mlp(ego_input) + self.time_embed
        K = self.k_mlp(peer_input) + self.time_embed
        V = self.v_mlp(peer_input) + self.time_embed
        
        attn_out, _ = self.attn(Q, K, V) 
        
        Q_t = Q[:, -1, :]              
        attn_out_t = attn_out[:, -1, :] 
        base_obs_t = ego_obs[:, -1, :] 
        
        ego_delta = self.ego_motion_decoder(Q_t) 
        peer_correction = self.peer_correction_decoder(torch.cat([Q_t, attn_out_t], dim=-1)) 
        alpha = self.gating(attn_out_t) 
        
        pred_ego_only = base_obs_t + ego_delta
        pred_final = pred_ego_only + alpha * peer_correction
        # pred_final = pred_ego_only.detach() + alpha * peer_correction
        
        return pred_final, pred_ego_only, alpha

# =====================================================================
# 2. 对比基线: 纯多层感知机展平拼接 (Concat-MLP Baseline)
# [破坏时序结构与几何分离特征，强行暴力拟合]
# =====================================================================
class ConcatMLPBaseline(nn.Module):
    def __init__(self, seq_len=5, hidden_dim=256):
        super().__init__()
        # 将 T 帧的所有数据全部展平: T * (6 + 6 + 7 + 7) = T * 26
        flatten_dim = seq_len * 26
        
        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6) # 直接输出残差
        )

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        B = ego_obs.size(0)
        # 拼接所有特征 [B, T, 26]
        all_features = torch.cat([ego_obs, peer_obs, ego_pose, peer_pose], dim=-1)
        # 展平为一维向量 [B, T*26]
        flattened = all_features.view(B, -1)
        
        delta = self.mlp(flattened)
        base_obs_t = ego_obs[:, -1, :]
        pred_final = base_obs_t + delta
        
        # 返回 dummy 变量适配训练代码
        dummy_alpha = torch.zeros(B, 1, device=ego_obs.device)
        return pred_final, pred_final.detach(), dummy_alpha

# =====================================================================
# 3. 对比基线: 直接回归交叉注意力 (Direct-Reg-CrossAttn Baseline)
# [剥离了双分支残差与门控，评估门控机制的必要性]
# =====================================================================
class DirectRegCrossAttnBaseline(nn.Module):
    def __init__(self, seq_len=5, hidden_dim=128, num_heads=4):
        super().__init__()
        input_dim = 6 + 7 
        self.time_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        self.q_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.k_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.v_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 直接从融合特征回归 6D 输出，没有独立的 ego_motion_decoder
        self.direct_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        ego_input = torch.cat([ego_obs, ego_pose], dim=-1)     
        peer_input = torch.cat([peer_obs, peer_pose], dim=-1) 
        
        Q = self.q_mlp(ego_input) + self.time_embed
        K = self.k_mlp(peer_input) + self.time_embed
        V = self.v_mlp(peer_input) + self.time_embed
        
        attn_out, _ = self.attn(Q, K, V) 
        attn_out_t = attn_out[:, -1, :] 
        
        delta = self.direct_decoder(attn_out_t)
        base_obs_t = ego_obs[:, -1, :]
        pred_final = base_obs_t + delta
        
        dummy_alpha = torch.zeros(base_obs_t.size(0), 1, device=ego_obs.device)
        return pred_final, pred_final.detach(), dummy_alpha
    

class SelfAttnBaseline(nn.Module):
    def __init__(self, seq_len=5, hidden_dim=128, num_heads=4):
        super().__init__()
        input_dim = 6 + 7 
        self.time_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        self.q_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.k_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.v_mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # 直接从融合特征回归 6D 输出，没有独立的 ego_motion_decoder
        self.direct_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        ego_input = torch.cat([ego_obs, ego_pose], dim=-1)     
        peer_input = torch.cat([peer_obs, peer_pose], dim=-1) 
        
        Q = self.q_mlp(ego_input) + self.time_embed
        K = self.k_mlp(ego_input) + self.time_embed
        V = self.v_mlp(ego_input) + self.time_embed
        
        attn_out, _ = self.attn(Q, K, V) 
        attn_out_t = attn_out[:, -1, :] 
        
        delta = self.direct_decoder(attn_out_t)
        base_obs_t = ego_obs[:, -1, :]
        pred_final = base_obs_t + delta
        
        dummy_alpha = torch.zeros(base_obs_t.size(0), 1, device=ego_obs.device)
        return pred_final, pred_final.detach(), dummy_alpha

# =====================================================================
# 4. 对比基线: LSTM 时序记忆网络 (LSTM Baseline)
# [经典的 RNN 变体，依靠内部细胞状态(Cell State)进行历史推演]
# =====================================================================
class LSTMBaseline(nn.Module):
    def __init__(self, seq_len=5, input_dim=26, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        B = ego_obs.size(0)
        # 拼接每一帧的所有特征 [B, T, 26]
        seq_input = torch.cat([ego_obs, peer_obs, ego_pose, peer_pose], dim=-1)
        
        # lstm_out shape: [B, T, hidden_dim]
        # hn shape: [num_layers, B, hidden_dim]
        lstm_out, (hn, cn) = self.lstm(seq_input)
        
        # 取最后一层的隐藏状态作为时序推演的特征聚合
        last_hidden = hn[-1] 
        
        delta = self.decoder(last_hidden)
        base_obs_t = ego_obs[:, -1, :]
        pred_final = base_obs_t + delta
        
        dummy_alpha = torch.zeros(B, 1, device=ego_obs.device)
        return pred_final, pred_final.detach(), dummy_alpha

# =====================================================================
# 5. 对比基线: 时间卷积网络 (TCN Baseline)
# [利用 1D 卷积和感受野处理时序特征，在轨迹预测中极受欢迎]
# =====================================================================
class TCNBaseline(nn.Module):
    def __init__(self, seq_len=5, input_dim=26, hidden_dim=64):
        super().__init__()
        # 对于 T=5 的短序列，使用两个普通 1D 卷积就能覆盖全部感受野
        # 输入形状要求：[B, Channels, Length]
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        # 经过 conv1 后，序列长度变为: 5 - 3 + 1 = 3
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        # 经过 conv2 后，序列长度变为: 3 - 3 + 1 = 1
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        B = ego_obs.size(0)
        # [B, T, 26]
        seq_input = torch.cat([ego_obs, peer_obs, ego_pose, peer_pose], dim=-1)
        
        # PyTorch Conv1d 期待的输入是 [Batch, Channels, Length]
        # 需要进行维度转置 [B, 26, T]
        x = seq_input.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x) # 此时 x 的形状为 [B, 128, 1]
        
        # 挤压掉最后的长度维度 1 -> [B, 128]
        feature = x.squeeze(-1)
        
        delta = self.decoder(feature)
        base_obs_t = ego_obs[:, -1, :]
        pred_final = base_obs_t + delta
        
        dummy_alpha = torch.zeros(B, 1, device=ego_obs.device)
        return pred_final, pred_final.detach(), dummy_alpha

# =====================================================================
# 6. 对比基线: 单帧空间交叉注意力 (Spatial-CrossAttn T=1 Baseline)
# [剥离时序维度，验证滑动时间窗的必要性]
# =====================================================================
class SpatialCrossAttnBaseline(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        # 复用 Proposed 的结构，但强制 T=1，去除了时间位置编码
        self.model = SpatioTemporalFusionNet(seq_len=1, hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(self, ego_obs, peer_obs, ego_pose, peer_pose):
        # 强行截取 Dataset 传入的 5 帧序列中的最后 1 帧 (即当前瞬时状态)
        # 截取后保持序列维度维度以适配 Attention，形状变为 [B, 1, Feature_dim]
        ego_obs_single = ego_obs[:, -1:, :]
        peer_obs_single = peer_obs[:, -1:, :]
        ego_pose_single = ego_pose[:, -1:, :]
        peer_pose_single = peer_pose[:, -1:, :]
        
        # 调用核心网络
        return self.model(ego_obs_single, peer_obs_single, ego_pose_single, peer_pose_single)