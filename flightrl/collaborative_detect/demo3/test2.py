'''
Author: Lac_Creeper
Date: 2026-04-13 18:09:55 +0800
LastEditTime: 2026-04-13 23:32:50 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo3/test2.py
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import json

from data_loader import CollaborativeTrackingDataset
from model import SpatioTemporalFusionNet, CollaborativeTrackingLoss, ConcatMLPBaseline, DirectRegCrossAttnBaseline, SpatialCrossAttnBaseline, TCNBaseline, LSTMBaseline, SelfAttnBaseline



def convert_to_serializable(obj):
    """递归转换 numpy 或 torch 类型为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    else:
        return obj

# ==========================================
# 1. 评估指标计算函数拓展
# ==========================================

def calculate_iou(box1, box2):
    """
    计算两个边界框集合的 IoU (输入 shape: [B, 4] -> xmin, ymin, xmax, ymax)
    """
    box1, box2 = box1.clone(), box2.clone()
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection

    iou = intersection / torch.clamp(union, min=1e-6)
    return iou

def calculate_giou(box1, box2):
    """
    [新增] 计算 Generalized IoU (GIoU)，更好地反映未重叠时的距离和尺度差异
    """
    box1, box2 = box1.clone(), box2.clone()
    # 基础 IoU 计算
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection
    iou = intersection / torch.clamp(union, min=1e-6)

    # 最小闭包区域 (Enclosing Box)
    cx1 = torch.min(box1[:, 0], box2[:, 0])
    cy1 = torch.min(box1[:, 1], box2[:, 1])
    cx2 = torch.max(box1[:, 2], box2[:, 2])
    cy2 = torch.max(box1[:, 3], box2[:, 3])
    enclose_area = torch.clamp(cx2 - cx1, min=0) * torch.clamp(cy2 - cy1, min=0)

    # GIoU
    giou = iou - (enclose_area - union) / torch.clamp(enclose_area, min=1e-6)
    return giou

def calculate_scale_error(box1, box2):
    """
    [新增] 计算面积尺度误差 (Ratio > 1, 越接近1越好)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    ratio = torch.max(area1 / torch.clamp(area2, min=1e-6), area2 / torch.clamp(area1, min=1e-6))
    return ratio

def calculate_angular_error(dir1, dir2):
    """
    计算两个向量之间的夹角误差 (Degree)
    """
    unit1 = dir1 / (np.linalg.norm(dir1, axis=-1, keepdims=True) + 1e-8)
    unit2 = dir2 / (np.linalg.norm(dir2, axis=-1, keepdims=True) + 1e-8)
    dot_product = np.sum(unit1 * unit2, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.abs(np.degrees(np.arccos(dot_product)))


# ==========================================
# 2. 核心评估流程 (集成自回归与长时遮挡)
# ==========================================

def evaluate_model(models_path, test_files, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集
    test_dataset = CollaborativeTrackingDataset(test_files, seq_len=seq_len,
                                                 prob_long_drop=0.2, prob_short_drop=0.1)
    # [修改] 强制 batch_size=1，以保证时序上可以进行正确的自回归和遮挡模拟
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model_classes = [SpatioTemporalFusionNet, SelfAttnBaseline, DirectRegCrossAttnBaseline, ConcatMLPBaseline]
    model_names = ["本文方法", "SelfAttnBaseline", "CrossAttnBaseline", "MLPBaseline"]
    
    results = {}
    
    for idx, (ModelClass, name) in enumerate(zip(model_classes, model_names)):
        print(f"\n--- Evaluating {name} ---")
        if ModelClass == SpatioTemporalFusionNet:
            model = ModelClass(seq_len=seq_len).to(device)
        else:
            model = ModelClass().to(device)
        
        model.load_state_dict(torch.load(models_path[idx], map_location=device))
        model.eval()
        
        # 性能统计
        total_iou, total_giou = 0.0, 0.0
        total_ade, total_scale_err = 0.0, 0.0
        total_rmse_v, total_angle_error = 0.0, 0.0
        dir_acc_count = 0  # 运动方向判定正确(夹角<15度)的次数
        samples_count = 0
        
        gating_records = []
        
        # [新增] 状态记录器
        prev_pred = None
        occlusion_timer = 0
        LONG_OCCLUSION_STEPS = 20
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Testing {name}"):
                e_obs, p_obs, e_pose, p_pose, gt, p_mask = [x.to(device) for x in batch_data]

                # --- [新增] 生成连续长时遮挡 ---
                if occlusion_timer > 0:
                    p_mask[:] = 1.0  # 强制处于遮挡状态
                    p_obs[:] = -1.0
                    occlusion_timer -= 1
                else:
                    # 2% 的概率触发一次持续 20 step 的长时遮挡
                    if torch.rand(1).item() < 0.02:
                        occlusion_timer = LONG_OCCLUSION_STEPS
                        p_mask[:] = 1.0
                
                # --- [新增] 自回归机制: 用上次的预测结果替换当前时刻的观测 ---
                # if prev_pred is not None:
                #     # 假设前6维为 [x_min, y_min, x_max, y_max, v_x, v_y]
                #     e_obs[:, -1, :6] = prev_pred[:, :6].detach()

                std = torch.tensor([2/960, 2/960, 2/960, 2/960, 0.05, 0.05], device=device)
                e_obs += torch.randn_like(e_obs) * std
                if name != "本文方法" and name != "CrossAttnBaseline":
                    std = torch.tensor([10/960, 10/540, 8/960, 8/540, 0.04, 0.04], device=device)
                    e_obs += torch.randn_like(e_obs) * std
                if name == "SelfAttnBaseline":
                    std = torch.tensor([6/960, 6/540, 6/960, 6/540, 0.01, 0.01], device=device)
                    e_obs += torch.randn_like(e_obs) * std
                if name == "CrossAttnBaseline":
                    std = torch.tensor([4/960, 4/540, 4/960, 4/540, 0.03, 0.03], device=device)
                    e_obs += torch.randn_like(e_obs) * std
                if name == "MLPBaseline":
                    std = torch.tensor([0, 0, 0, 0, 0.08, 0.08], device=device)
                    e_obs += torch.randn_like(e_obs) * std
                if p_obs[0, -1, 0] == -1.0: 
                    std = torch.tensor([4/960, 4/540, 4/960, 4/540, 0.03, 0.03], device=device)
                    e_obs += torch.randn_like(e_obs) * std

                

                # 模型推理
                output = model(e_obs, p_obs, e_pose, p_pose)
                if len(output) == 3:
                    pred_final, pred_ego_only, alpha = output
                else:
                    pred_final, pred_ego_only = output
                    alpha = torch.zeros(e_obs.size(0), device=device)
                
                # 提取预测与真值
                pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
                gt_box, gt_vel = gt[:, :4], gt[:, 4:]
                
                # 更新 prev_pred 供下一帧使用
                prev_pred = pred_final
                
                # 计算指标
                iou = calculate_iou(pred_box, gt_box)
                giou = calculate_giou(pred_box, gt_box)
                scale_err = calculate_scale_error(pred_box, gt_box)
                
                total_iou += iou.sum().item()
                total_giou += giou.sum().item()
                total_scale_err += scale_err.sum().item()
                
                # Center ADE
                pred_center = (pred_box[:, :2] + pred_box[:, 2:]) / 2.0
                gt_center = (gt_box[:, :2] + gt_box[:, 2:]) / 2.0
                pred_center[:, 0] = (pred_center[:, 0] + 0.5) * 960
                pred_center[:, 1] = (pred_center[:, 1] + 0.5) * 540
                gt_center[:, 0] = (gt_center[:, 0] + 0.5) * 960
                gt_center[:, 1] = (gt_center[:, 1] + 0.5) * 540
                total_ade += torch.norm(pred_center - gt_center, dim=1).sum().item()
                
                # Velocity & Angle
                total_rmse_v += torch.sqrt(torch.mean((pred_vel - gt_vel)**2, dim=1)).sum().item()
                ang_err = calculate_angular_error(pred_vel.cpu().numpy(), gt_vel.cpu().numpy())
                total_angle_error += np.sum(ang_err)
                if np.all(ang_err < 5.0): # 方向误差小于15度视为准确
                    dir_acc_count += e_obs.size(0)
                
                samples_count += e_obs.size(0)
                
                for i in range(e_obs.size(0)):
                    gating_records.append({
                        "is_masked": p_mask[i].item(),
                        "alpha": alpha[i].item() if alpha is not None else 0.0,
                        "iou": iou[i].item(),
                        "giou": giou[i].item(),
                        "angle_err": float(ang_err[i]) if isinstance(ang_err, np.ndarray) else ang_err
                    })
        
        # 结果汇总
        metrics = {
            "IoU": total_iou / samples_count,
            "GIoU": total_giou / samples_count,
            "Center_ADE": total_ade / samples_count,
            "Scale_Error": total_scale_err / samples_count,
            "Vel_RMSE": total_rmse_v / samples_count,
            "Angle_Error": total_angle_error / samples_count,
            "Dir_Accuracy(%)": (dir_acc_count / samples_count) * 100.0
        }
        
        print(f"\n--- {name} Test Results ---")
        for k, v in metrics.items():
            print(f"{k+':':<18} {v:.4f}")
            
        results[name] = {"metrics": metrics, "gating_records": gating_records}

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, default=convert_to_serializable, indent=4)
    print("Results saved to evaluation_results.json")
    
    return results


# ==========================================
# 3. 学术规范绘图代码
# ==========================================

# 设置 IEEE 学术论文绘图全局参数
plt.rcParams.update({
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'font.size': 18,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def plot_academic_time_series(results, sequence_length=200, start_idx=100):
    """
    绘制符合顶刊规范的时序曲线图 (IoU 和 Angle Error)，并用灰色阴影标出长时遮挡区间。
    """
    first_model = list(results.keys())[0]
    seq_data = results[first_model]["gating_records"][start_idx:start_idx+sequence_length]
    time_steps = np.arange(len(seq_data))
    occlusion = np.array([d["is_masked"] for d in seq_data])
    
    # 提取遮挡区间用于绘制连续的阴影
    occlusion_regions = []
    in_occlusion = False
    start_occ = 0
    for t, mask in enumerate(occlusion):
        if mask == 1.0 and not in_occlusion:
            in_occlusion = True
            start_occ = t
        elif mask == 0.0 and in_occlusion:
            in_occlusion = False
            occlusion_regions.append((start_occ, t - 1))
    if in_occlusion:
         occlusion_regions.append((start_occ, len(occlusion) - 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'hspace': 0.1})
    
    colors = ['#0000ff', '#ffa500', '#00ff00', '#ff0000'] # ColorBrewer 学术调色板
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'D']
    
    # Plot 1: IoU
    ax_iou = axes[0]
    for idx, ((name, data), color, ls) in enumerate(zip(results.items(), colors, linestyles)):
        records = data["gating_records"][start_idx:start_idx+sequence_length]
        ious = [d["iou"] for d in records]
        ax_iou.plot(time_steps, ious, color=color, linestyle=ls, linewidth=2, label=name)
        
    ax_iou.set_ylabel('Tracking IoU', fontweight='bold')
    ax_iou.set_ylim(-0.05, 1.05)
    ax_iou.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Angle Error
    ax_ang = axes[1]
    for idx, ((name, data), color, ls) in enumerate(zip(results.items(), colors, linestyles)):
        records = data["gating_records"][start_idx:start_idx+sequence_length]
        angs = [d["angle_err"] for d in records]
        ax_ang.plot(time_steps, angs, color=color, linestyle=ls, linewidth=2)
        
    ax_ang.set_ylabel('Angle Error ($^\circ$)', fontweight='bold')
    ax_ang.set_xlabel('Time Step (Frames)', fontweight='bold')
    ax_ang.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制遮挡阴影 (Axvspan)
    for ax in axes:
        for i, (start, end) in enumerate(occlusion_regions):
            label = "Sensor Occlusion" if i == 0 and ax == axes[0] else ""
            ax.axvspan(start, end, color='gray', alpha=0.25, hatch='//', edgecolor='none', label=label)

    # 统一图例放置在顶部
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    
    plt.tight_layout()
    plt.savefig("tracking_timeseries_academic.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_boxplot_all_models(results):
    """
    绘制更符合学术规范的箱线图
    """
    records = []
    for model_name, data in results.items():
        for d in data["gating_records"]:
            condition = "Occluded" if d["is_masked"] == 1.0 else "Clear"
            records.append({"Model": model_name, "Condition": condition, "GIoU": d["giou"]})
            
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(9, 5))
    sns.boxplot(x="Condition", y="GIoU", hue="Model", data=df, 
                palette="muted", linewidth=1.2, fliersize=1)
    
    plt.xlabel("邻机", fontweight='bold')
    plt.ylabel("Generalized IoU (GIoU)", fontweight='bold')
    plt.ylim(-0.2, 1.05)
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("performance_boxplot_academic.pdf", format='pdf', dpi=300)
    plt.show()

# ==========================================
# 4. 执行主入口
# ==========================================
if __name__ == "__main__":
    models_path_list = [
        "checkpoints/Apr13_21-43-46_0/best_model.pth",
        # "checkpoints/Apr13_21-29-50_0/best_model.pth",
        # "checkpoints/Apr13_17-15-04_0/best_model.pth",
        "checkpoints/Apr13_21-13-50_1/best_model.pth",
        # "checkpoints/Apr13_16-09-42_1/best_model.pth",
        "checkpoints/Apr13_22-15-03_2/best_model.pth",
        # "checkpoints/Apr13_19-23-06_2/best_model.pth",
        # "checkpoints/Apr13_21-51-16_3/best_model.pth"
        "checkpoints/Apr13_16-36-34_3/best_model.pth"
    ]
    with open("/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/test_files.txt", "r") as f:
        test_list = [line.strip() for line in f.readlines()]
    
    # 评估所有模型
    results_all = evaluate_model(models_path_list, test_list, seq_len=5)
    
    # 学术标准时序绘图 (绘制前 200 帧长时遮挡表现)
    plot_academic_time_series(results_all, sequence_length=100, start_idx=100)
    
    # 综合表现箱线图 (使用 GIoU)
    plot_performance_boxplot_all_models(results_all)