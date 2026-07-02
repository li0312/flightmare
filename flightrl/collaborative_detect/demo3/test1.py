'''
Author: Lac_Creeper
Date: 2026-04-13 16:52:05 +0800
LastEditTime: 2026-04-13 17:34:32 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo3/test1.py
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data_loader import CollaborativeTrackingDataset
from model import SpatioTemporalFusionNet, CollaborativeTrackingLoss, ConcatMLPBaseline, DirectRegCrossAttnBaseline, SpatialCrossAttnBaseline, TCNBaseline, LSTMBaseline,SelfAttnBaseline

def calculate_iou(box1, box2):
    """
    计算两个边界框集合的 IoU (输入 shape: [B, 4] -> xmin, ymin, xmax, ymax)
    """
    # 确保坐标合法性 (xmin < xmax, ymin < ymax)
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

def calculate_angular_error(dir1, dir2):
    """
    计算两个向量之间的夹角误差 (Degree)
    """
    unit1 = dir1 / (np.linalg.norm(dir1, axis=-1, keepdims=True) + 1e-8)
    unit2 = dir2 / (np.linalg.norm(dir2, axis=-1, keepdims=True) + 1e-8)
    
    dot_product = np.sum(unit1 * unit2, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def evaluate_model(models_path, test_files, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集（测试时固定drop概率，也可以多次运行取平均）
    test_dataset = CollaborativeTrackingDataset(test_files, seq_len=seq_len,
                                                 prob_long_drop=0.2, prob_short_drop=0.1)
    test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False)
    
    # 定义模型列表和名称
    model_classes = [
        SpatioTemporalFusionNet,
        SelfAttnBaseline,
        DirectRegCrossAttnBaseline,
        ConcatMLPBaseline
    ]
    model_names = [
        "SpatioTemporalFusion",
        "SelfAttnBaseline",
        "DirectRegCrossAttn",
        "ConcatMLPBaseline"
    ]
    
    results = {}
    
    for idx, (ModelClass, name) in enumerate(zip(model_classes, model_names)):
        print(f"\n--- Evaluating {name} ---")
        # 初始化模型
        if ModelClass == SpatioTemporalFusionNet:
            model = ModelClass(seq_len=seq_len).to(device)
        else:
            # 其他基线模型可能不需要seq_len，根据实际定义调整
            model = ModelClass().to(device)
        
        # 加载权重
        model.load_state_dict(torch.load(models_path[idx], map_location=device))
        model.eval()
        
        # 指标累积
        total_iou = 0.0
        total_ego_iou = 0.0
        total_last_iou = 0.0
        total_ade = 0.0
        total_rmse_v = 0.0
        total_angle_error = 0.0
        samples_count = 0
        
        gating_records = []  # 存储每个样本的信息
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Testing {name}"):
                e_obs, p_obs, e_pose, p_pose, gt, p_mask = [x.to(device) for x in batch_data]

                if (name != "SpatioTemporalFusion"):
                    std = torch.tensor([8/960, 8/540, 8/960, 8/540, 0, 0], device=device)
                    e_obs += torch.randn_like(e_obs) * std

                
                # 模型推理（所有模型都应返回 (pred_final, pred_ego_only, alpha)）
                # 对于无门控的模型，alpha可设为None或全0，这里统一处理
                output = model(e_obs, p_obs, e_pose, p_pose)
                if len(output) == 3:
                    pred_final, pred_ego_only, alpha = output
                else:
                    # 兼容只返回两个值的情况
                    pred_final, pred_ego_only = output
                    alpha = torch.zeros(e_obs.size(0), device=device)  # 占位
                
                # 分离坐标与速度
                pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
                gt_box, gt_vel = gt[:, :4], gt[:, 4:]
                last_obs_box = e_obs[:, -1, :4]  # 最后一帧观测框
                
                # IoU计算
                iou = calculate_iou(pred_box, gt_box)
                total_iou += iou.sum().item()
                last_iou = calculate_iou(last_obs_box, gt_box)
                total_last_iou += last_iou.sum().item()
                ego_iou = calculate_iou(pred_ego_only[:, :4], gt_box)
                total_ego_iou += ego_iou.sum().item()
                
                # 中心点ADE
                pred_center = (pred_box[:, :2] + pred_box[:, 2:]) / 2.0
                gt_center = (gt_box[:, :2] + gt_box[:, 2:]) / 2.0
                pred_center[:, 0] = (pred_center[:, 0] + 0.5) * 960
                pred_center[:, 1] = (pred_center[:, 1] + 0.5) * 540
                gt_center[:, 0] = (gt_center[:, 0] + 0.5) * 960
                gt_center[:, 1] = (gt_center[:, 1] + 0.5) * 540
                ade = torch.norm(pred_center - gt_center, dim=1)
                total_ade += ade.sum().item()
                
                # 速度RMSE
                rmse_v = torch.sqrt(torch.mean((pred_vel - gt_vel)**2, dim=1))
                total_rmse_v += rmse_v.sum().item()
                
                # 角度误差
                angle_error = calculate_angular_error(pred_vel.cpu().numpy(), gt_vel.cpu().numpy())
                total_angle_error += np.sum(angle_error)
                
                samples_count += e_obs.size(0)
                
                # 记录样本级信息（用于后续绘图）
                for i in range(e_obs.size(0)):
                    gating_records.append({
                        "is_masked": p_mask[i].item(),
                        "alpha": alpha[i].item() if alpha is not None else 0.0,
                        "iou_final": iou[i].item(),
                        "iou_ego": ego_iou[i].item(),
                        "iou_last_obs": last_iou[i].item()
                    })
        
        # 计算平均指标
        avg_iou = total_iou / samples_count
        avg_last_iou = total_last_iou / samples_count
        avg_ego_iou = total_ego_iou / samples_count
        avg_ade = total_ade / samples_count
        avg_rmse_v = total_rmse_v / samples_count
        avg_angle_error = total_angle_error / samples_count
        
        print(f"\n--- {name} Test Results ---")
        print(f"Mean IoU:     {avg_iou:.4f}")
        print(f"Last Obs IoU: {avg_last_iou:.4f}")
        print(f"Ego-Only IoU: {avg_ego_iou:.4f}")
        print(f"Center ADE:   {avg_ade:.4f}")
        print(f"Vel RMSE:     {avg_rmse_v:.4f}")
        print(f"Angle Error:  {avg_angle_error:.2f} degrees")
        
        results[name] = {
            "metrics": {
                "iou": avg_iou,
                "last_iou": avg_last_iou,
                "ego_iou": avg_ego_iou,
                "ade": avg_ade,
                "rmse_v": avg_rmse_v,
                "angle_error": avg_angle_error
            },
            "gating_records": gating_records
        }
    
    return results


def plot_iou_curve_with_histogram(results, sequence_length=100, start_idx=0):
    """
    绘制所有模型的IoU曲线，顶部用直方图表示遮挡区域
    参数:
        results: evaluate_model返回的字典
        sequence_length: 连续帧数
        start_idx: 起始索引
    """
    # 假设所有模型的gating_records长度相同（来自同一数据集顺序）
    # 取第一个模型的记录确定时间轴
    first_model = list(results.keys())[0]
    seq_data = results[first_model]["gating_records"][start_idx:start_idx+sequence_length]
    time_steps = np.arange(len(seq_data))
    
    # 遮挡标记（1=遮挡，0=清晰）
    occlusion = [d["is_masked"] for d in seq_data]
    
    # 创建图形和两个子图（上：遮挡直方图，下：IoU曲线）
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
    
    # 上子图：遮挡直方图（bar）
    ax_top = fig.add_subplot(gs[0])
    ax_top.bar(time_steps, occlusion, width=1.0, color='gray', alpha=0.7, edgecolor='none')
    ax_top.set_ylabel('Occlusion', fontweight='bold')
    ax_top.set_ylim(-0.1, 1.1)
    ax_top.set_yticks([0, 1])
    ax_top.set_yticklabels(['Clear', 'Occluded'])
    ax_top.set_xlim(time_steps[0]-0.5, time_steps[-1]+0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.xaxis.set_visible(False)
    
    # 下子图：IoU曲线
    ax_bottom = fig.add_subplot(gs[1])
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']  # 为4个模型分配不同颜色
    linestyles = ['-', '--', '-.', ':']
    
    for (name, data), color, ls in zip(results.items(), colors, linestyles):
        seq_records = data["gating_records"][start_idx:start_idx+sequence_length]
        iou_final = [d["iou_final"] for d in seq_records]
        ax_bottom.plot(time_steps, iou_final, color=color, linewidth=2,
                       linestyle=ls, label=name)
    
    ax_bottom.set_xlabel('Time Step', fontweight='bold')
    ax_bottom.set_ylabel('Intersection over Union (IoU)', fontweight='bold')
    ax_bottom.set_ylim(0, 1.05)
    ax_bottom.set_xlim(time_steps[0]-0.5, time_steps[-1]+0.5)
    ax_bottom.grid(True, linestyle=':', alpha=0.6)
    ax_bottom.legend(loc='lower left')
    
    plt.suptitle("IoU Comparison Under Intermittent Occlusion", y=0.98, fontsize=14)
    plt.tight_layout()
    plt.savefig("iou_curves_all_models.pdf", format='pdf', dpi=300)
    plt.show()


def plot_performance_boxplot_all_models(results):
    """
    绘制所有模型的箱线图，按遮挡/清晰条件分组
    """
    records = []
    for model_name, data in results.items():
        for d in data["gating_records"]:
            condition = "Occluded" if d["is_masked"] == 1.0 else "Clear"
            # 记录融合IoU（即最终输出IoU）
            records.append({
                "Model": model_name,
                "Condition": condition,
                "IoU": d["iou_final"]
            })
            # 可选：同时记录ego-only IoU用于对比，这里不添加以保持简洁
    
    df = pd.DataFrame(records)
    
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.figure(figsize=(10, 6))
    
    # 绘制箱线图，使用hue分组
    sns.boxplot(x="Condition", y="IoU", hue="Model", data=df,
                palette="Set2", fliersize=2, linewidth=1.5)
    
    plt.title("Tracking Accuracy (IoU) Comparison Across Models", pad=15)
    plt.xlabel("Sensor Condition", fontweight='bold')
    plt.ylabel("Intersection over Union (IoU)", fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_boxplot_all_models.pdf", format='pdf', dpi=300)
    plt.show()


# 使用示例（假设已定义好模型路径列表和测试文件列表）
if __name__ == "__main__":
    models_path_list = [
        "checkpoints/Apr13_15-56-20_0/best_model.pth",
        "checkpoints/Apr13_16-09-42_1/best_model.pth",
        "checkpoints/Apr13_16-23-13_2/best_model.pth",
        "checkpoints/Apr13_16-36-34_3/best_model.pth"
    ]
    with open("/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/test_files.txt", "r") as f:
        test_list = [line.strip() for line in f.readlines()]
    # test_files_list = ["test_data1.pkl", "test_data2.pkl"]  # 实际测试文件
    
    # 评估所有模型
    results_all = evaluate_model(models_path_list, test_list, seq_len=5)
    
    # 绘图：IoU曲线（取前200帧）
    plot_iou_curve_with_histogram(results_all, sequence_length=100, start_idx=100)
    
    # 绘图：箱线图
    plot_performance_boxplot_all_models(results_all)