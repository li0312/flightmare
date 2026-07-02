'''
Author: Lac_Creeper
Date: 2026-04-06 04:10:26 +0800
LastEditTime: 2026-04-13 16:42:12 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo3/test.py
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

MODEL_TYPE = 1  # 0: SpatioTemporalFusionNet, 1: SelfAttnBaseline, 2: DirectRegCrossAttnBaseline, 3: ConcatMLPBaseline

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

def plot_gating_response(gating_data, sequence_length=100):
    """
    抽取一段连续的测试结果，展示网络如何根据遮挡动态调节门控权重 alpha
    """
    # 截取一段具有代表性的序列（确保里面有遮挡发生）
    seq_data = gating_data[200 : 200 + sequence_length]
    
    time_steps = np.arange(sequence_length)
    alphas = [d["alpha"] for d in seq_data]
    
    # 将 mask 翻转为可视度 (mask=1代表不可视，我们用 1-mask 表现可视状态)
    visibility = [1.0 - d["is_masked"] for d in seq_data] 
    
    # 设置学术字体
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12})
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # 画出 alpha 权重的平滑曲线
    ax1.plot(time_steps, alphas, color='navy', linewidth=2.5, label=r'Fusion Gating Weight ($\alpha$)')
    ax1.set_xlabel('Time Steps', fontweight='bold')
    ax1.set_ylabel(r'Gating Weight $\alpha$', color='navy', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.set_ylim(-0.1, 1.1)
    
    # 实例化一个共享 x 轴的辅助 y 轴
    ax2 = ax1.twinx()  
    
    # 用灰色阴影填充表示僚机处于物理遮挡状态
    ax2.fill_between(time_steps, 0, visibility, color='gray', alpha=0.3, step='mid', label='Peer Visibility (Sensor Valid)')
    ax2.set_ylabel('Sensor Validity (1=Valid, 0=Masked)', color='dimgray', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax2.set_ylim(-0.1, 1.1)
    
    # 整理图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title("Dynamic Gating Response Under Intermittent Occlusion", pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("gating_response.pdf", format='pdf', dpi=300)
    plt.show()


def plot_performance_boxplot(gating_data):
    """
    绘制消融对比图：单机推演 (Ego-Only) vs 协同融合 (Proposed Fusion)
    区分正常工况与遮挡工况
    """
    # 将记录字典转换为 Pandas DataFrame 以便于使用 Seaborn 绘图
    records = []
    for d in gating_data:
        condition = "Occluded" if d["is_masked"] == 1.0 else "Clear"
        
        # 记录单机基线的 IoU
        records.append({"Model": "Ego-Only Baseline", "Condition": condition, "IoU": d["iou_ego"]})
        # 记录融合网络的 IoU
        records.append({"Model": "Proposed Spatio-Temporal Fusion", "Condition": condition, "IoU": d["iou_final"]})
        
    df = pd.DataFrame(records)
    
    plt.rcParams.update({'font.family': 'serif',
                        # 'font.serif': ['Times New Roman'], 
                        'font.size': 14})
    plt.figure(figsize=(8, 6))
    
    # 使用 seaborn 绘制高颜值箱线图
    palette = {"Ego-Only Baseline": "lightcoral", "Proposed Spatio-Temporal Fusion": "lightskyblue"}
    sns.boxplot(x="Condition", y="IoU", hue="Model", data=df, palette=palette, fliersize=2, linewidth=1.5)
    
    plt.title("Tracking Accuracy (IoU) Comparison Under Different Conditions", pad=15)
    plt.xlabel("Sensor Condition", fontweight='bold')
    plt.ylabel("Intersection over Union (IoU)", fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("performance_boxplot.pdf", format='pdf', dpi=300)
    plt.show()


def plot_iou_curve(gating_data, sequence_length=100, start_idx=0):
    """
    绘制随时间变化的 IoU 曲线（Ego-Only vs. Proposed Fusion）
    并标注遮挡区域（is_masked == 1）

    参数:
        gating_data: list of dict, 每个元素包含 'iou_ego', 'iou_final', 'is_masked'
                     （建议按时间步顺序排列）
        sequence_length: 绘制的连续帧数
        start_idx: 起始索引
    """
    # 截取一段连续序列
    seq_data = gating_data[start_idx : start_idx + sequence_length]
    
    time_steps = np.arange(len(seq_data))
    iou_ego = [d["iou_ego"] for d in seq_data]
    iou_final = [d["iou_final"] for d in seq_data]
    occlusion = [d["is_masked"] for d in seq_data]   # 1=遮挡, 0=清晰
    
    # 设置学术风格
    # plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 12})
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 绘制 IoU 曲线
    ax.plot(time_steps, iou_ego, color='coral', linewidth=2, linestyle='--', label='Ego-Only Baseline')
    ax.plot(time_steps, iou_final, color='steelblue', linewidth=2.5, label='Proposed Spatio-Temporal Fusion')
    
    # 标注遮挡区域（灰色阴影）
    occlusion_regions = []
    start = None
    for t, occ in enumerate(occlusion):
        if occ == 1 and start is None:
            start = t
        elif occ == 0 and start is not None:
            ax.axvspan(start, t, alpha=0.2, color='gray', label='Occlusion' if start == 0 else "")
            start = None
    if start is not None:  # 如果最后一段是遮挡
        ax.axvspan(start, len(time_steps)-1, alpha=0.2, color='gray')
    
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Intersection over Union (IoU)', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')
    
    # 添加标题
    plt.title("IoU Comparison Under Intermittent Occlusion", pad=15)
    plt.tight_layout()
    plt.savefig("iou_curve.pdf", format='pdf', dpi=300)
    plt.show()

def calculate_angular_error(dir1, dir2):
    """
    计算两个向量之间的夹角误差 (Degree)
    """
    unit1 = dir1 / (np.linalg.norm(dir1, axis=-1, keepdims=True) + 1e-8)
    unit2 = dir2 / (np.linalg.norm(dir2, axis=-1, keepdims=True) + 1e-8)
    
    dot_product = np.sum(unit1 * unit2, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def evaluate_model_plus(models_path, test_files, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    models_num = 4
    # if (MODEL_TYPE == 0):
    #     model = SpatioTemporalFusionNet(seq_len=seq_len).to(device)
    # elif (MODEL_TYPE == 1):
    #     model = SelfAttnBaseline().to(device)
    # elif (MODEL_TYPE == 2):
    #     model = DirectRegCrossAttnBaseline().to(device)
    # elif (MODEL_TYPE == 3):
    #     model = ConcatMLPBaseline().to(device)
    models.append(SpatioTemporalFusionNet(seq_len=seq_len).to(device))
    models.append(SelfAttnBaseline().to(device))
    models.append(DirectRegCrossAttnBaseline().to(device))
    models.append(ConcatMLPBaseline().to(device))
    for i in range(models_num):
        models[i].load_state_dict(torch.load(models_path[i], 
                                             map_location=device))
        models[i].eval()

    steps_per_episode = 2000
    chunk_size = 40
    ego_obs_seq = torch.zeros((steps_per_episode, 6))
    peer_obs_seq = torch.zeros((steps_per_episode, 6))
    ego_pose_seq = torch.zeros((steps_per_episode, 7))
    peer_pose_seq = torch.zeros((steps_per_episode, 7))
    gt_obs_seq = torch.zeros((steps_per_episode, 6))
    file_num = 0
    for file_path in test_files:
        with open(file_path, 'r') as f:
            chunk_data = json.load(f)
            for t, step_data in enumerate(chunk_data):
                id = file_num * chunk_size + t
                ego_bbox = step_data["bbox_a_last"]
                peer_bbox = step_data["bbox_b"]
                gt_bbox = step_data["bbox_a"]
                ego_obs = np.array([ego_bbox[0], ego_bbox[1], ego_bbox[2], ego_bbox[3], step_data["motion_a_last"][0], step_data["motion_a_last"][1]])
                peer_obs = np.array([peer_bbox[0], peer_bbox[1], peer_bbox[2], peer_bbox[3], step_data["motion_b"][0], step_data["motion_b"][1]])
                gt_obs = np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], step_data["motion_a"][0], step_data["motion_a"][1]])
                ego_obs[0] += np.random.normal(0, 8/960)
                ego_obs[1] += np.random.normal(0, 8/540)
                ego_obs[2] += np.random.normal(0, 8/960)
                ego_obs[3] += np.random.normal(0, 8/540)

                ego_pose = np.array(step_data["pose_a_delta"])
                peer_pose = np.array(step_data["pose_b_a"])
                ego_obs_seq[id] = torch.from_numpy(ego_obs)
                peer_obs_seq[id] = torch.from_numpy(peer_obs)
                ego_pose_seq[id] = torch.from_numpy(ego_pose)
                peer_pose_seq[id] = torch.from_numpy(peer_pose)
                gt_obs_seq[id] = torch.from_numpy(gt_obs)
        file_num += 1

    gating_records = []
    total_iou = 0.0
    total_good_iou = 0.0
    total_ego_iou = 0.0
    total_last_iou = 0.0
    total_ade = 0.0
    total_rmse_v = 0.0
    total_angle_error = 0.0
    samples_count = 0
    for start_idx in range(steps_per_episode - seq_len + 1):
        e_obs = ego_obs_seq[start_idx : start_idx + seq_len].unsqueeze(0).to(device)
        p_obs = peer_obs_seq[start_idx : start_idx + seq_len].unsqueeze(0).to(device)
        e_pose = ego_pose_seq[start_idx : start_idx + seq_len].unsqueeze(0).to(device)
        p_pose = peer_pose_seq[start_idx : start_idx + seq_len].unsqueeze(0).to(device)
        gt = gt_obs_seq[start_idx + seq_len - 1].unsqueeze(0).to(device)

        with torch.no_grad():
            for model in models:
                pred_final, pred_ego_only, alpha = model(e_obs, p_obs, e_pose, p_pose)

                pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
                gt_box, gt_vel = gt[:, :4], gt[:, 4:]
                last_obs_box = e_obs[:, -1, :4]
                iou = calculate_iou(pred_box, gt_box)
                total_iou += iou.sum().item()
                last_iou = calculate_iou(last_obs_box, gt_box)
                total_last_iou += last_iou.sum().item()
                ego_iou = calculate_iou(pred_ego_only[:, :4], gt_box)
                total_ego_iou += ego_iou.sum().item()

                pred_center = (pred_box[:, :2] + pred_box[:, 2:]) / 2.0
                gt_center = (gt_box[:, :2] + gt_box[:, 2:]) / 2.0
                ade = torch.norm(pred_center - gt_center, dim=1)
                total_ade += ade.sum().item()

                rmse_v = torch.sqrt(torch.mean((pred_vel - gt_vel)**2, dim=1))
                total_rmse_v += rmse_v.sum().item()
                angle_error = calculate_angular_error(pred_vel.cpu().numpy(), gt_vel.cpu().numpy())
                total_angle_error += np.sum(angle_error)
                
                samples_count += e_obs.size(0)
                # for i in 

            

                

        
        



def evaluate_model(models_path, test_files, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集，测试时可以将 prob_long_drop 设为不同值进行多次测试
    test_dataset = CollaborativeTrackingDataset(test_files, seq_len=seq_len, prob_long_drop=0.2, prob_short_drop=0.1)
    test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False)
    models = []
    models_num = 4
    # if (MODEL_TYPE == 0):
    #     model = SpatioTemporalFusionNet(seq_len=seq_len).to(device)
    # elif (MODEL_TYPE == 1):
    #     model = SelfAttnBaseline().to(device)
    # elif (MODEL_TYPE == 2):
    #     model = DirectRegCrossAttnBaseline().to(device)
    # elif (MODEL_TYPE == 3):
    #     model = ConcatMLPBaseline().to(device)
    models.append(SpatioTemporalFusionNet(seq_len=seq_len).to(device))
    models.append(SelfAttnBaseline().to(device))
    models.append(DirectRegCrossAttnBaseline().to(device))
    models.append(ConcatMLPBaseline().to(device))
    for i in range(models_num):
        models[i].load_state_dict(torch.load(models_path[i], 
                                             map_location=device))
        models[i].eval()
    
    total_iou = 0.0
    total_good_iou = 0.0
    total_ego_iou = 0.0
    total_last_iou = 0.0
    total_ade = 0.0
    total_rmse_v = 0.0
    total_angle_error = 0.0
    samples_count = 0
    
    # 用于保存特定结果以供绘图
    gating_records = [] 
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            e_obs, p_obs, e_pose, p_pose, gt, p_mask = [x.to(device) for x in batch_data]
            
            # 推理
            pred_final, pred_ego_only, alpha = model(e_obs, p_obs, e_pose, p_pose)
            
            # 分离坐标与速度
            pred_box, pred_vel = pred_final[:, :4], pred_final[:, 4:]
            gt_box, gt_vel = gt[:, :4], gt[:, 4:]
            last_obs_box = e_obs[:, -1, :4]  # 最后一帧的观测边界框
            
            # 1. 计算 IoU
            iou = calculate_iou(pred_box, gt_box)
            total_iou += iou.sum().item()
            last_iou = calculate_iou(last_obs_box, gt_box)
            total_last_iou += last_iou.sum().item()
            ego_iou = calculate_iou(pred_ego_only[:, :4], gt_box)
            total_ego_iou += ego_iou.sum().item()
            
            # 2. 计算中心点 ADE
            pred_center = (pred_box[:, :2] + pred_box[:, 2:]) / 2.0
            gt_center = (gt_box[:, :2] + gt_box[:, 2:]) / 2.0
            ade = torch.norm(pred_center - gt_center, dim=1)
            total_ade += ade.sum().item()
            
            # 3. 计算速度 RMSE
            rmse_v = torch.sqrt(torch.mean((pred_vel - gt_vel)**2, dim=1))
            total_rmse_v += rmse_v.sum().item()
            angle_error = calculate_angular_error(pred_vel.cpu().numpy(), gt_vel.cpu().numpy())
            total_angle_error += np.sum(angle_error)
            
            
            samples_count += e_obs.size(0)
            
            # 记录门控权重和对应的遮挡状态 (用于后续画图)
            for i in range(e_obs.size(0)):
                gating_records.append({
                    "is_masked": p_mask[i].item(),
                    "alpha": alpha[i].item(),
                    "iou_final": iou[i].item(),
                    "iou_ego": calculate_iou(last_obs_box[i:i+1, :4], gt_box[i:i+1, :4]).item()
                })

    avg_iou = total_iou / samples_count
    avg_last_iou = total_last_iou / samples_count
    avg_ego_iou = total_ego_iou / samples_count
    avg_ade = total_ade / samples_count
    avg_rmse_v = total_rmse_v / samples_count
    avg_angle_error = total_angle_error / samples_count
    
    print(f"\n--- Test Results ---")
    print(f"Mean IoU:     {avg_iou:.4f}")
    print(f"Last Obs IoU: {avg_last_iou:.4f}")
    print(f"Ego-Only IoU: {avg_ego_iou:.4f}")
    print(f"Center ADE:   {avg_ade:.4f}")
    print(f"Vel RMSE:     {avg_rmse_v:.4f}")
    print(f"Angle Error:  {avg_angle_error:.2f} degrees")
    
    return gating_records

# 运行测试
with open("/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/test_files.txt", "r") as f:
        test_list = [line.strip() for line in f.readlines()]
gating_data = evaluate_model("checkpoints/Apr13_01-35-00/spatiotemporal_net_ep50.pth", test_list)

plot_iou_curve(gating_data, sequence_length=150, start_idx=20)
# plot_performance_boxplot(gating_data)
# plot_gating_response(gating_data)
