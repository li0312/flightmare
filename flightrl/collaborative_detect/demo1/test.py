import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
from tqdm import tqdm

from data_loader import create_data_loaders
from models1 import LSTMNet, TemporalCNNNet, TransformerNet, CrossAttentionNet


def to_bbox6(x):
    center_u = x[:, 0]
    center_v = x[:, 1]
    height = x[:, 2]
    motion_x = x[:, 3]
    motion_y = x[:, 4]
    width = height / 2
    umin = center_u - width / 2
    umax = center_u + width / 2
    vmin = center_v - height / 2
    vmax = center_v + height / 2
    return np.stack([umin, umax, vmin, vmax, motion_x, motion_y], axis=-1)


def to_bbox4(x):
    """
    将 [center_u, center_v, width, height] 转换为 [umin, umax, vmin, vmax]
    """
    center_u, center_v, height = x
    width = height / 2
    umin = center_u - width / 2
    umax = center_u + width / 2
    vmin = center_v - height / 2
    vmax = center_v + height / 2
    return np.array([umin, umax, vmin, vmax])

def calculate_iou(box1, box2):
    """
    计算 IoU (Intersection over Union)
    box: [umin, umax, vmin, vmax]
    """
    inter_umin = np.maximum(box1[..., 0], box2[..., 0])
    inter_umax = np.minimum(box1[..., 1], box2[..., 1])
    inter_vmin = np.maximum(box1[..., 2], box2[..., 2])
    inter_vmax = np.minimum(box1[..., 3], box2[..., 3])

    inter_area = np.maximum(0, inter_umax - inter_umin) * np.maximum(0, inter_vmax - inter_vmin)
    area1 = (box1[..., 1] - box1[..., 0]) * (box1[..., 3] - box1[..., 2])
    area2 = (box2[..., 1] - box2[..., 0]) * (box2[..., 3] - box2[..., 2])
    
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-8)

def calculate_angular_error(dir1, dir2):
    """
    计算两个向量之间的夹角误差 (Degree)
    """
    unit1 = dir1 / (np.linalg.norm(dir1, axis=-1, keepdims=True) + 1e-8)
    unit2 = dir2 / (np.linalg.norm(dir2, axis=-1, keepdims=True) + 1e-8)
    
    dot_product = np.sum(unit1 * unit2, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型和归一化统计量
    checkpoint = torch.load(f'./checkpoints/best_{args.model}.pth', map_location=device)
    # model = LSTMNet(input_dim=16, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    model = CrossAttentionNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    if args.model == 'lstm':
        model = LSTMNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'attention':
        model = CrossAttentionNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'transformer':
        model = TransformerNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    # checkpoint = torch.load('./checkpoints/best_transformer.pth', map_location=device)
    # model = TransformerNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取训练时的归一化参数
    input_mean = checkpoint['input_mean']
    input_std = checkpoint['input_std']

    # 2. 数据加载 (关闭 augment)
    _, test_loader, _, test_dataset = create_data_loaders(
        data_dir='../dataset',
        train_meta='../dataset/test_files.txt', # 实际上这里可以用 test_meta
        val_meta='../dataset/test_files.txt',
        batch_size=1, # 测试时通常用 batch=1 方便分析序列
        sequence_length=20,
        num_workers=1
    )

    all_ious = []
    all_ious_last = []
    all_angle_errors = []

    print("Starting Evaluation...")
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.to(device).float()
            
            # 模型预测 [1, T, 6]
            pred_norm = model(batch_x).cpu().numpy()[0] 
            target_norm = batch_y.numpy()[0]

            # 3. 反归一化
            # # 注意：data_loader.py 中的 denormalize_output 用于 y
            # pred = test_dataset.denormalize_output(pred_norm)
            # target = test_dataset.denormalize_output(target_norm)
            pred = to_bbox6(pred_norm)
            target = to_bbox6(target_norm)

            # 4. 计算指标 (取序列最后一帧作为预测结果)
            for t in range(pred.shape[0]):
                pred_box = pred[t, :4]
                target_box = target[t, :4]
                last_box = batch_x.cpu().numpy()[0, t, 13:16]
                last_box = to_bbox4(last_box)
                iou_t = calculate_iou(pred_box, target_box)
                iou_last = calculate_iou(last_box, target_box)
                angle_err_t = calculate_angular_error(pred[t, 4:], target[t, 4:])
                all_ious.append(iou_t)
                all_ious_last.append(iou_last)
                all_angle_errors.append(abs(angle_err_t))
            # iou = calculate_iou(pred[-1, :4], target[-1, :4])
            # angle_err = calculate_angular_error(pred[-1, 4:], target[-1, 4:])
            # all_ious.append(iou)
            # all_angle_errors.append(abs(angle_err))

            # 可视化前 5 个样本
            if i < 10:
                x = batch_x.cpu().numpy()[0] # [T, 24]
                visualize_result(pred, target, x, i, args.save_dir)

    print(f"\n--- Test Results ---")
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Mean IoU_last: {np.mean(all_ious_last):.4f}")
    print(f"Mean Angular Error: {np.mean(all_angle_errors):.2f}°")
    print(f"Success Rate (IoU > 0.85): {np.mean(np.array(all_ious) > 0.85)*100:.2f}% | {np.sum(np.array(all_ious) > 0.85)} / {len(all_ious)}")
    print(f"Fail Rate (IoU < 0.5): {np.mean(np.array(all_ious) < 0.5)*100:.2f}% | {np.sum(np.array(all_ious) < 0.5)} / {len(all_ious)}")
    print(f"IoU min: {np.min(all_ious):.4f}, IoU max: {np.max(all_ious):.4f}")
    print(f"Angular Error min: {np.min(all_angle_errors):.2f}°, max: {np.max(all_angle_errors):.2f}°")
    print("---------------")
    print(f"Success Rate (IoU > 0.85): {np.mean(np.array(all_ious_last) > 0.85)*100:.2f}% | {np.sum(np.array(all_ious_last) > 0.85)} / {len(all_ious_last)}")
    print(f"Fail Rate (IoU < 0.5): {np.mean(np.array(all_ious_last) < 0.5)*100:.2f}% | {np.sum(np.array(all_ious_last) < 0.5)} / {len(all_ious_last)}")
    print(f"IoU min: {np.min(all_ious_last):.4f}, IoU max: {np.max(all_ious_last):.4f}")

def visualize_result(pred, target, x, idx, save_dir):
    """
    绘制全序列检测框：历史帧低透明度/细线，当前帧高透明度/粗线
    pred/target: [Seq, 6] (已反归一化)
    """
    os.makedirs(save_dir, exist_ok=True)
    seq_len = pred.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. 遍历序列中的每一帧进行绘制
    for t in range(seq_len):
        # 计算动态样式参数：随时间接近当前帧而增加
        # 透明度从 0.1 渐变到 0.8 (GT) 或 1.0 (Pred)
        alpha_val = 0.1 + 0.7 * (t / (seq_len - 1))
        # 线宽从 0.5 渐变到 2.0
        lw_val = 0.5 + 1.5 * (t / (seq_len - 1))
        
        # 提取当前帧的 box: [umin, umax, vmin, vmax]
        p_box = pred[t, :4]
        t_box = target[t, :4]
        b_box = to_bbox4(x[t, 4:7]) # 历史帧的 bbox_b 
        l_box = to_bbox4(x[t, 13:16]) # 历史帧的 bbox_a_last

        # 绘制预测框 (红色系列)
        rect_pred = patches.Rectangle(
            (p_box[0], p_box[2]), p_box[1] - p_box[0], p_box[3] - p_box[2],
            linewidth=lw_val, edgecolor='red', facecolor='none', 
            linestyle='--', alpha=alpha_val
        )
        
        # 绘制真值框 (绿色系列)
        rect_gt = patches.Rectangle(
            (t_box[0], t_box[2]), t_box[1] - t_box[0], t_box[3] - t_box[2],
            linewidth=lw_val, edgecolor='green', facecolor='none', 
            linestyle='-', alpha=alpha_val
        )
        rect_b = patches.Rectangle(
            (b_box[0], b_box[2]), b_box[1] - b_box[0], b_box[3] - b_box[2],
            linewidth=lw_val, edgecolor='blue', facecolor='none', 
            linestyle=':', alpha=alpha_val
        )
        rect_l = patches.Rectangle(
            (l_box[0], l_box[2]), l_box[1] - l_box[0], l_box[3] - l_box[2],
            linewidth=lw_val, edgecolor='cyan', facecolor='none', 
            linestyle=':', alpha=alpha_val
        )


        ax.add_patch(rect_pred)
        ax.add_patch(rect_gt)
        ax.add_patch(rect_b)
        ax.add_patch(rect_l)

    # 2. 绘制中心点轨迹 (用于辅助观察连贯性)
    p_centers_u = (pred[:, 0] + pred[:, 1]) / 2
    p_centers_v = (pred[:, 2] + pred[:, 3]) / 2
    t_centers_u = (target[:, 0] + target[:, 1]) / 2
    t_centers_v = (target[:, 2] + target[:, 3]) / 2
    
    ax.plot(t_centers_u, t_centers_v, color='green', alpha=0.3, label='GT Path')
    ax.plot(p_centers_u, p_centers_v, color='red', alpha=0.3, linestyle='--', label='Pred Path')

    # 3. 绘制当前帧 (最后一帧) 的运动方向箭头
    ax.quiver(p_centers_u[-1], p_centers_v[-1], pred[-1, 4], pred[-1, 5], 
              color='red', scale=15, width=0.005, label='Current Motion (Pred)')
    ax.quiver(t_centers_u[-1], t_centers_v[-1], target[-1, 4], target[-1, 5], 
              color='green', scale=15, width=0.005, label='Current Motion (GT)')
    

    # 4. 图例处理 (避免重复添加图例)
    # 手动创建图例句柄
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='green', lw=2, label='Target (GT)'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Estimate (Pred)'),
        Line2D([0], [0], color='blue', lw=2, linestyle=':', label='History bbox_b'),
        Line2D([0], [0], color='cyan', lw=2, linestyle=':', label='History bbox_a_last'),
        Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='History (Faded)')
    ]
    ax.legend(handles=custom_lines, loc='upper right')

    # 5. 坐标系美化
    ax.set_title(f"Sequence Detection History - Sample {idx}")
    ax.set_xlabel("U (pixels)")
    ax.set_ylabel("V (pixels)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.invert_yaxis() # 匹配图像坐标系
    ax.grid(True, linestyle=':', alpha=0.5)

    # 设置坐标轴范围 (防止框超出视野导致自动缩放太小)
    # 根据数据动态调整，或者固定为图像分辨率，如 ax.set_xlim(0, 1920)
    
    plt.savefig(os.path.join(save_dir, f"result_seq_{idx}.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--train_meta', type=str, required=True) # 需要用来初始化 dataset 结构
    # parser.add_argument('--test_meta', type=str, required=True)
    parser.add_argument('--model', type=str, default='attention', choices=['transformer', 'lstm', 'attention'])
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    
    args = parser.parse_args()
    test(args)