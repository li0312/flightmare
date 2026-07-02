'''
Author: Lac_Creeper
Date: 2026-04-06 02:33:40 +0800
LastEditTime: 2026-04-27 21:52:33 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo3/vis_traj.py
'''
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import glob
import os
import numpy as np
import matplotlib
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# for i in a:
#     print(i)

def plot_academic_hierarchical_trajectory(dataset_dir, ep_id, zoom_chunk_id=10):
    # ================= 1. 环境配置 (论文级样式) =================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Sans"],
        # "font.serif": ["Times New Roman"],
        # "font.family": "sans-serif",
        # "font.serif": ["simhei"],
        "axes.labelsize": 16,
        "font.size": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.figsize": (14, 10),
        "savefig.dpi": 600,
        "lines.linewidth": 3.5,
    })

    # ================= 2. 数据拼接 =================
    chunk_files = sorted(glob.glob(os.path.join(dataset_dir, f"ep_{ep_id:04d}_chunk_*.json")))
    full_data = []
    chunk_indices = [] # 记录每个chunk的起始索引
    
    current_idx = 0
    for f_path in chunk_files:
    # for idx in range(20):
        # f_path = chunk_files[idx]
        with open(f_path, 'r') as f:
            chunk = json.load(f)
            full_data.extend(chunk)
            chunk_indices.append(current_idx)
            current_idx += len(chunk)

    # 提取全局轨迹
    t_x = np.array([s["target_pose"][0] for s in full_data])
    t_y = np.array([s["target_pose"][1] for s in full_data])
    a_x = np.array([s["uav_a_pose"][0] for s in full_data])
    a_y = np.array([s["uav_a_pose"][1] for s in full_data])
    b_x = np.array([s["uav_b_pose"][0] for s in full_data])
    b_y = np.array([s["uav_b_pose"][1] for s in full_data])

    # ================= 3. 主图绘制 (全回合轨迹) =================
    fig, ax = plt.subplots()
    
    # 绘制背景全局轨迹
    ax.plot(t_x, t_y, color='lightcoral', linestyle='--', alpha=0.9, label='_nolegend_')
    ax.plot(a_x, a_y, color='lightskyblue', linestyle='-', alpha=0.9, label='_nolegend_')
    ax.plot(b_x, b_y, color='lightgreen', linestyle='-', alpha=0.9, label='_nolegend_')

    # 标注每个分段的采样起始点 (Sampling Markers)
    # 每隔一个分段标一个点，避免太拥挤
    # ax.scatter(t_x[chunk_indices[::]], t_y[chunk_indices[::]], 
    #            color='#d62728', marker='o', s=10, alpha=0.8, label='Chunk Sampling Points')
    ax.scatter(t_x[chunk_indices[::3]], t_y[chunk_indices[::3]], 
               color='#d62728', s=50, zorder=3, label='Sampling Points')
    ax.scatter(a_x[chunk_indices[::3]], a_y[chunk_indices[::3]], 
               color='blue', s=50, zorder=3)
    ax.scatter(b_x[chunk_indices[::3]], b_y[chunk_indices[::3]], 
               color='green', s=50, zorder=3)
    ax.scatter(t_x[0], t_y[0], color='black', s=80, label='Start')
    ax.scatter(t_x[-1], t_y[-1], color='black', marker='*', s=150, label='End')
    ax.scatter(a_x[-1], a_y[-1], color='blue', marker='*', s=150)
    ax.scatter(b_x[-1], b_y[-1], color='green', marker='*', s=150)


    # # ================= 4. 局部放大 (Zoom-in Inset) =================
    # # 选择一个典型的分段进行放大展示
    # start_f = chunk_indices[zoom_chunk_id]
    # end_f = start_f + 40
    
    # # 创建嵌入坐标轴 (轴缩放比例为 2.5)
    # ax_ins = zoomed_inset_axes(ax, 0.1, loc='upper left', borderpad=3) 
    
    # # 在放大图中绘制具体的协同构型
    # # ax_ins.plot(t_x[start_f:end_f], t_y[start_f:end_f], color='crimson', linewidth=3, label='Target (Local)')
    # # ax_ins.plot(a_x[start_f:end_f], a_y[start_f:end_f], color='navy', linewidth=2.5, label='UAV A (Local)')
    # # ax_ins.plot(b_x[start_f:end_f], b_y[start_f:end_f], color='forestgreen', linewidth=2.5, label='UAV B (Local)')
    # ax_ins.plot(t_x[start_f:end_f], t_y[start_f:end_f], color='#d62728', lw=2)
    # ax_ins.plot(a_x[start_f:end_f], a_y[start_f:end_f], color='#1f77b4', lw=2)
    # ax_ins.plot(b_x[start_f:end_f], b_y[start_f:end_f], color='#2ca02c', lw=2)

    # # 绘制终点时刻的协同三角形
    # ax_ins.plot([a_x[end_f-1], t_x[end_f-1]], [a_y[end_f-1], t_y[end_f-1]], 'gray', linestyle='--', linewidth=1)
    # ax_ins.plot([b_x[end_f-1], t_x[end_f-1]], [b_y[end_f-1], t_y[end_f-1]], 'gray', linestyle='--', linewidth=1)
    # ax_ins.scatter(t_x[end_f-1], t_y[end_f-1], color='#d62728', marker='*', s=80)

    # # 局部图刻度和格式调整
    # ax_ins.set_xticks([])
    # ax_ins.set_yticks([])
    
    # # 建立主图到局部图的连接线
    # mark_inset(ax, ax_ins, loc1=1, loc2=4, fc="none", ec="gray", linestyle='--')

    # ================= 5. 主图整体优化 =================
    # 画出总体的起始/结束状态作为参考
    # ax.scatter(t_x[0], t_y[0], color='black', marker='o', s=50, label='Episode Start')
    # ax.scatter(t_x[-1], t_y[-1], color='black', marker='X', s=80, label='Episode End')

    ax.set_xlabel('X坐标/m', fontweight='bold')
    ax.set_ylabel('Y坐标/m', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7, linewidth=0.5)
    
    # 图例处理
    # 手动整理图例，避免重复
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='crimson', lw=2, linestyle='--'),
        Line2D([0], [0], color='navy', lw=2),
        Line2D([0], [0], color='forestgreen', lw=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12)
    ]
    ax.legend(custom_lines, ['目标轨迹', 'UAV A 轨迹', 'UAV B 轨迹', '采样起点', '回合终点'], 
              loc='lower right', frameon=True, shadow=True)

    plt.title(f"数据集分析 (回合 {ep_id})", pad=12)
    # plt.tight_layout()
    
    # 保存为 PDF 矢量图供论文使用
    plt.savefig(f"./ep_{ep_id}_view.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    path = "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/samples"
    # 可视化第0回合，并放大第15个分段（采样点）
    plot_academic_hierarchical_trajectory(path, ep_id=26, zoom_chunk_id=15)