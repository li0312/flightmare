'''
Author: Lac_Creeper
Date: 2026-04-06 02:25:54 +0800
LastEditTime: 2026-04-06 02:27:30 +0800
LastEditors: Lac_Creeper
Description: 
FilePath: /flightrl/collaborative_detect/demo3/vis.py
'''
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_trajectory_from_json(json_file_path):
    """读取分块 JSON 并绘制二维飞行轨迹"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        print(f"File {json_file_path} is empty.")
        return

    # 提取绝对轨迹坐标
    target_x = [step["target_pose"][0] for step in data]
    target_y = [step["target_pose"][1] for step in data]

    uav_a_x = [step["uav_a_pose"][0] for step in data]
    uav_a_y = [step["uav_a_pose"][1] for step in data]

    uav_b_x = [step["uav_b_pose"][0] for step in data]
    uav_b_y = [step["uav_b_pose"][1] for step in data]

    # 初始化绘图
    plt.figure(figsize=(10, 8))
    plt.title(f"Collaborative Tracking Trajectory\nFile: {os.path.basename(json_file_path)}", fontsize=14)

    # 绘制连续轨迹
    plt.plot(target_x, target_y, 'r--', label='Target Trajectory', linewidth=2)
    plt.plot(uav_a_x, uav_a_y, 'b-', label='UAV A (Ego) Trajectory', linewidth=1.5)
    plt.plot(uav_b_x, uav_b_y, 'g-', label='UAV B (Peer) Trajectory', linewidth=1.5)

    # 标记起始点 (Start) 和 结束点 (End)
    plt.scatter(target_x[0], target_y[0], c='red', marker='o', s=100, edgecolors='black', label='Target Start')
    plt.scatter(target_x[-1], target_y[-1], c='red', marker='x', s=100)

    plt.scatter(uav_a_x[0], uav_a_y[0], c='blue', marker='o', s=100, edgecolors='black', label='UAV A Start')
    plt.scatter(uav_a_x[-1], uav_a_y[-1], c='blue', marker='x', s=100)

    plt.scatter(uav_b_x[0], uav_b_y[0], c='green', marker='o', s=100, edgecolors='black', label='UAV B Start')
    plt.scatter(uav_b_x[-1], uav_b_y[-1], c='green', marker='x', s=100)

    # 如果有遮挡标签（可选画图）
    # 例如：标记僚机在哪些点是失效的（在 Python 预处理加噪时用到，原数据现在全是完美的）

    plt.xlabel('Global X Position (m)', fontsize=12)
    plt.ylabel('Global Y Position (m)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal') # 保证 X 和 Y 轴比例一致，这样轨迹不会变形

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 指向你生成的 samples 文件夹
    dataset_dir = "/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/samples"
    
    # 查找所有的 json 文件
    json_files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    
    if len(json_files) == 0:
        print("未找到 JSON 文件，请检查路径。")
    else:
        print(f"找到 {len(json_files)} 个 JSON 文件。")
        # 可视化第一回合的第一个块 (chunk_0000)
        # 您可以修改索引查看特定的序列
        test_file = json_files[0] 
        print(f"Visualizing: {test_file}")
        plot_trajectory_from_json(test_file)