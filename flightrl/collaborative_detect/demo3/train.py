import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os

from data_loader import CollaborativeTrackingDataset
from model import SpatioTemporalFusionNet, CollaborativeTrackingLoss, ConcatMLPBaseline, DirectRegCrossAttnBaseline, SpatialCrossAttnBaseline, TCNBaseline, LSTMBaseline,SelfAttnBaseline, BaselineLoss

MODEL_TYPE = 3  # 0: SpatioTemporalFusionNet, 1: SelfAttnBaseline, 2: DirectRegCrossAttnBaseline, 3: ConcatMLPBaseline

def train_collaborative_model(train_files, val_files, epochs=50, batch_size=64, seq_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    os.makedirs(f"checkpoints/{timestamp}_{MODEL_TYPE}/", exist_ok=True)
    log_dir = os.path.join('./logs', f"{timestamp}_{MODEL_TYPE}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 1. 实例化 Dataset 和 DataLoader
    train_dataset = CollaborativeTrackingDataset(train_files, seq_len=seq_len)
    val_dataset = CollaborativeTrackingDataset(val_files, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 实例化网络与损失函数
    if (MODEL_TYPE == 0):
        model = SpatioTemporalFusionNet(seq_len=seq_len).to(device)
    elif (MODEL_TYPE == 1):
        model = SelfAttnBaseline().to(device)
    elif (MODEL_TYPE == 2):
        model = DirectRegCrossAttnBaseline().to(device)
    elif (MODEL_TYPE == 3):
        model = ConcatMLPBaseline(seq_len=seq_len).to(device)
    # model = SpatioTemporalFusionNet(seq_len=seq_len).to(device)
    # model = SelfAttnBaseline().to(device)
    # model = DirectRegCrossAttnBaseline().to(device)
    # model = ConcatMLPBaseline().to(device)
    if (MODEL_TYPE == 0):
        criterion = CollaborativeTrackingLoss()
    else:
        criterion = BaselineLoss()
    # criterion = CollaborativeTrackingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学习率调度器 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. 训练主循环
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(epochs):
        model.train()
        train_loss, train_ego_loss, train_gate_loss = 0.0, 0.0, 0.0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_data in pbar:
            e_obs, p_obs, e_pose, p_pose, gt, p_mask = [x.to(device) for x in batch_data]
            
            optimizer.zero_grad()
            
            # 前向传播 (解包三个返回值)
            pred_final, pred_ego_only, alpha = model(e_obs, p_obs, e_pose, p_pose)
            
            # 计算复合 Loss
            loss, loss_items = criterion(pred_final, pred_ego_only, alpha, gt, p_mask)
            l_ego = loss_items["loss_ego"]
            l_vel = loss_items["loss_vel"]
            l_bbox = loss_items["loss_bbox"]
            l_gate = loss_items["loss_gate"]
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪 (防止归一化坐标以外的极端值爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            writer.add_scalar('Loss/Total', loss_items["loss_total"], global_step)
            writer.add_scalar('Loss/bbox', loss_items["loss_bbox"], global_step)
            writer.add_scalar('Loss/vel', loss_items["loss_vel"], global_step)
            writer.add_scalar('Loss/ego', loss_items["loss_ego"], global_step)
            writer.add_scalar('Loss/gate', loss_items["loss_gate"], global_step)

            # writer.add_scalars('Train_Step_Comparison', {
            #     'bbox_Component': loss_items["loss_bbox"],
            #     'vel_Component': loss_items["loss_vel"],
            #     'ego_Component': loss_items["loss_ego"],
            #     'gate_Component': loss_items["loss_gate"]
            # }, global_step)
            
            # 记录指标
            train_loss += loss.item()
            train_ego_loss += l_ego
            train_gate_loss += l_gate
            global_step += 1
            # 动态更新进度条显示
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Gate': f"{l_gate:.4f}"})

        scheduler.step()
        
        # 4. 验证循环
        model.eval()
        val_metrics = {"total": 0, "bbox": 0, "vel": 0, "ego": 0, "gate": 0}
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                e_obs, p_obs, e_pose, p_pose, gt, p_mask = [x.to(device) for x in batch_data]
                pred_final, pred_ego_only, alpha = model(e_obs, p_obs, e_pose, p_pose)
                loss, v_items = criterion(pred_final, pred_ego_only, alpha, gt, p_mask)
                val_loss += loss.item()
                val_metrics["total"] += v_items["loss_total"]
                val_metrics["bbox"] += v_items["loss_bbox"]
                val_metrics["vel"] += v_items["loss_vel"]
                val_metrics["ego"] += v_items["loss_ego"]
                val_metrics["gate"] += v_items["loss_gate"]
                
        # 打印 Epoch 摘要
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        num_val_batches = len(val_loader)
        writer.add_scalar('Val_Epoch_Loss/Total', val_metrics["total"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/bbox', val_metrics["bbox"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/vel', val_metrics["vel"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/ego', val_metrics["ego"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/gate', val_metrics["gate"] / num_val_batches, epoch)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_file = os.path.join(f"./checkpoints/{timestamp}_{MODEL_TYPE}/", f"best_model.pth")
            torch.save(model.state_dict(), save_file)
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': best_val_loss,
            # }, save_file)
            print(f"Saved Best Model to {save_file}")


        # (可选) 模型保存机制
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/{timestamp}_{MODEL_TYPE}/net_ep{epoch+1}.pth")

if __name__ == "__main__":
    # 解析 C++ 输出的文件列表
    with open("/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/train_files.txt", "r") as f:
        train_list = [line.strip() for line in f.readlines()]
    
    with open("/home/lac/fm_test/src/flightmare/flightrl/collaborative_detect/dataset3/val_files.txt", "r") as f:
        val_list = [line.strip() for line in f.readlines()]
        
    # 启动训练
    os.makedirs("checkpoints", exist_ok=True)
    train_collaborative_model(train_list, val_list, epochs=50, batch_size=128, seq_len=5)