import torch
import torch.optim as optim
import os
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入你提供的 data_loader 模块
from data_loader import create_data_loaders
# 导入模型定义
from models1 import LSTMNet, DirectionNormLoss, TemporalCNNNet, TransformerNet, CrossAttentionNet, CollaborativeTrackingLoss

CIOU = 1.0
DIR = 1.0
BBOX = 0.5
DOT = 0.5


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('./logs', f"{timestamp}_{args.model}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 1. 加载数据
    # 根据 data_loader.py 的接口，它返回 4 个对象
    train_loader, val_loader, train_dataset, _ = create_data_loaders(
        data_dir='../dataset',
        train_meta='../dataset/train_files.txt',
        val_meta='../dataset/val_files.txt',
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        num_workers=4
    )

    # 2. 初始化模型、损失函数和优化器
    # model = LSTMNet(
    #     input_dim=16, 
    #     hidden_dim=args.hidden_dim, 
    #     num_layers=args.num_layers
    # ).to(device)
    # model = TemporalCNNNet().to(device)
    # model = CrossAttentionNet().to(device)
    # model = TransformerNet().to(device)
    if args.model == 'lstm':
        model = LSTMNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'attention':
        model = CrossAttentionNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'transformer':
        model = TransformerNet(input_dim=16, hidden_dim=args.hidden_dim).to(device)


    
    criterion = CollaborativeTrackingLoss(ciou_weight=CIOU, dir_weight=DIR, bbox_weight=BBOX, dot_weight=DOT)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率衰减：当验证集 Loss 不再下降时减小 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=3e-6)

    # 3. 训练循环
    best_val_loss = float('inf')
    global_step = 0
    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_x, batch_y in train_pbar:
            # 转换为 float32 并移动到设备
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss, loss_items = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪：防止循环网络的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar('Loss/Total', loss_items["total_loss"], global_step)
            writer.add_scalar('Loss/CIoU', loss_items["loss_ciou"], global_step)
            writer.add_scalar('Loss/Direction', loss_items["loss_dir"], global_step)
            writer.add_scalar('Loss/BBox', loss_items["loss_bbox"], global_step)

            # writer.add_scalars('Train_Step_Comparison', {
            #     'CIoU_Component': loss_items["loss_ciou"],
            #     'Dir_Component': loss_items["loss_dir"],
            #     'BBox_Component': loss_items["loss_bbox"]
            # }, global_step)
            
            total_train_loss += loss.item()
            global_step += 1
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 4. 验证循环
        model.eval()
        val_metrics = {"total": 0, "ciou": 0, "dir": 0, "bbox": 0}
        total_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device).float()
                val_y = val_y.to(device).float()
                
                val_outputs = model(val_x)
                v_loss, v_items = criterion(val_outputs, val_y)
                total_val_loss += v_loss.item()
                val_metrics["total"] += v_items["total_loss"]
                val_metrics["ciou"] += v_items["loss_ciou"]
                val_metrics["dir"] += v_items["loss_dir"]
                val_metrics["bbox"] += v_items["loss_bbox"]

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Weight/CIoU', CIOU, epoch)
        writer.add_scalar('Weight/Dir', DIR, epoch)
        writer.add_scalar('Weight/BBox', BBOX, epoch)
        writer.add_scalar('Weight/Dot', DOT, epoch)
        num_val_batches = len(val_loader)
        writer.add_scalar('Val_Epoch_Loss/Total', val_metrics["total"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/CIoU', val_metrics["ciou"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/Direction', val_metrics["dir"] / num_val_batches, epoch)
        writer.add_scalar('Val_Epoch_Loss/BBox', val_metrics["bbox"] / num_val_batches, epoch)
        
        scheduler.step(avg_val_loss)
        
        print(f"Summary - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_file = os.path.join(args.save_path, f"best_{args.model}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'input_mean': train_dataset.input_mean, # 保存归一化参数以便推理
                'input_std': train_dataset.input_std
            }, save_file)
            print(f"Saved Best Model to {save_file}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Collaborative Tracking Training (LSTM)")
    # parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    # parser.add_argument('--train_meta', type=str, required=True, help='训练集元数据文件(txt)')
    # parser.add_argument('--val_meta', type=str, required=True, help='验证集元数据文件(txt)')
    parser.add_argument('--model', type=str, default='attention', choices=['transformer', 'lstm', 'attention'])
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='模型保存路径')
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20, help='时序长度')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()
    train_model(args)