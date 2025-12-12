import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import UTKFaceDataset, train_transforms
from model import LaFViT
from tqdm import tqdm  # 引入 tqdm

# 命令行参数配置
parser = argparse.ArgumentParser(description='Train LaF-ViT')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='数据集路径')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
args = parser.parse_args()


def main():
    # 0. 打印训练配置
    print("=" * 40)
    print(f"🚀 Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  - {arg:<15}: {value}")
    print("=" * 40)

    # 1. 准备环境
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"✅ Running on device: {device}")

    # 2. 加载数据
    print("📂 Loading dataset...")
    full_dataset = UTKFaceDataset(args.data_dir, transform=train_transforms)

    # 简单切分 90% 训练, 10% 验证
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"📊 Training images: {len(train_ds)} | Batches: {len(train_loader)}")

    # 3. 初始化模型
    model = LaFViT(pretrained=True).to(device)

    # 4. 优化器和 Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_race = nn.CrossEntropyLoss()

    print("🔥 Start Training...")

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # 使用 tqdm 包装 train_loader
        # desc: 进度条左边的描述文字
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]")

        for batch in loop:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            optimizer.zero_grad()

            # Forward
            age_pred, gender_logits, race_logits = model(imgs)

            # Calculate Losses
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_logits, genders)
            loss_race = criterion_race(race_logits, races)

            # Multi-task Loss
            loss = loss_age + 0.5 * loss_gender + 0.5 * loss_race

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 实时更新进度条右侧的显示 (只显示最重要的指标)
            loop.set_postfix(loss=loss.item(), age_mse=loss_age.item())

        # 每个 Epoch 结束后的总结
        avg_loss = total_loss / len(train_loader)
        # 这里的 print 会保留在屏幕上，作为历史记录
        print(f"Epoch {epoch + 1} Done. Average Loss: {avg_loss:.4f}")

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_latest.pth'))

        # (可选) 每2个 epoch 多存一个备份，防止覆盖
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'laf_vit_epoch_{epoch + 1}.pth'))

    print("🎉 Training Complete.")


if __name__ == "__main__":
    main()