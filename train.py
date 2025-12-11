import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import UTKFaceDataset, train_transforms
from model import LaFViT

# 命令行参数配置
parser = argparse.ArgumentParser(description='Train LaF-ViT')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='数据集路径')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size (根据显存调整)')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
args = parser.parse_args()


def main():
    # 0. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Running on device: {device}")

    # 1. 加载数据
    print("Loading dataset...")
    # 注意：这里我们简单地全部用 train_transforms，实际项目建议分 train/val set
    full_dataset = UTKFaceDataset(args.data_dir, transform=train_transforms)

    # 简单切分 90% 训练, 10% 验证
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Training images: {len(train_ds)}")

    # 2. 初始化模型
    model = LaFViT(pretrained=True).to(device)

    # 3. 优化器和 Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    criterion_age = nn.MSELoss()  # 年龄用均方误差
    criterion_gender = nn.CrossEntropyLoss()
    criterion_race = nn.CrossEntropyLoss()

    # 4. 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)  # [B, 1]
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            optimizer.zero_grad()

            # Forward
            age_pred, gender_logits, race_logits = model(imgs)

            # Calculate Losses
            loss_age = criterion_age(age_pred, ages)
            loss_gender = criterion_gender(gender_logits, genders)
            loss_race = criterion_race(race_logits, races)

            # Multi-task Loss (权重可调: 比如主要关注 Age, 性别种族作为辅助)
            loss = loss_age + 0.5 * loss_gender + 0.5 * loss_race

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f} (Age: {loss_age.item():.2f})")

        # 每个 Epoch 结束保存一次
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_latest.pth'))

        # (可选) 保存 best model 逻辑可以在这里加

    print("Training Complete.")


if __name__ == "__main__":
    main()