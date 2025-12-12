import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, train_transforms, val_transforms  # 确保 dataset.py 里有 val_transforms
from model import LaFViT
from tqdm import tqdm

# 配置参数
parser = argparse.ArgumentParser(description='Train LaF-ViT')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--save_dir', type=str, default='./checkpoints')
args = parser.parse_args()


def set_seed(seed):
    """固定所有随机种子，保证 Split 和 初始化一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def validate(model, loader, device, criterion_age):
    """验证函数：计算 MAE 和 Accuracy"""
    model.eval()
    total_age_mae = 0
    correct_gender = 0
    correct_race = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            age_pred, gender_logits, race_logits = model(imgs)

            # 1. Age MAE (即使 Loss 是 MSE，验证指标通常也看 MAE)
            total_age_mae += torch.sum(torch.abs(age_pred - ages)).item()

            # 2. Gender Acc
            gender_preds = torch.argmax(gender_logits, dim=1)
            correct_gender += (gender_preds == genders).sum().item()

            # 3. Race Acc
            race_preds = torch.argmax(race_logits, dim=1)
            correct_race += (race_preds == races).sum().item()

            total_samples += len(imgs)

    return (total_age_mae / total_samples), (correct_gender / total_samples), (correct_race / total_samples)


def main():
    # 1. 设置随机种子
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Device: {device} | Seed: {args.seed}")

    # 2. 加载数据 & 划分
    # 注意：这里我们让 train 和 val 使用不同的 transform
    full_dataset = UTKFaceDataset(args.data_dir, transform=None)  # 先不加 transform

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 使用 generator 保证 split 结果固定
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 动态绑定 Transform (这也是一个小技巧，避免重复加载 dataset)
    # 我们需要构建一个新的 Dataset Wrapper 或者简单的在 Dataset 类里处理，
    # 这里为了简便，假设 dataset.py 里允许我们在外部覆盖 transform，
    # 或者我们直接实例化两次 Dataset (最稳妥做法)

    print("📂 Reloading datasets with specific transforms...")
    train_ds = UTKFaceDataset(args.data_dir, transform=train_transforms)
    val_ds = UTKFaceDataset(args.data_dir, transform=val_transforms)

    # 再次 Split (必须用同样的 seed)
    train_subset, _ = random_split(train_ds, [train_size, val_size], generator=generator)
    _, val_subset = random_split(val_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)  # 验证集不要 shuffle

    print(f"📊 Train: {len(train_subset)} | Val: {len(val_subset)}")

    # 3. 模型
    model = LaFViT(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 4. Loss (加入 Class Weights 解决 Race 只有 0% 的问题)
    # 0:White, 1:Black, 2:Asian, 3:Indian, 4:Others
    race_weights = torch.tensor([1.0, 2.5, 2.5, 3.0, 5.0]).to(device)

    criterion_age = nn.L1Loss()  # 使用 L1Loss (MAE Loss)
    criterion_gender = nn.CrossEntropyLoss()
    criterion_race = nn.CrossEntropyLoss(weight=race_weights)

    best_val_mae = float('inf')

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # --- 课程学习策略 ---
        if epoch < 5:
            phase = "Warm-up"
            w_age, w_gender, w_race = 0.0, 2.0, 5.0  # 只练分类
        else:
            phase = "Joint"
            w_age, w_gender, w_race = 1.0, 1.0, 2.0  # 全面训练

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [{phase}]")

        for batch in loop:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            optimizer.zero_grad()

            age_pred, gender_logits, race_logits = model(imgs)

            l_age = criterion_age(age_pred, ages)
            l_gender = criterion_gender(gender_logits, genders)
            l_race = criterion_race(race_logits, races)

            loss = (w_age * l_age) + (w_gender * l_gender) + (w_race * l_race)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=loss.item(), age_err=l_age.item())

        # --- 每个 Epoch 结束后进行 Validation ---
        val_mae, val_gender_acc, val_race_acc = validate(model, val_loader, device, criterion_age)

        print(f"Epoch {epoch + 1} Report:")
        print(f"  Train Loss : {total_loss / len(train_loader):.4f}")
        print(f"  Val Age MAE: {val_mae:.4f} (Target: <4.0)")
        print(f"  Val Gender : {val_gender_acc * 100:.2f}%")
        print(f"  Val Race   : {val_race_acc * 100:.2f}%")

        # 保存最新的
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_latest.pth'))

        # 保存验证集效果最好的 (Best Model)
        if val_mae < best_val_mae and epoch >= 5:  # Warm-up 期间不存 best
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_best.pth'))
            print("  🌟 New Best Model Saved!")


if __name__ == "__main__":
    main()