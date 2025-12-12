import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import UTKFaceDataset, train_transforms, val_transforms
from model import LaFViT  # <--- 类名已改
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LaFViT Training')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42, help='必须固定Seed以保证Eval能找到对应的Val集')
parser.add_argument('--save_dir', type=str, default='./checkpoints')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def validate(model, loader, device, stage):
    model.eval()
    total_mae, correct_gen, correct_race, count = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            age_pred, g_logits, r_logits = model(imgs, stage=stage)

            count += len(imgs)
            correct_gen += (torch.argmax(g_logits, 1) == genders).sum().item()
            correct_race += (torch.argmax(r_logits, 1) == races).sum().item()

            if stage == "stage2":
                total_mae += torch.sum(torch.abs(age_pred - ages)).item()
            else:
                total_mae = 99.9
    return (total_mae / count), (correct_gen / count), (correct_race / count)


def main():
    set_seed(args.seed)  # 1. 固定随机种子
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    stage1_epochs = int(args.epochs * 0.2)
    stage2_epochs = args.epochs - stage1_epochs

    print(f"🚀 Training LaFViT | Seed: {args.seed} | Device: {device}")

    # 2. 数据划分 (90% Train / 10% Val)
    # 使用 Generator 确保每次划分一致
    gen = torch.Generator().manual_seed(args.seed)

    # 先加载一次获取总长度
    temp_ds = UTKFaceDataset(args.data_dir, transform=None)
    train_len = int(0.9 * len(temp_ds))
    val_len = len(temp_ds) - train_len

    # 分别加载带不同 transform 的数据集
    train_ds_full = UTKFaceDataset(args.data_dir, transform=train_transforms)
    val_ds_full = UTKFaceDataset(args.data_dir, transform=val_transforms)

    # 切分
    train_subset, _ = random_split(train_ds_full, [train_len, val_len], generator=gen)
    _, val_subset = random_split(val_ds_full, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Split: Train={len(train_subset)} (90%), Val={len(val_subset)} (10%)")

    # 3. 初始化模型
    model = LaFViT(pretrained=True).to(device)

    criterion_age = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    # 初始优化器 (Stage 1)
    optimizer = optim.AdamW([
        {'params': model.demo_backbone.parameters()},
        {'params': model.gender_head.parameters()},
        {'params': model.race_head.parameters()}
    ], lr=args.lr)

    scheduler = None
    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()

        # --- 阶段切换 ---
        if epoch < stage1_epochs:
            stage = "stage1"
            # 冻结 Base
            for p in model.age_backbone.parameters(): p.requires_grad = False
            for p in model.age_head.parameters(): p.requires_grad = False
            # 解冻 Small
            for p in model.demo_backbone.parameters(): p.requires_grad = True

        elif epoch == stage1_epochs:
            print("\n🧊 Switch to Stage 2: Freezing Small, Training Base...")
            stage = "stage2"

            # 冻结 Small
            for p in model.demo_backbone.parameters(): p.requires_grad = False
            for p in model.gender_head.parameters(): p.requires_grad = False
            for p in model.race_head.parameters(): p.requires_grad = False

            # 解冻 Base
            for p in model.age_backbone.parameters(): p.requires_grad = True
            for p in model.age_head.parameters(): p.requires_grad = True

            optimizer = optim.AdamW([
                {'params': model.age_backbone.parameters()},
                {'params': model.age_head.parameters()}
            ], lr=args.lr)

            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

        else:
            stage = "stage2"

        # --- 训练循环 ---
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs} [{stage}]")
        for batch in loop:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            optimizer.zero_grad()
            age_pred, g_logits, r_logits = model(imgs, stage=stage)

            if stage == "stage1":
                loss = criterion_cls(g_logits, genders) + criterion_cls(r_logits, races)
                d_val = 0.0
            else:
                loss = criterion_age(age_pred, ages)
                d_val = loss.item()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc_g = (torch.argmax(g_logits, 1) == genders).float().mean()
                acc_r = (torch.argmax(r_logits, 1) == races).float().mean()
            loop.set_postfix(loss=loss.item(), mse=d_val, g=f"{acc_g:.2f}", r=f"{acc_r:.2f}")

        if scheduler: scheduler.step()

        # --- 验证 ---
        val_mae, val_gen, val_race = validate(model, val_loader, device, stage)
        print(f"  Val: AgeMAE={val_mae:.4f} | Gen={val_gen:.1%} | Race={val_race:.1%}")

        # --- 保存逻辑 (Best + Checkpoint) ---
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_latest.pth'))

        # 1. 保存 Best (Stage 2)
        if stage == "stage2" and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_best.pth'))
            print("  🌟 New Best Saved!")

        # 2. 每两个 Epoch 保存一次 Checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt_name = f'laf_vit_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), os.path.join(args.save_dir, ckpt_name))
            # print(f"  💾 Checkpoint saved: {ckpt_name}")


if __name__ == "__main__":
    main()