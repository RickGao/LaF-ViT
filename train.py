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
from model import LaFViT
from tqdm import tqdm
import logging
import sys
from datetime import datetime

# ==========================================
# 1. 命令行参数配置
# ==========================================
parser = argparse.ArgumentParser(description='LaFViT Training (Weighted + Norm + DiffLR)')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='数据集文件夹路径')
parser.add_argument('--epochs', type=int, default=30, help='训练总轮数')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-4, help='Stage 1 的初始学习率')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
args = parser.parse_args()


# ==========================================
# 2. 辅助函数
# ==========================================
def setup_logger(log_dir):
    """配置 Logger，文件名带时间戳"""
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_log_{current_time}.txt'
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("LaFViT")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_path


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
                # 🔥【改动点A】: 验证时需要还原年龄
                # 模型输出是 0.3 -> 还原成 30 岁
                pred_age_real = age_pred * 100.0
                total_mae += torch.sum(torch.abs(pred_age_real - ages)).item()
            else:
                total_mae = 99.9
    return (total_mae / count), (correct_gen / count), (correct_race / count)


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- Step A: 设置环境 ---
    log_dir = './log'
    ckpt_dir = args.save_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, log_path = setup_logger(log_dir)
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    stage1_epochs = int(args.epochs * 0.2)
    stage2_epochs = args.epochs - stage1_epochs

    logger.info("=" * 40)
    logger.info(f"🚀 Training LaFViT | Device: {device} | Seed: {args.seed}")
    logger.info(f"📂 Log saved to: {log_path}")
    logger.info(f"⚙️ Config: Epochs={args.epochs} (S1={stage1_epochs}, S2={stage2_epochs})")
    logger.info(f"✨ Enhancements: RaceWeights(1,1,1,2,3), AgeNorm(/100), DiffLR(x4)")
    logger.info("=" * 40)

    # --- Step B: 数据集加载 ---
    gen = torch.Generator().manual_seed(args.seed)
    temp_ds = UTKFaceDataset(args.data_dir, transform=None)
    train_len = int(0.9 * len(temp_ds))
    val_len = len(temp_ds) - train_len

    train_ds_full = UTKFaceDataset(args.data_dir, transform=train_transforms)
    val_ds_full = UTKFaceDataset(args.data_dir, transform=val_transforms)

    train_subset, _ = random_split(train_ds_full, [train_len, val_len], generator=gen)
    _, val_subset = random_split(val_ds_full, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    logger.info(f"📊 Dataset Split: Train={len(train_subset)}, Val={len(val_subset)}")

    # --- Step C: 模型初始化 ---
    logger.info("🧠 Initializing LaFViT (Small + Base)...")
    model = LaFViT(pretrained=True).to(device)

    # ==========================================
    # 🔥【改动点B】: Loss 配置
    # ==========================================
    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()

    # Race Class Weights: 0:White, 1:Black, 2:Asian, 3:Indian, 4:Others
    # 策略: White/Black/Asian=1.0, Indian=2.0, Others=3.0
    race_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0]).to(device)
    criterion_race = nn.CrossEntropyLoss(weight=race_weights)

    # 初始优化器 (Stage 1)
    optimizer = optim.AdamW([
        {'params': model.demo_backbone.parameters()},
        {'params': model.gender_head.parameters()},
        {'params': model.race_head.parameters()}
    ], lr=args.lr)

    scheduler = None
    best_val_mae = float('inf')

    # --- Step D: 训练循环 ---
    logger.info("🔥 Start Training Loop...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # --- 阶段切换逻辑 ---
        if epoch < stage1_epochs:
            stage = "stage1"
            # 冻结 Base, 激活 Small
            for p in model.age_backbone.parameters(): p.requires_grad = False
            for p in model.age_head.parameters(): p.requires_grad = False
            for p in model.demo_backbone.parameters(): p.requires_grad = True

        elif epoch == stage1_epochs:
            logger.info("🧊 Switch to Stage 2: Freezing Small, Training Base...")
            stage = "stage2"

            # 强制 Small 进入 eval 模式，防止 BN 统计漂移
            model.demo_backbone.eval()
            model.gender_head.eval()
            model.race_head.eval()

            # 冻结 Small
            for p in model.demo_backbone.parameters(): p.requires_grad = False
            for p in model.gender_head.parameters(): p.requires_grad = False
            for p in model.race_head.parameters(): p.requires_grad = False

            # 解冻 Base
            for p in model.age_backbone.parameters(): p.requires_grad = True
            for p in model.age_head.parameters(): p.requires_grad = True

            optimizer = optim.AdamW([
                # Backbone: 1e-5
                {'params': model.age_backbone.parameters(), 'lr': 1e-5},
                # Head: 4e-5
                {'params': model.age_head.parameters(), 'lr': 4e-5}
            ], weight_decay=0.05)  # <--- 从 1e-2 改成 0.05，增强约束

            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)
        else:
            stage = "stage2"
            # 保持 Small 为 eval 模式
            model.demo_backbone.eval()
            model.gender_head.eval()
            model.race_head.eval()

        # --- Tqdm 循环 ---
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs} [{stage}]")

        for batch in loop:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            # ==========================================
            # 🔥【改动点D】: 年龄归一化 (Age Normalization)
            # ==========================================
            if stage == "stage2":
                ages_target = ages / 100.0  # [0, 100] -> [0.0, 1.0]
            else:
                ages_target = ages  # stage1 不用 age，无所谓

            optimizer.zero_grad()
            age_pred, g_logits, r_logits = model(imgs, stage=stage)

            if stage == "stage1":
                # 分类任务: 包含了加权的 Race Loss
                loss = criterion_gender(g_logits, genders) + criterion_race(r_logits, races)
                d_val = 0.0
            else:
                # 回归任务: 拟合归一化后的年龄
                loss = criterion_age(age_pred, ages_target)
                d_val = loss.item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 进度条显示
            with torch.no_grad():
                acc_g = (torch.argmax(g_logits, 1) == genders).float().mean()
                acc_r = (torch.argmax(r_logits, 1) == races).float().mean()
            loop.set_postfix(loss=loss.item(), mse=d_val, g=f"{acc_g:.2f}", r=f"{acc_r:.2f}")

        if scheduler: scheduler.step()

        # --- 验证与日志 ---
        val_mae, val_gen, val_race = validate(model, val_loader, device, stage)
        avg_train_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1:02d} Report | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | Gen Acc: {val_gen:.2%} | Race Acc: {val_race:.2%}")

        # --- 保存 ---
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'laf_vit_latest.pth'))

        if stage == "stage2" and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'laf_vit_best.pth'))
            logger.info(f"  🌟 New Best Model Saved! (MAE: {best_val_mae:.4f})")

        if (epoch + 1) % 2 == 0:
            ckpt_name = f'laf_vit_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))

    logger.info("🎉 Training Complete.")


if __name__ == "__main__":
    main()