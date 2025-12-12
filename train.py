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

parser = argparse.ArgumentParser(description='LaFViT Training')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42, help='必须固定Seed以保证Eval能找到对应的Val集')
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
args = parser.parse_args()


# ==========================================
# 2. 辅助函数
# ==========================================
def setup_logger(log_dir):
    """配置 Logger，文件名带时间戳"""
    # 确保 log 文件夹存在
    os.makedirs(log_dir, exist_ok=True)

    # 获取当前时间，格式: YYYYMMDD_HHMMSS (例如: 20231212_120000)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_log_{current_time}.txt'

    log_path = os.path.join(log_dir, log_filename)

    # 创建 logger
    logger = logging.getLogger("LaFViT")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Handler 1: 文件 (File) -> 存入 log 文件夹
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler 2: 屏幕 (Stream/Console)
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
                total_mae += torch.sum(torch.abs(age_pred - ages)).item()
            else:
                total_mae = 99.9
    return (total_mae / count), (correct_gen / count), (correct_race / count)


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- Step A: 设置环境 ---

    # 1. 设置路径
    log_dir = './log'  # 日志文件夹
    ckpt_dir = args.save_dir  # 权重文件夹

    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. 初始化 Logger (文件名带时间了)
    logger, log_path = setup_logger(log_dir)

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    stage1_epochs = int(args.epochs * 0.2)
    stage2_epochs = args.epochs - stage1_epochs

    logger.info("=" * 40)
    logger.info(f"🚀 Training LaFViT | Device: {device} | Seed: {args.seed}")
    logger.info(f"📂 Log saved to: {log_path}")  # 这里会打印出具体带时间的文件名
    logger.info(f"💾 Checkpoints saved to: {ckpt_dir}")
    logger.info(
        f"⚙️ Config: Epochs={args.epochs} (S1={stage1_epochs}, S2={stage2_epochs}), Batch={args.batch_size}, LR={args.lr}")
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

    # --- Step D: 训练循环 ---
    logger.info("🔥 Start Training Loop...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # --- 阶段切换逻辑 ---
        if epoch < stage1_epochs:
            stage = "stage1"
            # 确保 Base 冻结
            for p in model.age_backbone.parameters(): p.requires_grad = False
            for p in model.age_head.parameters(): p.requires_grad = False
            # 确保 Small 激活
            for p in model.demo_backbone.parameters(): p.requires_grad = True

        elif epoch == stage1_epochs:
            logger.info("🧊 Switch to Stage 2: Freezing Small, Training Base...")
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

        # --- Tqdm 循环 ---
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
            total_loss += loss.item()

            # 进度条显示 (只在屏幕显示)
            with torch.no_grad():
                acc_g = (torch.argmax(g_logits, 1) == genders).float().mean()
                acc_r = (torch.argmax(r_logits, 1) == races).float().mean()
            loop.set_postfix(loss=loss.item(), mse=d_val, g=f"{acc_g:.2f}", r=f"{acc_r:.2f}")

        if scheduler: scheduler.step()

        # --- 验证与日志记录 ---
        val_mae, val_gen, val_race = validate(model, val_loader, device, stage)
        avg_train_loss = total_loss / len(train_loader)

        # 📝 核心日志：写入文件和屏幕
        logger.info(
            f"Epoch {epoch + 1:02d} Report | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | Gen Acc: {val_gen:.2%} | Race Acc: {val_race:.2%}")

        # --- 保存逻辑 ---
        # 始终保存最新的
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'laf_vit_latest.pth'))

        # 只在 Stage 2 保存最好的
        if stage == "stage2" and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'laf_vit_best.pth'))
            logger.info(f"  🌟 New Best Model Saved! (MAE: {best_val_mae:.4f})")

        # 每2轮保存一个断点
        if (epoch + 1) % 2 == 0:
            ckpt_name = f'laf_vit_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))

    logger.info("🎉 Training Complete.")


if __name__ == "__main__":
    main()