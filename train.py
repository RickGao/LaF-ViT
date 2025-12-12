import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, train_transforms, val_transforms
from model import LaFViT
from tqdm import tqdm

# ==========================================
# 1. 命令行参数配置
# ==========================================
parser = argparse.ArgumentParser(description='Train LaF-ViT (MSE + Uniform Weights)')
parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='数据集文件夹路径')
parser.add_argument('--epochs', type=int, default=20, help='训练总轮数')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
args = parser.parse_args()


# ==========================================
# 2. 辅助函数
# ==========================================
def set_seed(seed):
    """固定所有随机种子，保证 Split 和 初始化一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, loader, device):
    """验证函数：计算验证集上的 Age MAE 和 分类 Accuracy"""
    model.eval()
    total_age_ae = 0
    correct_gender = 0
    correct_race = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            # Forward pass
            age_pred, gender_logits, race_logits = model(imgs)

            # 1. Age MAE (即使训练用 MSE，验证时看 MAE 更直观)
            total_age_ae += torch.sum(torch.abs(age_pred - ages)).item()

            # 2. Gender Acc
            gender_preds = torch.argmax(gender_logits, dim=1)
            correct_gender += (gender_preds == genders).sum().item()

            # 3. Race Acc
            race_preds = torch.argmax(race_logits, dim=1)
            correct_race += (race_preds == races).sum().item()

            total_samples += len(imgs)

    avg_mae = total_age_ae / total_samples
    avg_gender_acc = correct_gender / total_samples
    avg_race_acc = correct_race / total_samples

    return avg_mae, avg_gender_acc, avg_race_acc


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- Step A: 设置环境 ---
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 40)
    print(f"🚀 Start Training | Device: {device} | Seed: {args.seed}")
    print("=" * 40)

    # --- Step B: 数据集加载与划分 ---
    # 定义 Generator 保证切分索引一致
    split_generator = torch.Generator().manual_seed(args.seed)

    # 临时加载以计算长度
    temp_dataset = UTKFaceDataset(args.data_dir, transform=None)
    train_size = int(0.9 * len(temp_dataset))
    val_size = len(temp_dataset) - train_size

    # 分别实例化并切分 (Train用增强，Val用标准)
    train_ds_full = UTKFaceDataset(args.data_dir, transform=train_transforms)
    val_ds_full = UTKFaceDataset(args.data_dir, transform=val_transforms)

    train_subset, _ = random_split(train_ds_full, [train_size, val_size], generator=split_generator)
    _, val_subset = random_split(val_ds_full, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Step C: 模型与优化器 ---
    print("🧠 Initializing LaF-ViT (Pretrained)...")
    model = LaFViT(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # --- Step D: 损失函数配置 (已按要求修改) ---

    # 1. Age: 使用 MSELoss
    criterion_age = nn.MSELoss()

    # 2. Gender: 标准交叉熵
    criterion_gender = nn.CrossEntropyLoss()

    # 3. Race: 标准交叉熵 (移除了 race_weights，所有种族一视同仁)
    criterion_race = nn.CrossEntropyLoss()

    best_val_mae = float('inf')

    # --- Step E: 训练循环 ---
    print("🔥 Start Training Loop (MSE + Uniform Weights)...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # === 课程学习策略 ===
        if epoch < 5:
            phase = "Warm-up"
            # 补课阶段：关掉 Age (w=0)，只练分类
            w_age, w_gender, w_race = 0.0, 1.0, 1.0
        else:
            phase = "Joint"
            # 联合阶段：
            # MSE 数值很大 (例如 50~100)，CrossEntropy 只有 ~1.0
            # 所以给 Age 乘 0.1，让它变成 5~10，与分类 Loss 保持在同一个量级
            w_age = 0.1
            w_gender = 1.0
            w_race = 1.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [{phase}]")

        for batch in loop:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).view(-1, 1)  # MSE 需要 float
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            optimizer.zero_grad()

            # Forward
            age_pred, gender_logits, race_logits = model(imgs)

            # Calculate Losses
            l_age = criterion_age(age_pred, ages)  # MSE
            l_gender = criterion_gender(gender_logits, genders)
            l_race = criterion_race(race_logits, races)

            # Weighted Sum
            loss = (w_age * l_age) + (w_gender * l_gender) + (w_race * l_race)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 实时显示 (注意：这里的 age 显示的是原始 MSE 值)
            loop.set_postfix(
                loss=loss.item(),
                mse=l_age.item(),
                gen=l_gender.item(),
                race=l_race.item()
            )

        # === Epoch 结束: 验证 ===
        val_mae, val_gender_acc, val_race_acc = validate(model, val_loader, device)

        # 打印报告
        print(f"Epoch {epoch + 1} Report:")
        print(f"  Train Loss : {total_loss / len(train_loader):.4f}")
        print(f"  Val Age MAE: {val_mae:.4f}")
        print(f"  Val Gender : {val_gender_acc * 100:.2f}%")
        print(f"  Val Race   : {val_race_acc * 100:.2f}%")

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_latest.pth'))

        # 保存 Best Model (Warm-up 之后才开始选)
        if epoch >= 5 and val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'laf_vit_best.pth'))
            print("  🌟 New Best Model Saved!")

    print("🎉 Training Complete.")


if __name__ == "__main__":
    main()