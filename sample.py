import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ==========================================
# 1. 标签映射字典
# ==========================================
GENDER_MAP = {0: 'Male', 1: 'Female'}
RACE_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Best Predictions from Validation Set (Individual Images)")
    parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='Dataset path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best checkpoint')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of images to save')

    # 验证集划分参数 (必须和训练一致)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--val_percent', type=int, default=10, help='Validation split percentage (default: 10)')

    # 输出目录 (默认存到 sample 文件夹)
    parser.add_argument('--output_dir', type=str, default='sample_individual', help='Output directory for individual images')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(tensor):
    """还原归一化的图片以便显示 (ImageNet Stats)"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 0. 准备输出目录 ---
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Sampling Individual Best Predictions | Device: {device} | Seed: {args.seed}")
    print(f"📂 Output folder: {args.output_dir}")

    # --- 1. 设置随机种子 (至关重要) ---
    set_seed(args.seed)

    # --- 2. 加载模型 ---
    print(f"🧠 Loading model from: {args.model_path}")
    # 注意：这里假设你的模型不需要 use_hard 参数，如果需要请自行添加
    model = LaFViT(pretrained=False)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 3. 严格复现验证集划分 ---
    print(f"📊 Reconstructing Validation Set (Split: {args.val_percent}%)")
    full_ds = UTKFaceDataset(args.data_dir, transform=val_transforms)
    total_len = len(full_ds)

    # 计算划分长度
    val_len = int(total_len * args.val_percent / 100)
    train_len = total_len - val_len

    # 使用 generator 确保和训练时的随机划分一模一样
    gen = torch.Generator().manual_seed(args.seed)
    _, val_subset = random_split(full_ds, [train_len, val_len], generator=gen)

    print(f"   -> Validation set size: {len(val_subset)} images")

    # Shuffle=True 这里是为了在验证集里随机挑图
    loader = DataLoader(val_subset, batch_size=32, shuffle=True, num_workers=2)

    # --- 4. 寻找“完美”样本 ---
    print("🔍 Searching for high-quality predictions (Age Err < 3, Gender & Race Correct)...")
    best_samples = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device)
            genders = batch['gender'].to(device)
            races = batch['race'].to(device)

            # 推理
            age_preds, g_logits, r_logits = model(imgs, stage="stage2")

            # 还原数值
            pred_ages = age_preds.flatten() * 100.0
            pred_genders = torch.argmax(g_logits, dim=1)
            pred_races = torch.argmax(r_logits, dim=1)

            # 遍历 Batch
            for i in range(len(imgs)):
                if len(best_samples) >= args.num_samples:
                    break

                # 筛选条件
                age_err = abs(pred_ages[i].item() - ages[i].item())
                g_correct = (pred_genders[i] == genders[i])
                r_correct = (pred_races[i] == races[i])

                # 挑选误差特别小 (< 3岁) 且分类全对的样本
                if age_err < 3.0 and g_correct and r_correct:
                    best_samples.append({
                        'img': imgs[i].cpu(),
                        'gt_age': ages[i].item(),
                        'pred_age': pred_ages[i].item(),
                        'gt_gen': genders[i].item(),
                        'pred_gen': pred_genders[i].item(),
                        'gt_race': races[i].item(),
                        'pred_race': pred_races[i].item()
                    })

            if len(best_samples) >= args.num_samples:
                break

    # --- 5. 独立绘图与保存 ---
    if not best_samples:
        print("⚠️ No perfect samples found in this batch. Try increasing error threshold or batch size.")
        return

    print(f"🎨 Saving {len(best_samples)} individual images to {args.output_dir}...")

    for idx, sample in enumerate(best_samples):
        # 创建一个新的画布
        plt.figure(figsize=(4, 4.5))

        # 显示图片
        vis_img = denormalize(sample['img'])
        plt.imshow(vis_img)
        plt.axis('off')

        # 准备标签文字
        p_age = sample['pred_age']
        t_age = sample['gt_age']
        p_gen = GENDER_MAP[sample['pred_gen']]
        t_gen = GENDER_MAP[sample['gt_gen']]
        p_race = RACE_MAP[sample['pred_race']]
        t_race = RACE_MAP[sample['gt_race']]

        # 构造文字：上面是预测值(GT)，下面是人口属性
        title_text = (
            f"Pred: {p_age:.1f} (GT: {t_age:.0f})\n"
            f"{p_gen} | {p_race}"
        )

        # 美化文字框，放在图片下方