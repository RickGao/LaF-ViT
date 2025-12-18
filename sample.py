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
    parser = argparse.ArgumentParser(description="Visualize Random Predictions (No Filtering)")
    parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='Dataset path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best checkpoint')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of images to save')

    # 验证集划分参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--val_percent', type=int, default=10, help='Validation split percentage')
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='sample_random', help='Output directory')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(tensor):
    """还原归一化的图片以便显示"""
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

    print(f"🚀 Sampling Random Predictions | Device: {device} | Seed: {args.seed}")
    print(f"📂 Output folder: {args.output_dir}")

    # --- 1. 设置随机种子 ---
    set_seed(args.seed)

    # --- 2. 加载模型 ---
    print(f"🧠 Loading model from: {args.model_path}")
    # 加上 use_hard 参数以兼容你的 Ablation 模型
    model = LaFViT(pretrained=False)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 3. 严格复现验证集划分 ---
    print(f"📊 Reconstructing Validation Set (Split: {args.val_percent}%)")
    full_ds = UTKFaceDataset(args.data_dir, transform=val_transforms)
    total_len = len(full_ds)

    val_len = int(total_len * args.val_percent / 100)
    train_len = total_len - val_len

    gen = torch.Generator().manual_seed(args.seed)
    _, val_subset = random_split(full_ds, [train_len, val_len], generator=gen)

    print(f"   -> Validation set size: {len(val_subset)} images")

    # Shuffle=True 保证随机性
    loader = DataLoader(val_subset, batch_size=32, shuffle=True, num_workers=2)

    # --- 4. 收集样本 (不筛选) ---
    print("🔍 Collecting random samples (showing both Correct and Incorrect predictions)...")
    samples = []

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
                if len(samples) >= args.num_samples:
                    break

                # 直接添加，不再判断 if correct
                samples.append({
                    'img': imgs[i].cpu(),
                    'gt_age': ages[i].item(),
                    'pred_age': pred_ages[i].item(),
                    'gt_gen': genders[i].item(),
                    'pred_gen': pred_genders[i].item(),
                    'gt_race': races[i].item(),
                    'pred_race': pred_races[i].item()
                })

            if len(samples) >= args.num_samples:
                break

    # --- 5. 独立绘图与保存 ---
    print(f"🎨 Saving {len(samples)} individual images to {args.output_dir}...")

    for idx, sample in enumerate(samples):
        plt.figure(figsize=(4, 5.0))  # 稍微调高一点画布，给三行文字留空间

        # 显示图片
        vis_img = denormalize(sample['img'])
        plt.imshow(vis_img)
        plt.axis('off')

        # 准备标签文字
        p_age = sample['pred_age']
        t_age = sample['gt_age']

        p_gen_str = GENDER_MAP[sample['pred_gen']]
        t_gen_str = GENDER_MAP[sample['gt_gen']]

        p_race_str = RACE_MAP[sample['pred_race']]
        t_race_str = RACE_MAP[sample['gt_race']]

        # 构造详细的三行文字：Pred vs GT
        title_text = (
            f"Age: {p_age:.1f} (GT: {t_age:.0f})\n"
            f"Gen: {p_gen_str} (GT: {t_gen_str})\n"
            f"Race: {p_race_str} (GT: {t_race_str})"
        )

        # 标记颜色：如果误差大或者分类错，可以用红色边框(这里简单起见只用文字)
        # 美化文字框
        plt.title(title_text, fontsize=11, fontweight='bold', pad=10,
                  bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.tight_layout()

        # 生成唯一的文件名
        filename = f"sample_{idx}_GT{t_age:.0f}_{t_gen_str}_{t_race_str}.png"
        save_path = os.path.join(args.output_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {filename}")

    print(f"✅ Done!")


if __name__ == "__main__":
    main()