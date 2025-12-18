import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 标签映射字典
# ==========================================
GENDER_MAP = {0: 'Male', 1: 'Female'}
RACE_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Best Predictions")
    parser.add_argument('--data_dir', type=str, default='./data/UTKFace', help='Dataset path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best checkpoint')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of images to show')
    parser.add_argument('--use_hard', action='store_true', help='Use hard conditioning')
    parser.add_argument('--output', type=str, default='best_samples_vis.png', help='Output filename')
    return parser.parse_args()


def denormalize(tensor):
    """还原归一化的图片以便显示 (ImageNet Stats)"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # [C, H, W] -> [H, W, C]
    img = tensor.permute(1, 2, 0).cpu().numpy()

    # 反归一化
    img = img * std + mean
    return np.clip(img, 0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sampling Best Predictions | Device: {device}")

    # --- 1. 加载模型 ---
    print(f"📂 Loading model from: {args.model_path}")
    model = LaFViT(pretrained=False, use_hard_conditioning=args.use_hard)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 2. 准备验证集 ---
    # 必须保持和训练时一样的 Split 逻辑，确保看的是没见过的验证集
    full_ds = UTKFaceDataset(args.data_dir, transform=val_transforms)
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len

    # 固定种子以复现 Split
    gen = torch.Generator().manual_seed(42)
    _, val_subset = random_split(full_ds, [train_len, val_len], generator=gen)

    # Shuffle=True 让我们每次运行能随机看到不同的好样本
    loader = DataLoader(val_subset, batch_size=32, shuffle=True, num_workers=2)

    # --- 3. 寻找“好”样本 ---
    print("🔍 Searching for high-quality predictions...")
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

            # 遍历 Batch 里的每一张图
            for i in range(len(imgs)):
                if len(best_samples) >= args.num_samples:
                    break

                # --- 筛选标准 ---
                # 1. 年龄误差小于 3 岁
                age_err = abs(pred_ages[i].item() - ages[i].item())
                # 2. 性别预测正确
                g_correct = (pred_genders[i] == genders[i])
                # 3. 种族预测正确
                r_correct = (pred_races[i] == races[i])

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

    # --- 4. 绘图 ---
    if not best_samples:
        print("⚠️ No samples met the 'good prediction' criteria.")
        return

    print(f"🎨 Plotting {len(best_samples)} samples...")
    fig, axes = plt.subplots(1, len(best_samples), figsize=(4 * len(best_samples), 5))

    # 如果只有一张图，axes 不是列表，转成列表处理
    if args.num_samples == 1:
        axes = [axes]

    for idx, sample in enumerate(best_samples):
        ax = axes[idx]

        # 显示图片
        vis_img = denormalize(sample['img'])
        ax.imshow(vis_img)
        ax.axis('off')

        # 准备文字信息
        p_age = sample['pred_age']
        t_age = sample['gt_age']

        p_gen = GENDER_MAP[sample['pred_gen']]
        t_gen = GENDER_MAP[sample['gt_gen']]

        p_race = RACE_MAP[sample['pred_race']]
        t_race = RACE_MAP[sample['gt_race']]

        # 格式化文本
        title_text = (
            f"Age: {p_age:.1f} (GT: {t_age:.0f})\n"
            f"{p_gen} | {p_race}"
        )

        # 在图片下方添加文本框
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()