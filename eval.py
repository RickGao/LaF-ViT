import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UTKFaceDataset, val_transforms  # 注意用 val_transforms
from model import LaFViT
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--model_path', type=str, required=True, help='训练好的 .pth 文件路径')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    model = LaFViT(pretrained=False)  # 推理不需要下载预训练权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 加载数据 (全部数据用于测试，或者你可以单独指定测试集)
    dataset = UTKFaceDataset(args.data_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. 存储结果用于分析
    # 种族映射: 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
    race_errors = {0: [], 1: [], 2: [], 3: [], 4: []}
    total_ae = 0  # Absolute Error
    count = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages_true = batch['age'].to(device).view(-1, 1)
            races = batch['race'].cpu().numpy()  # 拿回 CPU 做字典索引

            # Forward
            age_pred, _, _ = model(imgs)

            # 计算绝对误差 (Absolute Error)
            abs_err = torch.abs(age_pred - ages_true).cpu().numpy()  # [B, 1]

            # 记录全局误差
            total_ae += np.sum(abs_err)
            count += len(imgs)

            # 按种族记录误差 (Fairness Metrics)
            for i in range(len(imgs)):
                r = races[i]
                err = abs_err[i][0]
                if r in race_errors:
                    race_errors[r].append(err)

    # 4. 打印报告
    print("=" * 40)
    print(f"Global MAE (Mean Absolute Error): {total_ae / count:.4f} years")
    print("=" * 40)
    print("Fairness Analysis (MAE by Race):")

    race_names = ["White", "Black", "Asian", "Indian", "Others"]
    for r_idx in range(5):
        errors = race_errors[r_idx]
        if len(errors) > 0:
            mae = np.mean(errors)
            std = np.std(errors)
            print(f"  {race_names[r_idx]:<10}: MAE = {mae:.4f} (std: {std:.2f}) | Samples: {len(errors)}")
        else:
            print(f"  {race_names[r_idx]:<10}: No samples found.")
    print("=" * 40)


if __name__ == "__main__":
    main()