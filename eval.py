import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--model_path', type=str, required=True, help='Checkpoint路径')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42, help='必须和训练时保持一致')
parser.add_argument('--val_percent', type=int, default=10, help='验证集比例')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Evaluation | Device: {device} | Seed: {args.seed}")

    # 1. 加载模型
    print(f"📂 Loading model: {args.model_path}")
    model = LaFViT(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 智能构建数据集
    full_dataset = UTKFaceDataset(args.data_dir, transform=val_transforms)
    total_len = len(full_dataset)

    if args.val_percent > 0:
        train_len = int(total_len * (100 - args.val_percent) / 100)
        val_len = total_len - train_len
        gen = torch.Generator().manual_seed(args.seed)
        _, val_subset = random_split(full_dataset, [train_len, val_len], generator=gen)
        print(f"⚠️ Mode: Validation Set Only ({args.val_percent}%)")
        dataset = val_subset
    else:
        print("⚠️ Mode: Full Dataset")
        dataset = full_dataset

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. 评估
    race_metrics = {k: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for k in range(5)}

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages_true = batch['age'].to(device).view(-1, 1)
            genders_true = batch['gender'].to(device)
            races_true = batch['race'].to(device)

            # 推理
            age_pred, gender_logits, race_logits = model(imgs, stage="stage2")

            # 🔥【关键修复】: 还原年龄！(归一化逆操作)
            # 模型输出 0.45 -> 还原为 45.0 岁
            age_pred_real = age_pred * 100.0

            # 计算误差 (用还原后的值和真实值比较)
            abs_err = torch.abs(age_pred_real - ages_true).cpu().numpy()

            gender_correct = (torch.argmax(gender_logits, 1) == genders_true).cpu().numpy()
            race_correct = (torch.argmax(race_logits, 1) == races_true).cpu().numpy()
            races_cpu = races_true.cpu().numpy()

            for i in range(len(imgs)):
                r = races_cpu[i]
                race_metrics[r]['age_errs'].append(abs_err[i][0])
                race_metrics[r]['gender_hits'].append(int(gender_correct[i]))
                race_metrics[r]['race_hits'].append(int(race_correct[i]))

    # 4. 打印报告
    print("\n" + "=" * 85)
    print(f"{'FAIRNESS REPORT':^85}")
    print("=" * 85)
    print(f"{'Group':<10} | {'Count':<8} | {'Age MAE':<10} | {'Std':<8} | {'Gen Acc':<10} | {'Race Acc':<10}")
    print("-" * 85)

    race_names = ["White", "Black", "Asian", "Indian", "Others"]
    all_errs, all_gen, all_race = [], [], []

    for r_idx in range(5):
        data = race_metrics[r_idx]
        if len(data['age_errs']) > 0:
            mae = np.mean(data['age_errs'])
            std = np.std(data['age_errs'])
            g_acc = np.mean(data['gender_hits']) * 100
            r_acc = np.mean(data['race_hits']) * 100

            all_errs.extend(data['age_errs'])
            all_gen.extend(data['gender_hits'])
            all_race.extend(data['race_hits'])

            print(
                f"{race_names[r_idx]:<10} | {len(data['age_errs']):<8} | {mae:<10.4f} | {std:<8.2f} | {g_acc:<9.1f}% | {r_acc:<9.1f}%")

    print("-" * 85)
    if all_errs:
        print(
            f"{'OVERALL':<10} | {len(all_errs):<8} | {np.mean(all_errs):<10.4f} | {'-':<8} | {np.mean(all_gen) * 100:<9.1f}% | {np.mean(all_race) * 100:<9.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    main()