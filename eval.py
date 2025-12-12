import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--model_path', type=str, required=True, help='训练好的 .pth 文件路径')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_limit', type=int, default=0, help='测试集数量限制 (0表示跑全量)')
parser.add_argument('--seed', type=int, default=42, help='随机种子，保证每次选取的测试集一致')
args = parser.parse_args()


def set_seed(seed):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 0. 固定随机种子 (关键步骤！)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device} | Seed: {args.seed}")

    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    model = LaFViT(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 加载数据并进行随机切片
    full_dataset = UTKFaceDataset(args.data_dir, transform=val_transforms)
    total_len = len(full_dataset)

    # --- 核心修改逻辑开始 ---
    # 生成一个从 0 到 total_len-1 的随机排列索引
    # 因为我们前面设置了 seed，所以这个排列顺序每次运行都是固定的
    indices = torch.randperm(total_len).tolist()

    if args.test_limit > 0:
        limit = min(args.test_limit, total_len)
        print(f"⚠️ Slicing dataset: selecting random {limit} samples from {total_len}.")
        # 取打乱后的前 limit 个索引
        selected_indices = indices[:limit]
        dataset = Subset(full_dataset, selected_indices)
    else:
        # 如果不限制，就用全部数据（也可以选择打乱或不打乱，这里直接用原始的即可）
        dataset = full_dataset
    # --- 核心修改逻辑结束 ---

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. 存储结果结构体
    race_metrics = {0: {'age_errs': [], 'gender_hits': [], 'race_hits': []},
                    1: {'age_errs': [], 'gender_hits': [], 'race_hits': []},
                    2: {'age_errs': [], 'gender_hits': [], 'race_hits': []},
                    3: {'age_errs': [], 'gender_hits': [], 'race_hits': []},
                    4: {'age_errs': [], 'gender_hits': [], 'race_hits': []}}

    print("Starting evaluation...")

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages_true = batch['age'].to(device).view(-1, 1)
            genders_true = batch['gender'].to(device)
            races_true = batch['race'].to(device)

            # Forward
            age_pred, gender_logits, race_logits = model(imgs)

            # Metrics Calculation
            abs_err = torch.abs(age_pred - ages_true).cpu().numpy()
            gender_preds = torch.argmax(gender_logits, dim=1)
            gender_correct = (gender_preds == genders_true).cpu().numpy()
            race_preds = torch.argmax(race_logits, dim=1)
            race_correct = (race_preds == races_true).cpu().numpy()
            races_cpu = races_true.cpu().numpy()

            for i in range(len(imgs)):
                r = races_cpu[i]
                race_metrics[r]['age_errs'].append(abs_err[i][0])
                race_metrics[r]['gender_hits'].append(int(gender_correct[i]))
                race_metrics[r]['race_hits'].append(int(race_correct[i]))

    # 4. 打印报告 (保持之前的格式)
    print("\n" + "=" * 85)
    print(f"{'FAIRNESS ANALYSIS REPORT (Seed: ' + str(args.seed) + ')':^85}")
    print("=" * 85)
    print(
        f"{'Race Group':<10} | {'Samples':<8} | {'Age MAE':<10} | {'Age Std':<8} | {'Gender Acc':<12} | {'Race Acc':<12}")
    print("-" * 85)

    race_names = ["White", "Black", "Asian", "Indian", "Others"]
    all_age_errs = []
    all_gender_hits = []
    all_race_hits = []

    for r_idx in range(5):
        data = race_metrics[r_idx]
        n_samples = len(data['age_errs'])

        if n_samples > 0:
            mae = np.mean(data['age_errs'])
            std = np.std(data['age_errs'])
            gender_acc = np.mean(data['gender_hits']) * 100
            race_acc = np.mean(data['race_hits']) * 100

            all_age_errs.extend(data['age_errs'])
            all_gender_hits.extend(data['gender_hits'])
            all_race_hits.extend(data['race_hits'])

            print(
                f"{race_names[r_idx]:<10} | {n_samples:<8} | {mae:<10.4f} | {std:<8.2f} | {gender_acc:<10.2f}%  | {race_acc:<10.2f}%")
        else:
            print(f"{race_names[r_idx]:<10} | {'0':<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<12} | {'N/A':<12}")

    print("-" * 85)

    if len(all_age_errs) > 0:
        global_mae = np.mean(all_age_errs)
        global_gender_acc = np.mean(all_gender_hits) * 100
        global_race_acc = np.mean(all_race_hits) * 100
        print(
            f"{'OVERALL':<10} | {len(all_age_errs):<8} | {global_mae:<10.4f} | {'-':<8} | {global_gender_acc:<10.2f}%  | {global_race_acc:<10.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    main()