import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import numpy as np
import random
import sys

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--model_path', type=str, required=True, help='Checkpoint路径')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42, help='必须和训练时保持一致')
parser.add_argument('--val_percent', type=int, default=10, help='验证集比例')
parser.add_argument('--use_hard', action='store_true', help='开启 Hard Conditioning 模式 (必须与训练时一致)')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_row(name, count, mae, std, g_acc, r_acc):
    """辅助函数：打印表格的一行"""
    print(f"{name:<20} | {count:<8} | {mae:<10.4f} | {std:<8.2f} | {g_acc:<9.1f}% | {r_acc:<9.1f}%")


def analyze_metrics(data_dict, category_name):
    """
    辅助函数：计算并打印某一类的统计数据
    返回该类的 (mae, g_acc, r_acc) 用于加权平均
    """
    if len(data_dict['age_errs']) == 0:
        return None

    mae = np.mean(data_dict['age_errs'])
    std = np.std(data_dict['age_errs'])
    g_acc = np.mean(data_dict['gender_hits']) * 100
    r_acc = np.mean(data_dict['race_hits']) * 100
    count = len(data_dict['age_errs'])

    print_row(category_name, count, mae, std, g_acc, r_acc)
    return mae


def main():
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Evaluation | Device: {device} | Seed: {args.seed}")

    # 1. 加载模型
    print(f"📂 Loading model: {args.model_path}")
    # 注意：确保 use_hard_conditioning 参数与训练时一致
    model = LaFViT(pretrained=False, use_hard_conditioning=args.use_hard)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 构建数据集
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

    # ==========================================
    # 3. 初始化数据容器
    # ==========================================
    # 结构: metrics[key] = {'age_errs': [], 'gender_hits': [], 'race_hits': []}

    # 按种族 (0-4)
    race_metrics = {r: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for r in range(5)}
    # 按性别 (0:Male, 1:Female)
    gender_metrics = {g: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for g in range(2)}
    # 按组合 (Race_idx * 2 + Gender_idx) -> 比如 0=WhiteMale, 1=WhiteFemale
    combo_metrics = {k: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for k in range(10)}
    # 整体
    overall_metrics = {'age_errs': [], 'gender_hits': [], 'race_hits': []}

    print("Starting evaluation...")

    # ==========================================
    # 4. 推理循环
    # ==========================================
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", file=sys.stdout):
            imgs = batch['image'].to(device)
            ages_true = batch['age'].to(device).view(-1, 1)
            genders_true = batch['gender'].to(device)
            races_true = batch['race'].to(device)

            # --- Forward ---
            age_pred, gender_logits, race_logits = model(imgs, stage="stage2")

            # --- 还原真实年龄 ---
            age_pred_real = age_pred * 100.0

            # --- 计算误差 ---
            abs_errs = torch.abs(age_pred_real - ages_true).cpu().numpy()
            gender_hits = (torch.argmax(gender_logits, 1) == genders_true).cpu().numpy()
            race_hits = (torch.argmax(race_logits, 1) == races_true).cpu().numpy()

            # 转 CPU 以便索引
            g_cpu = genders_true.cpu().numpy()
            r_cpu = races_true.cpu().numpy()

            # --- 填入容器 ---
            for i in range(len(imgs)):
                r = int(r_cpu[i])
                g = int(g_cpu[i])
                combo_key = r * 2 + g  # 0~9 的唯一索引

                err = abs_errs[i][0]
                gh = int(gender_hits[i])
                rh = int(race_hits[i])

                # 1. Race
                race_metrics[r]['age_errs'].append(err)
                race_metrics[r]['gender_hits'].append(gh)
                race_metrics[r]['race_hits'].append(rh)

                # 2. Gender
                gender_metrics[g]['age_errs'].append(err)
                gender_metrics[g]['gender_hits'].append(gh)
                gender_metrics[g]['race_hits'].append(rh)

                # 3. Combo
                combo_metrics[combo_key]['age_errs'].append(err)
                combo_metrics[combo_key]['gender_hits'].append(gh)
                combo_metrics[combo_key]['race_hits'].append(rh)

                # 4. Overall
                overall_metrics['age_errs'].append(err)
                overall_metrics['gender_hits'].append(gh)
                overall_metrics['race_hits'].append(rh)

    # ==========================================
    # 5. 打印完整报告
    # ==========================================
    race_names = ["White", "Black", "Asian", "Indian", "Others"]
    gender_names = ["Male", "Female"]

    print("\n" + "=" * 90)
    print(f"{'COMPREHENSIVE FAIRNESS REPORT':^90}")
    print("=" * 90)
    print(f"{'Group':<20} | {'Count':<8} | {'Age MAE':<10} | {'Std':<8} | {'Gen Acc':<10} | {'Race Acc':<10}")
    print("-" * 90)

    # --- Section 1: 按性别 ---
    print(f" >>> BY GENDER")
    for g in range(2):
        analyze_metrics(gender_metrics[g], gender_names[g])
    print("-" * 90)

    # --- Section 2: 按种族 ---
    print(f" >>> BY RACE")
    for r in range(5):
        analyze_metrics(race_metrics[r], race_names[r])
    print("-" * 90)

    # --- Section 3: 交叉组合 (最详细) ---
    print(f" >>> BY SUBGROUP (Race + Gender)")
    for r in range(5):
        for g in range(2):
            combo_key = r * 2 + g
            name = f"{race_names[r]}-{gender_names[g]}"
            analyze_metrics(combo_metrics[combo_key], name)

    # --- Section 4: 总体 ---
    print("=" * 90)
    if len(overall_metrics['age_errs']) > 0:
        analyze_metrics(overall_metrics, "OVERALL")
    print("=" * 90)


# 需要加个 tqdm 方便看进度，如果没有装 tqdm 可以删掉
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

if __name__ == "__main__":
    main()