import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from dataset import UTKFaceDataset, val_transforms
from model import LaFViT
import numpy as np
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/UTKFace')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--val_percent', type=int, default=10)
parser.add_argument('--use_hard', action='store_true')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_row(name, count, mae, std, g_acc, r_acc):
    print(f"{name:<20} | {count:<8} | {mae:<10.4f} | {std:<8.2f} | {g_acc:<9.1f}% | {r_acc:<9.1f}%")


def analyze_metrics(data_dict, category_name):

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

    print(f"📂 Loading model: {args.model_path}")
    model = LaFViT(pretrained=False, use_hard_conditioning=args.use_hard)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

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

    race_metrics = {r: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for r in range(5)}
    gender_metrics = {g: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for g in range(2)}
    combo_metrics = {k: {'age_errs': [], 'gender_hits': [], 'race_hits': []} for k in range(10)}
    overall_metrics = {'age_errs': [], 'gender_hits': [], 'race_hits': []}

    print("Starting evaluation...")


    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", file=sys.stdout):
            imgs = batch['image'].to(device)
            ages_true = batch['age'].to(device).view(-1, 1)
            genders_true = batch['gender'].to(device)
            races_true = batch['race'].to(device)

            # --- Forward ---
            age_pred, gender_logits, race_logits = model(imgs, stage="stage2")

            age_pred_real = age_pred * 100.0

            abs_errs = torch.abs(age_pred_real - ages_true).cpu().numpy()
            gender_hits = (torch.argmax(gender_logits, 1) == genders_true).cpu().numpy()
            race_hits = (torch.argmax(race_logits, 1) == races_true).cpu().numpy()

            g_cpu = genders_true.cpu().numpy()
            r_cpu = races_true.cpu().numpy()

            for i in range(len(imgs)):
                r = int(r_cpu[i])
                g = int(g_cpu[i])
                combo_key = r * 2 + g

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

    race_names = ["White", "Black", "Asian", "Indian", "Others"]
    gender_names = ["Male", "Female"]

    print("\n" + "=" * 90)
    print(f"{'COMPREHENSIVE FAIRNESS REPORT':^90}")
    print("=" * 90)
    print(f"{'Group':<20} | {'Count':<8} | {'Age MAE':<10} | {'Std':<8} | {'Gen Acc':<10} | {'Race Acc':<10}")
    print("-" * 90)

    print(f" >>> BY GENDER")
    for g in range(2):
        analyze_metrics(gender_metrics[g], gender_names[g])
    print("-" * 90)

    print(f" >>> BY RACE")
    for r in range(5):
        analyze_metrics(race_metrics[r], race_names[r])
    print("-" * 90)

    print(f" >>> BY SUBGROUP (Race + Gender)")
    for r in range(5):
        for g in range(2):
            combo_key = r * 2 + g
            name = f"{race_names[r]}-{gender_names[g]}"
            analyze_metrics(combo_metrics[combo_key], name)

    print("=" * 90)
    if len(overall_metrics['age_errs']) > 0:
        analyze_metrics(overall_metrics, "OVERALL")
    print("=" * 90)


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

if __name__ == "__main__":
    main()