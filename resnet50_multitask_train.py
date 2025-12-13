import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import random
import numpy as np

# ===== 直接用你现有的 dataset.py =====
from dataset import UTKFaceDataset, train_transforms, val_transforms


# =====================================
# Config
# =====================================
DATA_DIR = "./data/UTKFace"
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# Utils
# =====================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================
# Model: ResNet50 + Multi-Head
# =====================================
class ResNet50MultiHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        base = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove fc
        feat_dim = base.fc.in_features  # 2048

        self.age_head = nn.Linear(feat_dim, 1)
        self.gender_head = nn.Linear(feat_dim, 2)
        self.race_head = nn.Linear(feat_dim, 5)

    def forward(self, x):
        feat = self.backbone(x)        # (B, 2048, 1, 1)
        feat = feat.flatten(1)         # (B, 2048)

        age = self.age_head(feat)
        gender = self.gender_head(feat)
        race = self.race_head(feat)

        return age, gender, race


# =====================================
# Validation
# =====================================
def validate(model, loader):
    model.eval()
    mae, correct_g, correct_r, count = 0, 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            ages = batch["age"].to(DEVICE).view(-1, 1)
            genders = batch["gender"].to(DEVICE)
            races = batch["race"].to(DEVICE)

            age_pred, g_logits, r_logits = model(imgs)

            age_real = age_pred * 100.0
            mae += torch.sum(torch.abs(age_real - ages)).item()

            correct_g += (g_logits.argmax(1) == genders).sum().item()
            correct_r += (r_logits.argmax(1) == races).sum().item()
            count += imgs.size(0)

    return mae / count, correct_g / count, correct_r / count


# =====================================
# Main
# =====================================
def main():
    set_seed(SEED)

    # ===== Dataset =====
    full_ds = UTKFaceDataset(DATA_DIR, transform=None)
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len

    train_ds = UTKFaceDataset(DATA_DIR, transform=train_transforms)
    val_ds = UTKFaceDataset(DATA_DIR, transform=val_transforms)

    gen = torch.Generator().manual_seed(SEED)
    train_subset, _ = random_split(train_ds, [train_len, val_len], generator=gen)
    _, val_subset = random_split(val_ds, [train_len, val_len], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ===== Model =====
    model = ResNet50MultiHead(pretrained=True).to(DEVICE)

    # ===== Loss =====
    criterion_age = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()

    race_weights = torch.tensor([1.0, 1.0, 1.0, 1.5, 4.5]).to(DEVICE)
    criterion_race = nn.CrossEntropyLoss(weight=race_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_mae = float("inf")

    # ===== Training Loop =====
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            imgs = batch["image"].to(DEVICE)
            ages = batch["age"].to(DEVICE).view(-1, 1)
            genders = batch["gender"].to(DEVICE)
            races = batch["race"].to(DEVICE)

            ages_norm = ages / 100.0

            optimizer.zero_grad()
            age_pred, g_logits, r_logits = model(imgs)

            loss_age = criterion_age(age_pred, ages_norm)
            loss_g = criterion_gender(g_logits, genders)
            loss_r = criterion_race(r_logits, races)

            loss = loss_age + loss_g + 1.5 * loss_r
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_mae, val_g, val_r = validate(model, val_loader)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Val MAE: {val_mae:.2f} | "
            f"G Acc: {val_g:.2%} | "
            f"R Acc: {val_r:.2%}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "resnet50_multitask_best.pth")
            print(f"  🌟 New Best MAE: {best_mae:.2f}")

    print("✅ Training finished")


if __name__ == "__main__":
    main()
