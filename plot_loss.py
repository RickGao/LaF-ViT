import re
import matplotlib.pyplot as plt

def parse_loss_with_stage(log_path):
    epoch_re = re.compile(
        r"Epoch\s+(\d+)\s+Report\s+\|\s+Train Loss:\s*([0-9.]+)"
    )
    stage2_re = re.compile(r"Switch to Stage 2", re.IGNORECASE)

    epochs, losses = [], []
    stage2_epoch = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if stage2_epoch is None and stage2_re.search(line):
                stage2_epoch = -1

            m = epoch_re.search(line)
            if not m:
                continue

            ep = int(m.group(1))
            loss = float(m.group(2))
            epochs.append(ep)
            losses.append(loss)

            if stage2_epoch == -1:
                stage2_epoch = ep

    return epochs, losses, stage2_epoch

log_path = "./log/train_log_20251213_mae.txt"
epochs, losses, stage2_epoch = parse_loss_with_stage(log_path)

# ---- Split stages
s1_epochs = [e for e in epochs if e < stage2_epoch]
s1_losses = losses[:len(s1_epochs)]

s2_epochs = [e for e in epochs if e >= stage2_epoch]
s2_losses = losses[len(s1_epochs):]

# ---- Plot Stage 1
plt.figure(figsize=(6, 4))
plt.plot(s1_epochs, s1_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Stage 1 Train Loss (Small + Base)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Plot Stage 2
plt.figure(figsize=(6, 4))
plt.plot(s2_epochs, s2_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Stage 2 Train Loss (Base Only)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()