import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LaFViT(nn.Module):
    def __init__(self, pretrained=True):
        super(LaFViT, self).__init__()

        # =========================================================
        # 1. 定义两个 Backbone
        # =========================================================

        # --- Stream A: Demographic (ViT-Small) ---
        # drop_path_rate=0.1: 给小模型一点点正则化
        self.demo_backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.1  # <--- 新增
        )
        self.demo_dim = 384

        # --- Stream B: Age (ViT-Base) ---
        # drop_path_rate=0.2: 给大模型强力的正则化 (20%概率丢层)
        self.age_backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.2  # <--- 新增 (这非常关键！)
        )
        self.age_dim = 768

        # =========================================================
        # 2. 定义 Heads
        # =========================================================

        self.gender_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.race_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

        # Age Head
        combined_dim = self.age_dim + 2 + 5
        self.age_head = nn.Sequential(
            nn.Linear(combined_dim, 512),  # 第一层变宽
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),  # 增加一层中间层
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1)  # 输出层
        )

    def forward(self, x, stage="stage2"):
        # Path A: Demographic Stream (ViT-Small)
        if stage == "stage2":
            with torch.no_grad():
                features_demo = self.demo_backbone(x)
                gender_logits = self.gender_head(features_demo)
                race_logits = self.race_head(features_demo)
                gender_probs = F.softmax(gender_logits, dim=1)
                race_probs = F.softmax(race_logits, dim=1)
        else:
            features_demo = self.demo_backbone(x)
            gender_logits = self.gender_head(features_demo)
            race_logits = self.race_head(features_demo)
            gender_probs = F.softmax(gender_logits, dim=1)
            race_probs = F.softmax(race_logits, dim=1)

        if stage == "stage1":
            dummy_age = torch.zeros(x.size(0), 1).to(x.device)
            return dummy_age, gender_logits, race_logits

        # Path B: Age Stream (ViT-Base)
        features_age = self.age_backbone(x)
        combined = torch.cat([features_age, gender_probs, race_probs], dim=1)
        age_pred = self.age_head(combined)

        return age_pred, gender_logits, race_logits