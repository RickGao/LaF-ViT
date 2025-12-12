import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class LaFViT(nn.Module):
    def __init__(self, pretrained=True):
        super(LaFViT, self).__init__()

        # 1. ViT Backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.feature_dim = 768

        # 2. Demographic Heads (升级: 从单层 Linear 变为 MLP)
        # 给分类头更多的参数，让它们能独立学出特征，不依赖 Age 的梯度
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        self.race_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

        # 3. Age Regression Head
        combined_dim = self.feature_dim + 2 + 5
        self.age_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, ablation_hard_label=False):
        # Step 1: Features
        features = self.backbone(x)

        # Step 2: Logits
        gender_logits = self.gender_head(features)
        race_logits = self.race_head(features)

        # Step 3: Soft Probs
        gender_probs = F.softmax(gender_logits, dim=1)
        race_probs = F.softmax(race_logits, dim=1)

        # Ablation Logic
        if ablation_hard_label:
            g_idx = torch.argmax(gender_probs, dim=1)
            r_idx = torch.argmax(race_probs, dim=1)
            gender_probs = F.one_hot(g_idx, num_classes=2).float()
            race_probs = F.one_hot(r_idx, num_classes=5).float()

        # Step 4: Concat
        combined = torch.cat([features, gender_probs, race_probs], dim=1)

        # Step 5: Age
        age_pred = self.age_head(combined)

        return age_pred, gender_logits, race_logits