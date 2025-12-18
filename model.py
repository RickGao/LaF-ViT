import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LaFViT(nn.Module):
    def __init__(self, pretrained=True, use_hard_conditioning=False):
        super(LaFViT, self).__init__()

        self.use_hard_conditioning = use_hard_conditioning

        # --- Demographic Backbone (ViT-Small) ---
        self.demo_backbone = timm.create_model(
            'vit_small_patch16_224', pretrained=pretrained, num_classes=0, drop_path_rate=0.05
        )
        self.demo_dim = 384

        # --- Age Backbone (ViT-Base) ---
        self.age_backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=pretrained, num_classes=0, drop_path_rate=0.2
        )
        self.age_dim = 768

        # --- Classification Heads ---
        self.gender_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)
        )
        self.race_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5)
        )

        # --- Age Regression Head ---
        # Input: Age Features + Gender One-hot/Softmax (2) + Race One-hot/Softmax (5)
        combined_dim = self.age_dim + 2 + 5
        self.age_head = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, stage="stage2"):
        # --- Stream A: Demographic (ViT-Small) ---
        # Note: Compute Logits only; Softmax/Argmax applied during conditioning logic
        if stage == "stage2":
            with torch.no_grad():
                features_demo = self.demo_backbone(x)
                gender_logits = self.gender_head(features_demo)
                race_logits = self.race_head(features_demo)
        else:
            features_demo = self.demo_backbone(x)
            gender_logits = self.gender_head(features_demo)
            race_logits = self.race_head(features_demo)

        if stage == "stage1":
            # Return dummy age value during demographic pre-training stage
            dummy_age = torch.zeros(x.size(0), 1).to(x.device)
            return dummy_age, gender_logits, race_logits

        # --- Stream B: Age (ViT-Base) ---
        features_age = self.age_backbone(x)

        # --- Conditioning Logic ---
        if self.use_hard_conditioning:
            # === Hard Mode (Ablation) ===
            # 1. Identify indices with maximum probability
            g_idx = torch.argmax(gender_logits, dim=1)
            r_idx = torch.argmax(race_logits, dim=1)

            # 2. Convert to One-Hot vectors (Float)
            g_cond = F.one_hot(g_idx, num_classes=2).float()
            r_cond = F.one_hot(r_idx, num_classes=5).float()
        else:
            # === Soft Mode (Default) ===
            # Use Softmax probabilities directly for differentiable flow
            g_cond = F.softmax(gender_logits, dim=1)
            r_cond = F.softmax(race_logits, dim=1)

        # --- Feature Fusion ---
        # Dimensions remain consistent across both hard and soft modes
        combined = torch.cat([features_age, g_cond, r_cond], dim=1)

        age_pred = self.age_head(combined)

        return age_pred, gender_logits, race_logits