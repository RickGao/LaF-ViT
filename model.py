import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LaFViT(nn.Module):
    # 1. 增加开关参数 use_hard_conditioning
    def __init__(self, pretrained=True, use_hard_conditioning=False):
        super(LaFViT, self).__init__()

        self.use_hard_conditioning = use_hard_conditioning  # 记录开关状态

        # =========================================================
        # Backbone 和 Head 定义 (完全不变)
        # =========================================================
        self.demo_backbone = timm.create_model(
            'vit_small_patch16_224', pretrained=pretrained, num_classes=0, drop_path_rate=0.05
        )
        self.demo_dim = 384

        self.age_backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=pretrained, num_classes=0, drop_path_rate=0.2
        )
        self.age_dim = 768

        self.gender_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)
        )
        self.race_head = nn.Sequential(
            nn.Linear(self.demo_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5)
        )

        combined_dim = self.age_dim + 2 + 5
        self.age_head = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, stage="stage2"):
        # --- Stream A: Demographic (ViT-Small) ---
        # 注意：这里只计算 Logits，暂时不计算 Softmax/Probs
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
            dummy_age = torch.zeros(x.size(0), 1).to(x.device)
            return dummy_age, gender_logits, race_logits

        # --- Stream B: Age (ViT-Base) ---
        features_age = self.age_backbone(x)

        # =========================================================
        # 🔥 核心修改: 根据开关决定 Conditioning 方式
        # =========================================================
        if self.use_hard_conditioning:
            # === Hard Mode (Ablation) ===
            # 1. 找到最大概率的索引 (Argmax)
            g_idx = torch.argmax(gender_logits, dim=1)
            r_idx = torch.argmax(race_logits, dim=1)

            # 2. 转成 One-Hot 向量 (必须转 float)
            # 例如: [0.1, 0.9] -> index 1 -> [0.0, 1.0]
            g_cond = F.one_hot(g_idx, num_classes=2).float()
            r_cond = F.one_hot(r_idx, num_classes=5).float()
        else:
            # === Soft Mode (Default) ===
            # 直接计算 Softmax 概率
            # 例如: [0.1, 0.9] -> [0.1, 0.9]
            g_cond = F.softmax(gender_logits, dim=1)
            r_cond = F.softmax(race_logits, dim=1)

        # 拼接 (维度在两种模式下都是一样的)
        combined = torch.cat([features_age, g_cond, r_cond], dim=1)

        age_pred = self.age_head(combined)

        return age_pred, gender_logits, race_logits