import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LaFViT(nn.Module):
    def __init__(self, pretrained=True):
        super(LaFViT, self).__init__()

        # 1. 加载 ViT Backbone (使用 patch16_224)
        # num_classes=0 表示去掉原始的 ImageNet 分类头，只取 Feature
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        # ViT Base 的特征维度通常是 768
        self.feature_dim = 768

        # 2. Demographic Heads (辅助任务: 性别和种族分类)
        self.gender_head = nn.Linear(self.feature_dim, 2)  # 0/1
        self.race_head = nn.Linear(self.feature_dim, 5)  # 0-4

        # 3. Age Regression Head (主任务)
        # 输入 = 视觉特征(768) + 性别概率(2) + 种族概率(5)
        combined_dim = self.feature_dim + 2 + 5

        self.age_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加 Dropout 防止过拟合
            nn.Linear(256, 1)  # 输出具体的年龄值
        )

    def forward(self, x, ablation_hard_label=False):
        """
        Args:
            x: 输入图像 [Batch, 3, 224, 224]
            ablation_hard_label: 如果为 True，做消融实验，使用 Hard Label (0/1) 拼接
        """
        # Step 1: 提取全局视觉特征
        features = self.backbone(x)  # [Batch, 768]

        # Step 2: 预测人口统计学 Logits
        gender_logits = self.gender_head(features)
        race_logits = self.race_head(features)

        # Step 3: 生成 Soft Probabilities (Soft Conditioning)
        gender_probs = F.softmax(gender_logits, dim=1)  # [Batch, 2]
        race_probs = F.softmax(race_logits, dim=1)  # [Batch, 5]

        # --- Ablation Study Logic (消融实验) ---
        if ablation_hard_label:
            # 如果是 Hard Label，就把概率变成 One-Hot
            # 例如: [0.1, 0.9] -> [0, 1]
            g_idx = torch.argmax(gender_probs, dim=1)
            r_idx = torch.argmax(race_probs, dim=1)
            gender_probs = F.one_hot(g_idx, num_classes=2).float()
            race_probs = F.one_hot(r_idx, num_classes=5).float()
        # -------------------------------------

        # Step 4: 拼接特征 (Concatenation)
        # 将 "视觉理解" 与 "人口统计学先验" 结合
        combined = torch.cat([features, gender_probs, race_probs], dim=1)

        # Step 5: 预测年龄
        age_pred = self.age_head(combined)

        return age_pred, gender_logits, race_logits