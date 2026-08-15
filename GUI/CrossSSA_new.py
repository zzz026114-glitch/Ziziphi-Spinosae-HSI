import torch
import torch.nn as nn
import torchvision.models as models


# =========================================================
# Cross Spectral-Spatial Attention
# 空间特征引导光谱特征
# =========================================================
class CrossSSA(nn.Module):

    def __init__(
        self,
        spectral_dim,
        spatial_dim
    ):

        super(CrossSSA, self).__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                spatial_dim,
                spectral_dim
            ),

            nn.ReLU(),

            nn.Linear(
                spectral_dim,
                spectral_dim
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        spectral_feat,
        spatial_feat
    ):

        # spatial -> attention
        attn = self.attention(
            spatial_feat
        )

        # reweight spectral
        spectral_feat = (
            spectral_feat * attn
        )

        return spectral_feat


# =========================================================
# Spectral Branch
# 输入: 原始127 band
# =========================================================
class SpectralBranch(nn.Module):

    def __init__(
        self,
        spectral_dim=128
    ):

        super(SpectralBranch, self).__init__()

        self.conv1 = nn.Conv1d(
            1,
            16,
            kernel_size=7,
            padding=3
        )

        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            16,
            32,
            kernel_size=5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            64,
            spectral_dim
        )

    def forward(self, x):

        # x: (B,127,H,W)

        x = torch.mean(
            x,
            dim=(2,3)
        )

        # (B,127)
        # ->
        # (B,1,127)

        x = x.unsqueeze(1)

        x = torch.relu(
            self.bn1(self.conv1(x))
        )

        x = torch.relu(
            self.bn2(self.conv2(x))
        )

        x = torch.relu(
            self.bn3(self.conv3(x))
        )

        x = self.pool(x).squeeze(-1)

        x = self.fc(x)

        return x


# =========================================================
# Spatial Branch
# 输入: PCA25
# =========================================================
class SpatialBranch(nn.Module):

    def __init__(
        self,
        in_channels=25,
        spatial_dim=256
    ):

        super(SpatialBranch, self).__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(
            512,
            spatial_dim
        )

    def forward(self, x):

        # x: (B,25,H,W)

        x = self.backbone(x)

        x = self.fc(x)

        return x


# =========================================================
# Spectral-Spatial Network
# =========================================================
class SpectralSpatialNet(nn.Module):

    def __init__(
        self,
        num_classes,
        spectral_dim=128,
        spatial_dim=256,
        dropout=0.5
    ):

        super(SpectralSpatialNet, self).__init__()

        # 光谱分支
        self.spectral_branch = SpectralBranch(
            spectral_dim=spectral_dim
        )

        # 空间分支
        self.spatial_branch = SpatialBranch(
            in_channels=25,
            spatial_dim=spatial_dim
        )

        # Cross SSA
        self.cross_ssa = CrossSSA(
            spectral_dim,
            spatial_dim
        )

        fusion_dim = (
            spectral_dim
            +
            spatial_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                num_classes
            )
        )

    # =====================================================
    # Feature
    # =====================================================
    def forward_feature(
        self,
        raw_x,
        pca_x
    ):

        spectral_feat = self.spectral_branch(
            raw_x
        )

        spatial_feat = self.spatial_branch(
            pca_x
        )

        # Cross Attention
        spectral_feat = self.cross_ssa(
            spectral_feat,
            spatial_feat
        )

        fusion_feat = torch.cat(
            [spectral_feat, spatial_feat],
            dim=1
        )

        return fusion_feat

    # =====================================================
    # Forward
    # =====================================================
    def forward(
            self,
            raw_x,
            pca_x
    ):
        fusion_feat = self.forward_feature(
            raw_x,
            pca_x
        )

        logits = self.classifier(
            fusion_feat
        )

        return fusion_feat, logits