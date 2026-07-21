# -*- coding: utf-8 -*-
"""
可插拔注意力的 ResNet 二分类网络（与现有 main 兼容）
- 支持 resnet18 / resnet50 作为骨干
- 可选模块：space（空间注意力）、rsa（向量阶段模块）、simpletrans（跨 batch 混合）
- 头部输出为 num_classes=2（与 CrossEntropyLoss / softmax 一致）
- forward() 仅返回 logits；如需特征，调用 .extract_features(x) 或 .forward_with_feat(x)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------- 可选依赖（rsaBlock）若不存在则退化为恒等 -------
try:
    from rsaModules import rsaBlock  # noqa: F401
except Exception:
    class rsaBlock(nn.Module):
        
        def __init__(self, in_dim: int, hidden: int = 1024, drop: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
                nn.Linear(hidden, in_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 支持 [B,C] 或 [B,C,1,1]
            if x.dim() == 4:
                x = x.flatten(1)  # [B,C,1,1] -> [B,C]
            return x + self.net(x)


# ------- 空间注意力 -------
class Space_Attention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.soft = nn.Softmax(dim=2)  # 在通道内做空间 softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        att = self.conv(x)                        # (B,C,H,W)
        b, c, h, w = att.shape
        att = att.view(b, c, -1)                  # (B,C,H*W)
        att = self.soft(att)                      # 每个通道按空间归一化
        att = att.view(b, c, h, w)                # 回到 (B,C,H,W)
        # 以每通道最大值做归一，保证不放大到 >1
        m = att.view(b, c, -1).amax(dim=2, keepdim=True).clamp_min(1e-6)  # (B,C,1)
        att = att.view(b, c, -1) / m              # (B,C,H*W)
        att = att.view(b, c, h, w) 
        return x * att


# ------- 简化“跨 batch”特征混合模块 -------
class SimpleTrans_Feature(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.norm = nn.LayerNorm(in_dim)
        self.fc   = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_dim, 1))
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        b, c, h, w = x.shape
        feat = self.pool(x).view(b, c)           # (B,C)
        score = self.fc(self.norm(feat))         # (B,1)
        attn  = self.soft(score @ score.t() * 0.5)  # (B,B)
        x_flat = x.view(b, -1)                   # (B, C*H*W)
        x_mix  = attn @ x_flat                   # (B, C*H*W)
        return x + x_mix.view(b, c, h, w)


# ------- 主干网络 -------
class HybridResNet(nn.Module):
    """
    method 例子：
      - 'resnet18' / 'resnet50'
      - 'resnet18_space' / 'resnet50_space'
      - 'resnet18_space_rsa' / 'resnet50_space_rsa'
    simpletrans=True 时，会在特征图阶段加入 SimpleTrans_Feature
    """
    def __init__(self,
                 method: str = 'resnet18',
                 num_classes: int = 1,
                 pretrained: bool = True,
                 simpletrans: bool = False):
        super().__init__()
        self.method = method
        self.use_space = ('space' in method)
        self.use_rsa   = ('rsa'   in method)
        self.use_simpletrans = bool(simpletrans)

        if 'resnet50' in method:
            backbone = models.resnet50
            weights_enum = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.feat_dim = 2048
        else:
            backbone = models.resnet18
            weights_enum = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.feat_dim = 512

        base = backbone(weights=weights_enum)
        # 拆出卷积特征（到 layer4）
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.use_space:
            self.space_attn = Space_Attention(self.feat_dim)
        if self.use_simpletrans:
            self.simpletrans_block = SimpleTrans_Feature(self.feat_dim)
        if self.use_rsa:
            self.rsa = rsaBlock(self.feat_dim)

        # —— 2 类头：与 CrossEntropyLoss / softmax 兼容 ——
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    # 提取 (B,C,H,W) 特征图
    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.use_simpletrans:
            x = self.simpletrans_block(x)
        if self.use_space:
            x = self.space_attn(x)
        return x

    # 提取 (B,C) 向量特征
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_feature_map(x)
        x = self.avgpool(x).flatten(1)
        if self.use_rsa:
            x = self.rsa(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)          # (B,C)
        logits = self.classifier(feat)           # (B,2)
        return logits

    def forward_with_feat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits, feat


def build_custom_model(method: str = 'resnet18',
                       num_classes: int = 2,
                       pretrained: str = 'imagenet',
                       device: str = 'cuda',
                       simpletrans: bool = False) -> HybridResNet:
    """
    统一构造函数，行为与你 main 里的 build_model 一致：
      - pretrained='imagenet' → 使用 ImageNet 预训练；其他取值视为 False
      - 返回放到 device 上
    """
    use_pre = isinstance(pretrained, str) and pretrained.lower() == 'imagenet'
    m = HybridResNet(method=method, num_classes=num_classes,
                     pretrained=use_pre, simpletrans=simpletrans)
    return m.to(device)
