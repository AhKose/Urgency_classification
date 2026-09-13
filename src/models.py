"""Model definitions for the three architectures under comparison.

All share: 640x640 RGB input, ImageNet normalisation, a 2-node output, and a
3-level progressive-unfreezing scheme (0 = head only, 1 = + late blocks,
2 = + mid+late blocks).

  yolo_tl        : proposed. YOLOv8-L backbone from the domain-adapted caries
                   detector (best.pt). Multi-scale features from early (P3),
                   middle (P4) and late (P5/SPPF) blocks are global-average-
                   pooled, concatenated (1280-d) and fed to a 2-hidden-layer
                   MLP head with BN + dropout.
  resnet50v2     : timm resnetv2_50, ImageNet.
  efficientnet_b3: timm efficientnet_b3, ImageNet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# indices into YOLOv8 DetectionModel.model that we tap for multi-scale features
_YOLO_TAPS = (4, 6, 9)              # P3/8 (256), P4/16 (512), P5/32-SPPF (512)
_YOLO_BACKBONE_END = 10             # layers [0, 10) form the backbone
_YOLO_UNFREEZE = {0: [], 1: [7, 8, 9], 2: [5, 6, 7, 8, 9]}


# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
class YOLOBackboneClassifier(nn.Module):
    FEATURE_DIM = 256 + 512 + 512   # 1280

    def __init__(self, dropout: float = 0.3, hidden: int = 256, pretrain: str = "caries"):
        super().__init__()
        from ultralytics import YOLO

        if pretrain == "caries":
            det = YOLO(str(C.DETECTOR_WEIGHTS)).model        # COCO -> adult caries detector
        elif pretrain == "coco":
            det = YOLO(str(C.COCO_WEIGHTS)).model            # COCO only
        elif pretrain == "scratch":
            det = YOLO("yolov8l.yaml").model                 # random init
        else:
            raise ValueError(f"unknown pretrain '{pretrain}'")
        full = det.model                                     # nn.Sequential (23 layers)
        self.backbone = nn.Sequential(*[full[i] for i in range(_YOLO_BACKBONE_END)])
        self.head = MLPHead(self.FEATURE_DIM, hidden, dropout)
        self.set_unfreeze(0)

    # -- features -------------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        h = x
        for i, layer in enumerate(self.backbone):
            h = layer(h)
            if i in _YOLO_TAPS:
                feats.append(torch.flatten(nn.functional.adaptive_avg_pool2d(h, 1), 1))
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    # -- progressive unfreezing --------------------------------------------------
    def set_unfreeze(self, level: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for idx in _YOLO_UNFREEZE.get(level, []):
            for p in self.backbone[idx].parameters():
                p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

    def backbone_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())


# --------------------------------------------------------------------------- #
class TimmClassifier(nn.Module):
    """timm backbone (ImageNet) + dropout + linear head, with staged unfreezing."""

    def __init__(self, timm_name: str, dropout: float = 0.3, hidden: int = 256):
        super().__init__()
        import timm

        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = MLPHead(feat_dim, hidden, dropout)
        self._blocks = self._ordered_blocks()
        self.set_unfreeze(0)

    def _ordered_blocks(self) -> list[nn.Module]:
        """Return coarse stage modules, shallow -> deep, for progressive unfreezing."""
        bb = self.backbone
        if hasattr(bb, "stages"):                 # resnetv2 / convnext style
            return list(bb.stages)
        if hasattr(bb, "blocks"):                 # efficientnet style
            return list(bb.blocks)
        # fallback: every immediate child
        return [m for m in bb.children() if sum(p.numel() for p in m.parameters()) > 0]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def set_unfreeze(self, level: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n = len(self._blocks)
        # level 1 -> last third of blocks; level 2 -> last two thirds
        frac = {0: 0, 1: max(1, n // 3), 2: max(1, 2 * n // 3)}.get(level, 0)
        for blk in self._blocks[n - frac:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        # always keep norm layers after the last unfrozen block trainable-safe:
        for p in self.head.parameters():
            p.requires_grad_(True)

    def backbone_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())


# --------------------------------------------------------------------------- #
_YOLO_PRETRAIN = {"yolo_tl": "caries", "yolo_coco": "coco", "yolo_scratch": "scratch"}


def build_model(name: str, dropout: float, hidden: int) -> nn.Module:
    if name in _YOLO_PRETRAIN:
        return YOLOBackboneClassifier(dropout=dropout, hidden=hidden, pretrain=_YOLO_PRETRAIN[name])
    if name in C.TIMM_NAME:
        return TimmClassifier(C.TIMM_NAME[name], dropout=dropout, hidden=hidden)
    raise ValueError(f"unknown model '{name}'")


if __name__ == "__main__":
    for m in C.MODELS:
        net = build_model(m, dropout=0.3, hidden=256).eval()
        with torch.no_grad():
            out = net(torch.zeros(2, 3, C.MODEL_INPUT, C.MODEL_INPUT))
        tp = sum(p.numel() for p in net.parameters())
        for lvl in (0, 1, 2):
            (net.set_unfreeze(lvl))
            tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"{m:16s} out={tuple(out.shape)} params={tp/1e6:.1f}M  L{lvl} trainable={tr/1e6:.2f}M")
