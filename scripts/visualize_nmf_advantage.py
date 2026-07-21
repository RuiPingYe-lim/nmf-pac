#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF advantage visualization (without retraining).

Goal:
- Build a direct comparison figure: w/o NMF vs w/ NMF
- Reuse existing checkpoint, model, data, prototype and pseudo-label logic
- No training / no algorithm invention
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.decomposition._nmf import non_negative_factorization

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_net import build_custom_model
from data import NPYSliceDataset, NPYInferDataset
from uda_core.prototypes import PrototypeBank
from uda_core.thresholds import ClasswiseEMAThreshold

PAPER_STYLE = {
    "base_fontsize": 10.5,
    "axes_labelsize": 10.5,
    "panel_title_size": 11,
    "panel_title_pad": 8,
    "panel_title_linespacing": 1.15,
    "tick_labelsize": 9,
    "legend_fontsize": 9,
    "annotation_size": 9,
    "small_annotation_size": 8,
    "colorbar_tick_size": 8,
    "suptitle_size": 13,
    "suptitle_linespacing": 1.15,
    "subplot_wspace": 0.24,
    "subplot_hspace": 0.28,
    "tight_h_pad": 1.1,
    "tight_w_pad": 1.0,
    "raw_overlay_alpha": 0.72,
    "raw_overlay_cmap": "jet",
    "evidence_overlay_alpha": 0.78,
    "evidence_overlay_cmap": "jet",
    "diff_overlay_alpha": 0.14,
    "signed_abs_percentile": 98.0,
    "signed_zero_guard": 1e-4,
    "signed_alpha_gamma": 1.35,
    "signed_saturation": 0.72,
    "signed_neutral_gray": 0.86,
    "colorbar_fraction": 0.014,
    "colorbar_pad": 0.008,
    "focus_top_percent": 12.0,
    "focus_low_percentile": 70.0,
    "focus_high_percentile": 99.0,
    "focus_gamma": 1.35,
    "focus_eps": 1e-6,
    "overlay_base_tint": (0.08, 0.16, 0.50),  # kept for backward compatibility (unused)
    "overlay_base_tint_mix": 0.0,
    "overlay_min_alpha_ratio": 0.0,
    "attn_clip_low_q": 15.0,
    "attn_clip_high_q": 99.5,
    "attn_gamma": 1.7,
    "attn_alpha_max": 0.90,
    "attn_overlay_threshold": 0.22,
    "attn_alpha_gamma": 1.2,
    "attn_anatomy_low_q": 8.0,
}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": PAPER_STYLE["base_fontsize"],
            "axes.labelsize": PAPER_STYLE["axes_labelsize"],
            "axes.titlesize": PAPER_STYLE["panel_title_size"],
            "xtick.labelsize": PAPER_STYLE["tick_labelsize"],
            "ytick.labelsize": PAPER_STYLE["tick_labelsize"],
            "legend.fontsize": PAPER_STYLE["legend_fontsize"],
        }
    )


def _set_panel_title(ax, text: str) -> None:
    ax.set_title(
        text,
        fontsize=PAPER_STYLE["panel_title_size"],
        pad=PAPER_STYLE["panel_title_pad"],
        linespacing=PAPER_STYLE["panel_title_linespacing"],
    )


def _normalize_map_robust(m: np.ndarray, lo_q: float, hi_q: float, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(m, dtype=np.float32)
    lo = float(np.percentile(a, lo_q))
    hi = float(np.percentile(a, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def focus_positive_map(
    hm: np.ndarray,
    top_percent: float = 15.0,
    lo_q: float = 70.0,
    hi_q: float = 99.0,
    gamma: float = 1.35,
    eps: float = 1e-6,
) -> np.ndarray:
    base = _normalize_map_robust(hm, lo_q=lo_q, hi_q=hi_q, eps=eps)
    if base.size == 0:
        return base
    thr = float(np.percentile(base, max(0.0, 100.0 - float(top_percent))))
    sparse = np.where(base >= thr, base, 0.0).astype(np.float32)
    peak = sparse.max()
    if peak > eps:
        sparse = sparse / peak
    sparse = np.power(np.clip(sparse, 0.0, 1.0), float(gamma)).astype(np.float32)
    return sparse


def focus_signed_map(
    hm: np.ndarray,
    q_abs: float = 98.0,
    top_percent: float = 15.0,
    eps: float = 1e-6,
) -> np.ndarray:
    a = np.asarray(hm, dtype=np.float32)
    if a.size == 0:
        return a
    v = float(np.percentile(np.abs(a), q_abs))
    if (not np.isfinite(v)) or v < eps:
        return np.zeros_like(a, dtype=np.float32)
    s = np.clip(np.abs(a) / v, 0.0, 1.0).astype(np.float32)
    thr = float(np.percentile(s, max(0.0, 100.0 - float(top_percent))))
    s = np.where(s >= thr, s, 0.0).astype(np.float32)
    if s.max() > eps:
        s = s / s.max()
    return (np.sign(a) * s).astype(np.float32)


def map_focus_metrics(hm: Optional[np.ndarray]) -> Dict[str, float]:
    if hm is None:
        return {
            "peakiness": 0.0,
            "sparsity": 0.0,
            "compactness": 0.0,
            "area_ratio": 1.0,
            "focus_score": 0.0,
        }
    a = np.asarray(hm, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return {
            "peakiness": 0.0,
            "sparsity": 0.0,
            "compactness": 0.0,
            "area_ratio": 1.0,
            "focus_score": 0.0,
        }
    x = np.abs(a)
    xn = _normalize_map_robust(
        x,
        lo_q=float(PAPER_STYLE["focus_low_percentile"]),
        hi_q=float(PAPER_STYLE["focus_high_percentile"]),
        eps=float(PAPER_STYLE["focus_eps"]),
    )
    thr = float(np.percentile(xn, max(0.0, 100.0 - float(PAPER_STYLE["focus_top_percent"]))))
    mask = (xn >= thr).astype(np.float32)
    area_ratio = float(mask.mean())
    sparsity = float(1.0 - area_ratio)
    mean_val = float(xn.mean())
    p99 = float(np.percentile(xn, 99.0))
    peakiness = float(p99 / (mean_val + 1e-6))

    ys, xs = np.indices(xn.shape, dtype=np.float32)
    w = xn + 1e-8
    w_sum = float(w.sum())
    cy = float((ys * w).sum() / w_sum)
    cx = float((xs * w).sum() / w_sum)
    dy = (ys - cy) / max(1.0, float(xn.shape[0] - 1))
    dx = (xs - cx) / max(1.0, float(xn.shape[1] - 1))
    spread = float(np.sqrt((w * (dx * dx + dy * dy)).sum() / w_sum))
    compactness = float(1.0 / (1.0 + 10.0 * spread))
    # Heuristic joint-region prior: center of knee slice tends to carry key structures.
    joint_cy = 0.55
    joint_cx = 0.50
    roi_dist = float(np.sqrt((cy / max(1.0, float(xn.shape[0] - 1)) - joint_cy) ** 2 + (cx / max(1.0, float(xn.shape[1] - 1)) - joint_cx) ** 2))
    roi_closeness = float(np.clip(1.0 - roi_dist / 0.55, 0.0, 1.0))

    focus_score = float(
        0.35 * np.clip(peakiness / 8.0, 0.0, 1.0)
        + 0.25 * np.clip(sparsity, 0.0, 1.0)
        + 0.20 * np.clip(compactness, 0.0, 1.0)
        + 0.20 * np.clip(roi_closeness, 0.0, 1.0)
    )
    return {
        "peakiness": peakiness,
        "sparsity": sparsity,
        "compactness": compactness,
        "area_ratio": area_ratio,
        "roi_closeness": roi_closeness,
        "focus_score": focus_score,
    }


def branch_attention_map(rec: CompareRecord, use_nmf: bool, for_target_pseudo: bool = False) -> Optional[np.ndarray]:
    maps = rec.proto_maps_w if use_nmf else rec.proto_maps_wo
    if maps is None or maps.ndim != 3 or maps.shape[0] == 0:
        return None
    if for_target_pseudo:
        cls = int(rec.cls_w if use_nmf else rec.cls_wo)
    else:
        cls = int(rec.y_true) if rec.y_true is not None else int(rec.pred_cls)
    cls = int(np.clip(cls, 0, maps.shape[0] - 1))
    return maps[cls].astype(np.float32)


def normalize_attention_pair(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> (Optional[np.ndarray], Optional[np.ndarray], float):
    if a is None and b is None:
        return None, None, 1.0
    vals = []
    if a is not None:
        vals.append(np.asarray(a, dtype=np.float32).reshape(-1))
    if b is not None:
        vals.append(np.asarray(b, dtype=np.float32).reshape(-1))
    merged = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)
    lo = float(np.percentile(merged, PAPER_STYLE["focus_low_percentile"]))
    hi = float(np.percentile(merged, PAPER_STYLE["focus_high_percentile"]))
    if (not np.isfinite(hi)) or (hi - lo < float(PAPER_STYLE["focus_eps"])):
        hi = lo + 1.0

    def _norm(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None:
            return None
        xn = np.clip((np.asarray(x, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return focus_positive_map(
            xn,
            top_percent=float(PAPER_STYLE["focus_top_percent"]),
            lo_q=0.0,
            hi_q=100.0,
            gamma=float(PAPER_STYLE["focus_gamma"]),
            eps=float(PAPER_STYLE["focus_eps"]),
        )

    a_n = _norm(a)
    b_n = _norm(b)
    vmax = 1.0
    return a_n, b_n, vmax


@dataclass
class CompareRecord:
    domain: str
    case_id: str
    y_true: Optional[int]
    pred_cls: int
    pred_conf: float
    pred_p1: float

    cls_wo: int
    conf_wo: float
    p1_wo: float
    margin_wo: float

    cls_w: int
    conf_w: float
    p1_w: float
    margin_w: float

    tau_wo: Optional[float]
    keep_wo: Optional[bool]
    tau_w: Optional[float]
    keep_w: Optional[bool]

    fmap: Optional[np.ndarray]
    raw_map: Optional[np.ndarray]
    proto_maps_wo: Optional[np.ndarray]
    proto_maps_w: Optional[np.ndarray]

    score: float


def parse_args():
    ap = argparse.ArgumentParser("Visualize NMF advantage (w/o NMF vs w/ NMF)")

    ap.add_argument("--ckpt", type=str,
                    default="outputs/checkpoint.pth")

    ap.add_argument("--src_root", type=str,
                    default="${DATA_ROOT}/MRNet-v1.0/knees_npy")
    ap.add_argument("--src_csv", type=str,
                    default="${DATA_ROOT}/MRNet-v1.0/valid_0.csv")
    ap.add_argument("--tgt_root", type=str,
                    default="${DATA_ROOT}/KneeMRI/knees_npy")
    ap.add_argument("--tgt_csv", type=str,
                    default="${DATA_ROOT}/KneeMRI/test_0.csv")

    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--plane", type=str, default="sagittal", choices=["sagittal", "coronal", "axial"])
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--id_col_src", type=str, default="case_id")
    ap.add_argument("--label_col_src", type=str, default="label")
    ap.add_argument("--id_col_tgt", type=str, default="case_id")
    ap.add_argument("--label_col_tgt", type=str, default="label")

    ap.add_argument("--single_file_case_src", action="store_true", default=True)
    ap.add_argument("--single_file_case_tgt", action="store_true", default=True)
    ap.add_argument("--id_zero_pad_src", type=int, default=0)
    ap.add_argument("--id_zero_pad_tgt", type=int, default=0)

    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")

    # two prototype branches
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--Kmax", type=int, default=1)
    ap.add_argument("--num_components", type=int, default=None,
                    help="Alias for K/Kmax to keep CLI compatible with previous visualization scripts.")
    ap.add_argument("--tau_proto", type=float, default=0.07)
    ap.add_argument("--proto_m", type=float, default=0.97)
    ap.add_argument("--nmf_assign_iters", type=int, default=100)
    ap.add_argument("--beta_loss", type=str, default="frobenius",
                    choices=["frobenius", "kullback-leibler", "itakura-saito"])

    # target pseudo selection thresholds
    ap.add_argument("--ema_m", type=float, default=0.95)

    ap.add_argument("--num_source_cases", type=int, default=2)
    ap.add_argument("--num_target_cases", type=int, default=2)
    ap.add_argument("--src_case_ids", type=str, default="",
                    help="Optional comma-separated source case_ids to force selection.")
    ap.add_argument("--tgt_case_ids", type=str, default="",
                    help="Optional comma-separated target case_ids to force selection.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    if args.num_components is not None:
        args.K = int(args.num_components)
        args.Kmax = int(args.num_components)
    return args


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def parse_case_ids(text: str) -> List[str]:
    if text is None:
        return []
    parts = [s.strip() for s in str(text).split(",")]
    return [s for s in parts if s]


def _extract_xy_from_batch(batch):
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")))
        if x is None or y is None:
            raise ValueError("Cannot extract (x, y) from dict batch")
        return x, y
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"Unsupported batch type for (x, y): {type(batch)}")


def _extract_x_y_cid_from_batch(batch, has_label: bool):
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")) if has_label else None)
        cid = batch.get("case_id", batch.get("cid", batch.get("id")))
        if x is None:
            raise ValueError("Cannot extract x from dict batch")
        if has_label and y is None:
            raise ValueError("Cannot extract y from dict batch")
        return x, y, cid
    if isinstance(batch, (tuple, list)):
        if has_label:
            if len(batch) < 3:
                raise ValueError("Expected (x, y, cid, ...) for labeled batch")
            return batch[0], batch[1], batch[2]
        if len(batch) < 2:
            raise ValueError("Expected (x, cid, ...) for unlabeled batch")
        return batch[0], None, batch[1]
    raise ValueError(f"Unsupported batch type for (x, y, cid): {type(batch)}")


def _cid_to_str(cid):
    if cid is None:
        return None
    if isinstance(cid, (list, tuple)):
        return str(cid[0]) if len(cid) > 0 else None
    if torch.is_tensor(cid):
        if cid.numel() == 0:
            return None
        if cid.numel() == 1:
            return str(cid.detach().cpu().item())
        return str(cid.detach().cpu().flatten()[0].item())
    return str(cid)


class XYOnlyLoader:
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            yield _extract_xy_from_batch(batch)

    def __len__(self):
        return len(self.base_loader)


def tensor_to_img01(x: torch.Tensor) -> np.ndarray:
    t = x.detach().cpu().float().clone()
    t = t * 0.5 + 0.5
    t = t.clamp(0.0, 1.0)
    img = t.permute(1, 2, 0).numpy()
    if img.shape[2] == 3:
        gray = img.mean(axis=2)
    else:
        gray = img[..., 0]
    return norm01(gray)


def overlay_heatmap(base_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.72, cmap: str = "jet") -> np.ndarray:
    base = np.asarray(base_img, dtype=np.float32)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    elif base.ndim == 3 and base.shape[2] == 1:
        base = np.repeat(base, 3, axis=2)
    elif base.ndim == 3 and base.shape[2] >= 3:
        base = base[..., :3]
    else:
        raise ValueError(f"overlay_heatmap: unsupported base image shape={base.shape}")
    if float(base.max()) > 1.0:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)

    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.squeeze(hm)
    if hm.ndim == 3:
        if hm.shape[0] in (1, 3) and hm.shape[0] != hm.shape[-1]:
            hm = hm.mean(axis=0)
        else:
            hm = hm.mean(axis=-1)
    if hm.ndim != 2:
        raise ValueError(f"overlay_heatmap: expected 2D heatmap after squeeze, got shape={hm.shape}")
    # Robust normalization avoids being dominated by outliers.
    lo = float(np.percentile(hm, 2.0))
    hi = float(np.percentile(hm, 98.0))
    if hi - lo < 1e-8:
        hm = np.zeros_like(hm, dtype=np.float32)
    else:
        hm = np.clip((hm - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    h, w = int(base.shape[0]), int(base.shape[1])
    if hm.shape != (h, w):
        ht = torch.from_numpy(hm).view(1, 1, hm.shape[0], hm.shape[1]).float()
        ht = F.interpolate(ht, size=(h, w), mode="bilinear", align_corners=False)
        hm = norm01(ht[0, 0].cpu().numpy().astype(np.float32))

    cm = plt.get_cmap(cmap)
    heat_rgb = cm(np.clip(hm, 0, 1))[..., :3].astype(np.float32)
    # Low responses stay close to transparent so grayscale anatomy is visible.
    alpha_map = (float(alpha) * np.clip(hm, 0.0, 1.0)).astype(np.float32)[..., None]
    return np.clip((1.0 - alpha_map) * base + alpha_map * heat_rgb, 0.0, 1.0)


def overlay_attention_emphasis(base_img: np.ndarray, heat01: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """
    Attention overlay tuned for stronger foreground/background separation:
    - low response: darker/cooler and weaker
    - high response: brighter/warmer and stronger
    """
    base = np.asarray(base_img, dtype=np.float32)
    if base.ndim == 3:
        if base.shape[2] == 1:
            base = base[..., 0]
        else:
            base = base[..., :3].mean(axis=2)
    if base.ndim != 2:
        raise ValueError(f"overlay_attention_emphasis: unsupported base image shape={base.shape}")
    if float(base.max()) > 1.0:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)
    base_rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    h = np.asarray(heat01, dtype=np.float32)
    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"overlay_attention_emphasis: expected 2D heatmap, got shape={h.shape}")
    if h.shape != (base.shape[0], base.shape[1]):
        ht = torch.from_numpy(h).view(1, 1, h.shape[0], h.shape[1]).float()
        ht = F.interpolate(ht, size=(base.shape[0], base.shape[1]), mode="bilinear", align_corners=False)
        h = ht[0, 0].cpu().numpy().astype(np.float32)

    p_low = float(np.percentile(h, PAPER_STYLE["attn_clip_low_q"]))
    p_high = float(np.percentile(h, PAPER_STYLE["attn_clip_high_q"]))
    if (not np.isfinite(p_high)) or (p_high - p_low < float(PAPER_STYLE["focus_eps"])):
        h_norm = np.zeros_like(h, dtype=np.float32)
    else:
        h_norm = np.clip((h - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    h_emph = np.power(h_norm, float(PAPER_STYLE["attn_gamma"])).astype(np.float32)

    # Suppress coloring in low-intensity anatomy/background regions to keep black background.
    a_low = float(np.percentile(base, PAPER_STYLE["attn_anatomy_low_q"]))
    anatomy = np.clip((base - a_low) / max(1e-6, 1.0 - a_low), 0.0, 1.0).astype(np.float32)

    cm = plt.get_cmap(cmap)
    heat_rgb = cm(np.clip(h_emph, 0.0, 1.0))[..., :3].astype(np.float32)
    thr = float(PAPER_STYLE["attn_overlay_threshold"])
    alpha_strength = np.clip((h_emph - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
    alpha_strength = np.power(alpha_strength, float(PAPER_STYLE["attn_alpha_gamma"])).astype(np.float32)
    alpha_strength = alpha_strength * anatomy
    alpha_map = (float(PAPER_STYLE["attn_alpha_max"]) * alpha_strength).astype(np.float32)[..., None]
    return np.clip((1.0 - alpha_map) * base_rgb + alpha_map * heat_rgb, 0.0, 1.0)


def robust_signed_limit(hm: np.ndarray, q: float = 98.0, eps: float = 1e-4) -> float:
    a = np.asarray(hm, dtype=np.float32)
    if a.size == 0:
        return float(eps)
    v = float(np.percentile(np.abs(a), q))
    if not np.isfinite(v) or v < eps:
        return float(eps)
    return v


def _resize_map_like(hm: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"_resize_map_like expects 2D map, got shape={hm.shape}")
    h, w = int(ref_img.shape[0]), int(ref_img.shape[1])
    if hm.shape == (h, w):
        return hm
    t = torch.from_numpy(hm).view(1, 1, hm.shape[0], hm.shape[1]).float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def prototype_evidence_map(
    proto_maps: Optional[np.ndarray],
    positive_class: int,
    negative_class: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Build a discriminative prototype evidence map.
    Positive values support the reference class; negative values support alternatives.
    """
    if proto_maps is None:
        return None
    pm = np.asarray(proto_maps, dtype=np.float32)
    if pm.ndim != 3 or pm.shape[0] < 2:
        return None

    c = int(np.clip(positive_class, 0, pm.shape[0] - 1))
    classes = [i for i in range(pm.shape[0]) if i != c]
    if len(classes) == 0:
        return None

    if negative_class is not None and 0 <= int(negative_class) < pm.shape[0] and int(negative_class) != c:
        neg = pm[int(negative_class)]
    elif pm.shape[0] == 2:
        neg = pm[classes[0]]
    else:
        # Multi-class fallback: contrast against mean evidence of all other classes.
        neg = np.mean(pm[classes], axis=0)

    return (pm[c] - neg).astype(np.float32)


def overlay_signed_heatmap(
    base_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.25,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Overlay signed evidence map on grayscale image.
    Warm color => stronger support for positive/reference class.
    Cool color => support for opposite/other classes.
    """
    base = np.asarray(base_img, dtype=np.float32)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    elif base.ndim == 3 and base.shape[2] == 1:
        base = np.repeat(base, 3, axis=2)
    elif base.ndim == 3 and base.shape[2] >= 3:
        base = base[..., :3]
    else:
        raise ValueError(f"overlay_signed_heatmap: unsupported base image shape={base.shape}")
    if float(base.max()) > 1.0:
        base = base / 255.0
    base = np.clip(base, 0.0, 1.0)
    base = np.clip(0.85 * base + 0.15, 0.0, 1.0)

    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.squeeze(hm)
    if hm.ndim == 3:
        if hm.shape[0] in (1, 3) and hm.shape[0] != hm.shape[-1]:
            hm = hm.mean(axis=0)
        else:
            hm = hm.mean(axis=-1)
    if hm.ndim != 2:
        raise ValueError(f"overlay_signed_heatmap: expected 2D heatmap after squeeze, got shape={hm.shape}")
    hm = _resize_map_like(hm, base)

    if vmin is None or vmax is None:
        v = robust_signed_limit(
            hm,
            q=float(PAPER_STYLE["signed_abs_percentile"]),
            eps=float(PAPER_STYLE["signed_zero_guard"]),
        )
        vmin, vmax = -v, v
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
    cm = plt.get_cmap(cmap)
    heat_rgb = cm(norm(hm))[..., :3].astype(np.float32)
    # Desaturate and pull near-zero evidence toward light gray for a cleaner paper look.
    sat = float(PAPER_STYLE["signed_saturation"])
    gray = heat_rgb.mean(axis=-1, keepdims=True)
    heat_rgb = gray + sat * (heat_rgb - gray)
    # Near-zero evidence should stay almost transparent for a cleaner look.
    v = max(abs(float(vmin)), abs(float(vmax)), 1e-6)
    strength = np.clip(np.abs(hm) / v, 0.0, 1.0)
    neutral = float(PAPER_STYLE["signed_neutral_gray"])
    heat_rgb = (1.0 - strength[..., None]) * neutral + strength[..., None] * heat_rgb
    alpha_map = (float(alpha) * np.power(strength, float(PAPER_STYLE["signed_alpha_gamma"]))).astype(np.float32)[..., None]
    return np.clip((1.0 - alpha_map) * base + alpha_map * heat_rgb, 0.0, 1.0)


def load_model(args, device: torch.device):
    method = args.backbone.replace("custom_", "") if args.backbone.startswith("custom_") else args.backbone
    model = build_custom_model(method=method, num_classes=int(args.num_classes), pretrained=args.pretrained, device=str(device))

    try:
        sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if not isinstance(sd, dict):
        raise RuntimeError("Unsupported checkpoint format")

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Init] loaded ckpt={args.ckpt} | missing={len(getattr(msg, 'missing_keys', []))} unexpected={len(getattr(msg, 'unexpected_keys', []))}")
    model.eval()
    return model


def build_source_loader(args):
    ds = NPYSliceDataset(
        npy_root=args.src_root,
        csv_file=args.src_csv,
        plane=args.plane,
        id_col=args.id_col_src,
        label_col=args.label_col_src,
        resize=args.resize,
        single_file_case=bool(args.single_file_case_src),
        id_zero_pad=int(args.id_zero_pad_src),
        augment=False,
        return_case_id=True,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return ds, dl


def build_target_loader(args):
    df = pd.read_csv(args.tgt_csv, dtype={args.id_col_tgt: str})
    has_label = args.label_col_tgt in df.columns
    if has_label:
        ds = NPYSliceDataset(
            npy_root=args.tgt_root,
            csv_file=args.tgt_csv,
            plane=args.plane,
            id_col=args.id_col_tgt,
            label_col=args.label_col_tgt,
            resize=args.resize,
            single_file_case=bool(args.single_file_case_tgt),
            id_zero_pad=int(args.id_zero_pad_tgt),
            augment=False,
            return_case_id=True,
        )
    else:
        case_ids = df[args.id_col_tgt].astype(str).tolist()
        ds = NPYInferDataset(
            npy_root=args.tgt_root,
            case_ids=case_ids,
            plane=args.plane,
            resize=args.resize,
            single_file_case=bool(args.single_file_case_tgt),
            id_zero_pad=int(args.id_zero_pad_tgt),
        )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return ds, dl, has_label


def build_proto(args, model, src_loader, device: torch.device, init_mode: str) -> PrototypeBank:
    proto = PrototypeBank(
        num_classes=int(args.num_classes),
        feat_dim=int(getattr(model, "feat_dim", 2048)),
        K=int(args.K) if args.K is not None else None,
        Kmax=int(args.Kmax),
        proto_m=float(args.proto_m),
        temp_proto=float(args.tau_proto),
        device=str(device),
    )
    proto.from_source_init(
        model=model,
        dl_src=XYOnlyLoader(src_loader),
        K=int(args.K) if args.K is not None else None,
        Kmax=int(args.Kmax),
        searchK=(args.K is None),
        init_mode=init_mode,
    )
    return proto


def extract_fmap(model, x: torch.Tensor) -> Optional[np.ndarray]:
    if not hasattr(model, "extract_feature_map"):
        return None
    try:
        fmap = model.extract_feature_map(x)
        return fmap[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def classifier_linear_cam_map(
    model,
    fmap: Optional[np.ndarray],
    cls_idx: int,
) -> Optional[np.ndarray]:
    if fmap is None or fmap.ndim != 3:
        return None
    if not hasattr(model, "classifier") or not hasattr(model.classifier, "weight"):
        return None
    w = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    if w.ndim != 2 or w.shape[1] != fmap.shape[0]:
        return None
    c = int(np.clip(cls_idx, 0, w.shape[0] - 1))
    cam = np.tensordot(w[c], fmap, axes=(0, 0)).astype(np.float32)
    return _normalize_map_robust(
        cam,
        lo_q=float(PAPER_STYLE["focus_low_percentile"]),
        hi_q=float(PAPER_STYLE["focus_high_percentile"]),
        eps=float(PAPER_STYLE["focus_eps"]),
    )


def raw_response_map(model, fmap: Optional[np.ndarray], feat_vec: np.ndarray, pred_cls: int) -> Optional[np.ndarray]:
    cam = classifier_linear_cam_map(model, fmap, cls_idx=pred_cls)
    if cam is not None:
        return cam
    if fmap is not None and fmap.ndim == 3:
        return _normalize_map_robust(
            np.mean(np.abs(fmap), axis=0),
            lo_q=float(PAPER_STYLE["focus_low_percentile"]),
            hi_q=float(PAPER_STYLE["focus_high_percentile"]),
            eps=float(PAPER_STYLE["focus_eps"]),
        )
    if feat_vec.ndim == 1 and feat_vec.size > 0:
        vec = np.abs(feat_vec)
        side = int(np.ceil(np.sqrt(vec.size)))
        pad = side * side - vec.size
        if pad > 0:
            vec = np.pad(vec, (0, pad), mode="constant")
        return _normalize_map_robust(
            vec.reshape(side, side),
            lo_q=float(PAPER_STYLE["focus_low_percentile"]),
            hi_q=float(PAPER_STYLE["focus_high_percentile"]),
            eps=float(PAPER_STYLE["focus_eps"]),
        )
    return None


def class_proto_vectors(proto: PrototypeBank, num_classes: int) -> np.ndarray:
    mu = proto.mu.detach().cpu().numpy().astype(np.float32)
    out = []
    for c in range(num_classes):
        s, e = proto.offsets[c]
        vc = mu[s:e].mean(axis=0)
        vc = vc / (np.linalg.norm(vc) + 1e-8)
        out.append(vc.astype(np.float32))
    return np.stack(out, axis=0)


def parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for s in str(text).split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _l2_normalize_np(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / (n + eps)).astype(np.float32)


def extract_domain_features(model, loader, has_label: bool, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    cids: List[str] = []
    with torch.no_grad():
        for batch in loader:
            x, y, cid = _extract_x_y_cid_from_batch(batch, has_label=has_label)
            x = x.to(device, non_blocking=True)
            logits, feat = model.forward_with_feat(x)
            _ = logits
            feats.append(feat[0].detach().cpu().numpy().astype(np.float32))
            cids.append(str(_cid_to_str(cid)))
            if has_label:
                if torch.is_tensor(y):
                    labels.append(int(y.item()))
                else:
                    labels.append(int(y))
    feat_arr = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, int(getattr(model, "feat_dim", 2048))), dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int64) if has_label else None
    return feat_arr, label_arr, cids


def group_source_features_by_class(src_feats: np.ndarray, src_labels: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for c in range(int(num_classes)):
        m = (src_labels == c)
        out[c] = src_feats[m].astype(np.float32) if np.any(m) else np.zeros((0, src_feats.shape[1]), dtype=np.float32)
    return out


def build_class_prototypes_from_source_feats(
    feats_by_cls: Dict[int, np.ndarray],
    num_classes: int,
    method: str,
    nmf_max_iter: int,
) -> np.ndarray:
    feat_dim = None
    for _k, _v in feats_by_cls.items():
        if _v is not None and _v.ndim == 2 and _v.shape[1] > 0:
            feat_dim = int(_v.shape[1])
            break
    if feat_dim is None:
        feat_dim = 2048
    protos: List[np.ndarray] = []
    for c in range(int(num_classes)):
        Xc = feats_by_cls.get(c)
        if Xc is None or Xc.shape[0] == 0:
            v = np.zeros((feat_dim,), dtype=np.float32)
            v[0] = 1.0
            protos.append(v)
            continue
        if method == "mean":
            vc = Xc.mean(axis=0).astype(np.float32)
        elif method == "nmf":
            xmin = Xc.min(axis=0, keepdims=True)
            xmax = Xc.max(axis=0, keepdims=True)
            scale = np.maximum(xmax - xmin, 1e-6)
            Xs = (Xc - xmin) / scale
            try:
                W, H, _ = non_negative_factorization(
                    Xs,
                    n_components=1,
                    init="nndsvd",
                    solver="mu",
                    beta_loss="frobenius",
                    max_iter=int(nmf_max_iter),
                    tol=1e-6,
                    random_state=42,
                    alpha_H=0.0,
                    alpha_W=0.0,
                    l1_ratio=0.0,
                )
                _ = W
            except Exception:
                _, H, _ = non_negative_factorization(
                    Xs,
                    n_components=1,
                    init="random",
                    solver="mu",
                    beta_loss="frobenius",
                    max_iter=int(nmf_max_iter),
                    tol=1e-6,
                    random_state=42,
                    alpha_H=0.0,
                    alpha_W=0.0,
                    l1_ratio=0.0,
                )
            vc = (H * scale + xmin)[0].astype(np.float32)
        else:
            raise ValueError(f"Unknown prototype method={method}")
        protos.append(vc)
    return _l2_normalize_np(np.stack(protos, axis=0).astype(np.float32), axis=1)


def nearest_proto_predictions_and_metrics(
    feats: np.ndarray,
    labels: Optional[np.ndarray],
    protos: np.ndarray,
) -> Dict[str, np.ndarray]:
    if feats.size == 0:
        return {
            "pred": np.zeros((0,), dtype=np.int64),
            "sim": np.zeros((0, protos.shape[0]), dtype=np.float32),
            "class_margin": np.zeros((0,), dtype=np.float32),
            "true_margin": np.zeros((0,), dtype=np.float32),
            "accuracy": np.nan,
            "margin_mean": np.nan,
        }
    fn = _l2_normalize_np(feats.astype(np.float32), axis=1)
    pn = _l2_normalize_np(protos.astype(np.float32), axis=1)
    sim = (fn @ pn.T).astype(np.float32)
    pred = np.argmax(sim, axis=1).astype(np.int64)
    if sim.shape[1] >= 2:
        part = np.partition(sim, kth=sim.shape[1] - 2, axis=1)
        class_margin = (part[:, -1] - part[:, -2]).astype(np.float32)
    else:
        class_margin = np.zeros((sim.shape[0],), dtype=np.float32)
    if labels is not None and labels.shape[0] == sim.shape[0] and sim.shape[1] > 1:
        true_margin = np.zeros((sim.shape[0],), dtype=np.float32)
        for i in range(sim.shape[0]):
            c = int(labels[i])
            s = sim[i]
            if s.shape[0] == 2:
                oth = 1 - c
            else:
                masked = s.copy()
                masked[c] = -1e9
                oth = int(np.argmax(masked))
            true_margin[i] = float(s[c] - s[oth])
        acc = float(np.mean(pred == labels))
        margin_mean = float(np.mean(true_margin))
    else:
        true_margin = class_margin.copy()
        acc = float("nan")
        margin_mean = float(np.mean(class_margin))
    return {
        "pred": pred,
        "sim": sim,
        "class_margin": class_margin,
        "true_margin": true_margin,
        "accuracy": acc,
        "margin_mean": margin_mean,
    }


def _relative_l2_gap(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    da = np.asarray(a, dtype=np.float32)
    db = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(da - db) / (np.linalg.norm(da) + eps))


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den < eps:
        return float("nan")
    return float(np.dot(x, y) / den)


def compute_static_similarity_metrics(
    proto_mean_cls: np.ndarray,
    proto_nmf_cls: np.ndarray,
    tgt_feats: np.ndarray,
    tgt_labels: Optional[np.ndarray],
    tgt_records: List[CompareRecord],
) -> Dict[str, object]:
    proto_mean_cls = _l2_normalize_np(proto_mean_cls.astype(np.float32), axis=1)
    proto_nmf_cls = _l2_normalize_np(proto_nmf_cls.astype(np.float32), axis=1)
    cos_per_class = np.sum(proto_mean_cls * proto_nmf_cls, axis=1).astype(np.float32)
    rel_l2_per_class = np.asarray(
        [_relative_l2_gap(proto_mean_cls[c], proto_nmf_cls[c]) for c in range(proto_mean_cls.shape[0])],
        dtype=np.float32,
    )

    pred_mean = nearest_proto_predictions_and_metrics(tgt_feats, tgt_labels, proto_mean_cls)["pred"]
    pred_nmf = nearest_proto_predictions_and_metrics(tgt_feats, tgt_labels, proto_nmf_cls)["pred"]
    if pred_mean.size > 0 and pred_nmf.size > 0 and pred_mean.size == pred_nmf.size:
        pred_agreement = float(np.mean(pred_mean == pred_nmf))
    else:
        pred_agreement = float("nan")

    corr_vals: List[float] = []
    for r in tgt_records:
        m0 = _evidence_raw_map(r.proto_maps_wo)
        m1 = _evidence_raw_map(r.proto_maps_w)
        if m0 is None or m1 is None:
            continue
        if m0.shape != m1.shape:
            m1 = _resize_map_like(m1, m0)
        cc = _pearson_corr(m0, m1)
        if np.isfinite(cc):
            corr_vals.append(float(cc))
    map_corr = float(np.mean(corr_vals)) if corr_vals else float("nan")

    close_cos = bool(np.nanmean(cos_per_class) >= 0.99)
    close_l2 = bool(np.nanmean(rel_l2_per_class) <= 0.05)
    close_agree = bool((not np.isfinite(pred_agreement)) or pred_agreement >= 0.95)
    close_map = bool((not np.isfinite(map_corr)) or map_corr >= 0.90)
    static_is_close = bool(close_cos and close_l2 and close_agree and close_map)

    return {
        "cosine_per_class": [float(v) for v in cos_per_class.tolist()],
        "relative_l2_gap_per_class": [float(v) for v in rel_l2_per_class.tolist()],
        "target_prediction_agreement": float(pred_agreement),
        "response_map_correlation": float(map_corr),
        "static_similarity_is_close": static_is_close,
    }


def run_source_robustness_analysis(
    args,
    src_feats_by_cls: Dict[int, np.ndarray],
    tgt_feats: np.ndarray,
    tgt_labels: Optional[np.ndarray],
) -> Dict[str, object]:
    ratios = sorted(set(parse_float_list(args.robustness_ratios)))
    if 0.0 not in ratios:
        ratios = [0.0] + ratios
    ratios = [float(max(0.0, r)) for r in ratios]
    num_classes = int(args.num_classes)
    repeats = max(1, int(args.robustness_repeats))

    base_mean = build_class_prototypes_from_source_feats(src_feats_by_cls, num_classes, method="mean", nmf_max_iter=int(args.nmf_max_iter))
    base_nmf = build_class_prototypes_from_source_feats(src_feats_by_cls, num_classes, method="nmf", nmf_max_iter=int(args.nmf_max_iter))

    class_pools: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        others = [src_feats_by_cls[k] for k in range(num_classes) if k != c and src_feats_by_cls.get(k) is not None and src_feats_by_cls[k].shape[0] > 0]
        class_pools[c] = np.concatenate(others, axis=0).astype(np.float32) if others else np.zeros((0, base_mean.shape[1]), dtype=np.float32)

    rows: List[Dict[str, float]] = []
    for ratio in ratios:
        for rep in range(repeats):
            rng = np.random.default_rng(int(args.robustness_seed) + int(round(ratio * 1000)) * 100 + rep)
            contam_feats_by_cls: Dict[int, np.ndarray] = {}
            for c in range(num_classes):
                Xc = src_feats_by_cls[c]
                n = int(Xc.shape[0])
                n_inj = int(round(float(ratio) * n))
                if n_inj > 0 and class_pools[c].shape[0] > 0:
                    idx = rng.choice(class_pools[c].shape[0], size=n_inj, replace=(n_inj > class_pools[c].shape[0]))
                    inj = class_pools[c][idx].astype(np.float32)
                    contam_feats_by_cls[c] = np.concatenate([Xc, inj], axis=0).astype(np.float32)
                else:
                    contam_feats_by_cls[c] = Xc.astype(np.float32)

            p_mean = build_class_prototypes_from_source_feats(contam_feats_by_cls, num_classes, method="mean", nmf_max_iter=int(args.nmf_max_iter))
            p_nmf = build_class_prototypes_from_source_feats(contam_feats_by_cls, num_classes, method="nmf", nmf_max_iter=int(args.nmf_max_iter))

            drift_mean_cls = [_relative_l2_gap(p_mean[c], base_mean[c]) for c in range(num_classes)]
            drift_nmf_cls = [_relative_l2_gap(p_nmf[c], base_nmf[c]) for c in range(num_classes)]

            mean_eval = nearest_proto_predictions_and_metrics(tgt_feats, tgt_labels, p_mean)
            nmf_eval = nearest_proto_predictions_and_metrics(tgt_feats, tgt_labels, p_nmf)
            rows.append(
                {
                    "ratio": float(ratio),
                    "repeat": int(rep),
                    "drift_mean_avg": float(np.mean(drift_mean_cls)),
                    "drift_nmf_avg": float(np.mean(drift_nmf_cls)),
                    "acc_mean": float(mean_eval["accuracy"]),
                    "acc_nmf": float(nmf_eval["accuracy"]),
                    "margin_mean": float(mean_eval["margin_mean"]),
                    "margin_nmf": float(nmf_eval["margin_mean"]),
                    **{f"drift_mean_c{c}": float(drift_mean_cls[c]) for c in range(num_classes)},
                    **{f"drift_nmf_c{c}": float(drift_nmf_cls[c]) for c in range(num_classes)},
                }
            )
    df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, float]] = []
    for ratio in ratios:
        d = df[df["ratio"] == float(ratio)]
        out = {"ratio": float(ratio)}
        for col in ["drift_mean_avg", "drift_nmf_avg", "acc_mean", "acc_nmf", "margin_mean", "margin_nmf"]:
            out[f"{col}_mean"] = float(d[col].mean()) if len(d) else float("nan")
            out[f"{col}_std"] = float(d[col].std(ddof=0)) if len(d) else float("nan")
        for c in range(num_classes):
            for pfx in [f"drift_mean_c{c}", f"drift_nmf_c{c}"]:
                out[f"{pfx}_mean"] = float(d[pfx].mean()) if len(d) else float("nan")
                out[f"{pfx}_std"] = float(d[pfx].std(ddof=0)) if len(d) else float("nan")
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    return {
        "per_run": df,
        "summary": summary,
        "ratios": ratios,
        "base_mean": base_mean,
        "base_nmf": base_nmf,
    }


def draw_source_robustness_figure(out_png: str, out_pdf: str, robust: Dict[str, object], num_classes: int, has_target_label: bool) -> None:
    _apply_paper_style()
    summary: pd.DataFrame = robust["summary"]
    ratios = np.asarray(summary["ratio"].tolist(), dtype=np.float32) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for c in range(int(num_classes)):
        axes[0, 0].plot(ratios, summary[f"drift_mean_c{c}_mean"].to_numpy(dtype=np.float32), alpha=0.45, linewidth=1.2, label=f"class {c}")
    axes[0, 0].plot(ratios, summary["drift_mean_avg_mean"].to_numpy(dtype=np.float32), color="#d95f02", linewidth=2.4, label="mean(avg)")
    axes[0, 0].set_title("Prototype Drift (Mean Prototype)")
    axes[0, 0].set_xlabel("Injected outlier ratio (%)")
    axes[0, 0].set_ylabel("Relative L2 drift vs clean prototype")
    axes[0, 0].grid(alpha=0.3, linestyle="--")
    axes[0, 0].legend(loc="best", fontsize=8)

    for c in range(int(num_classes)):
        axes[0, 1].plot(ratios, summary[f"drift_nmf_c{c}_mean"].to_numpy(dtype=np.float32), alpha=0.45, linewidth=1.2, label=f"class {c}")
    axes[0, 1].plot(ratios, summary["drift_nmf_avg_mean"].to_numpy(dtype=np.float32), color="#1b9e77", linewidth=2.4, label="nmf(avg)")
    axes[0, 1].set_title("Prototype Drift (NMF Prototype, K=1)")
    axes[0, 1].set_xlabel("Injected outlier ratio (%)")
    axes[0, 1].set_ylabel("Relative L2 drift vs clean prototype")
    axes[0, 1].grid(alpha=0.3, linestyle="--")
    axes[0, 1].legend(loc="best", fontsize=8)

    acc_mean = summary["acc_mean_mean"].to_numpy(dtype=np.float32)
    acc_nmf = summary["acc_nmf_mean"].to_numpy(dtype=np.float32)
    acc_mean_std = summary["acc_mean_std"].to_numpy(dtype=np.float32)
    acc_nmf_std = summary["acc_nmf_std"].to_numpy(dtype=np.float32)
    axes[1, 0].plot(ratios, acc_mean, color="#d95f02", linewidth=2.2, marker="o", label="Mean")
    axes[1, 0].plot(ratios, acc_nmf, color="#1b9e77", linewidth=2.2, marker="o", label="NMF")
    axes[1, 0].fill_between(ratios, acc_mean - acc_mean_std, acc_mean + acc_mean_std, color="#d95f02", alpha=0.18)
    axes[1, 0].fill_between(ratios, acc_nmf - acc_nmf_std, acc_nmf + acc_nmf_std, color="#1b9e77", alpha=0.18)
    axes[1, 0].set_title("Target Nearest-Prototype Accuracy")
    axes[1, 0].set_xlabel("Injected outlier ratio (%)")
    axes[1, 0].set_ylabel("Accuracy" if has_target_label else "N/A (no target labels)")
    axes[1, 0].grid(alpha=0.3, linestyle="--")
    axes[1, 0].legend(loc="best")

    mg_mean = summary["margin_mean_mean"].to_numpy(dtype=np.float32)
    mg_nmf = summary["margin_nmf_mean"].to_numpy(dtype=np.float32)
    mg_mean_std = summary["margin_mean_std"].to_numpy(dtype=np.float32)
    mg_nmf_std = summary["margin_nmf_std"].to_numpy(dtype=np.float32)
    axes[1, 1].plot(ratios, mg_mean, color="#d95f02", linewidth=2.2, marker="o", label="Mean")
    axes[1, 1].plot(ratios, mg_nmf, color="#1b9e77", linewidth=2.2, marker="o", label="NMF")
    axes[1, 1].fill_between(ratios, mg_mean - mg_mean_std, mg_mean + mg_mean_std, color="#d95f02", alpha=0.18)
    axes[1, 1].fill_between(ratios, mg_nmf - mg_nmf_std, mg_nmf + mg_nmf_std, color="#1b9e77", alpha=0.18)
    axes[1, 1].set_title("Target Nearest-Prototype Margin")
    axes[1, 1].set_xlabel("Injected outlier ratio (%)")
    axes[1, 1].set_ylabel("Mean margin")
    axes[1, 1].grid(alpha=0.3, linestyle="--")
    axes[1, 1].legend(loc="best")

    fig.suptitle("Source-side Robustness Analysis (Contamination vs Prototype/Target Behavior)", fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _gini_from_weights(w: np.ndarray) -> float:
    x = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    s = float(np.sum(x))
    if s <= 0:
        return 0.0
    x = np.sort(x / s)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / n) - (n + 1.0) / n)


def run_nmf_sample_contribution_analysis(
    src_feats_by_cls: Dict[int, np.ndarray],
    num_classes: int,
    nmf_max_iter: int,
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    curve_data: Dict[int, np.ndarray] = {}
    for c in range(int(num_classes)):
        Xc = src_feats_by_cls[c]
        n = int(Xc.shape[0])
        if n == 0:
            curve_data[c] = np.zeros((0,), dtype=np.float32)
            continue
        xmin = Xc.min(axis=0, keepdims=True)
        xmax = Xc.max(axis=0, keepdims=True)
        scale = np.maximum(xmax - xmin, 1e-6)
        Xs = (Xc - xmin) / scale
        try:
            W, H, _ = non_negative_factorization(
                Xs,
                n_components=1,
                init="nndsvd",
                solver="mu",
                beta_loss="frobenius",
                max_iter=int(nmf_max_iter),
                tol=1e-6,
                random_state=42,
                alpha_H=0.0,
                alpha_W=0.0,
                l1_ratio=0.0,
            )
            _ = H
        except Exception:
            W, _, _ = non_negative_factorization(
                Xs,
                n_components=1,
                init="random",
                solver="mu",
                beta_loss="frobenius",
                max_iter=int(nmf_max_iter),
                tol=1e-6,
                random_state=42,
                alpha_H=0.0,
                alpha_W=0.0,
                l1_ratio=0.0,
            )
        w = np.clip(W[:, 0].astype(np.float64), 0.0, None)
        if float(np.sum(w)) <= 0:
            w = np.ones_like(w, dtype=np.float64)
        w = w / np.sum(w)
        w_sorted = np.sort(w)[::-1].astype(np.float32)
        curve_data[c] = w_sorted
        uniform = float(1.0 / max(1, n))
        ess = float(1.0 / np.sum(np.square(w)))
        gini = float(_gini_from_weights(w))
        summary_rows.append(
            {
                "class": int(c),
                "num_samples": int(n),
                "uniform_weight": uniform,
                "max_weight": float(np.max(w)),
                "min_weight": float(np.min(w)),
                "ess": ess,
                "ess_ratio": float(ess / max(1.0, float(n))),
                "gini": gini,
            }
        )
        for i, wi in enumerate(w_sorted.tolist()):
            rows.append(
                {
                    "class": int(c),
                    "rank": int(i + 1),
                    "weight": float(wi),
                    "uniform_weight": uniform,
                }
            )
    return {
        "weights_long": pd.DataFrame(rows),
        "summary": pd.DataFrame(summary_rows),
        "curves": curve_data,
    }


def draw_nmf_contribution_figure(out_png: str, out_pdf: str, contrib: Dict[str, object], num_classes: int) -> None:
    _apply_paper_style()
    curves: Dict[int, np.ndarray] = contrib["curves"]
    summary: pd.DataFrame = contrib["summary"]
    ncls = int(num_classes)
    fig, axes = plt.subplots(1, ncls, figsize=(6.2 * ncls, 4.8), squeeze=False)
    for c in range(ncls):
        ax = axes[0, c]
        w = curves.get(c, np.zeros((0,), dtype=np.float32))
        if w.size == 0:
            ax.text(0.5, 0.5, f"class {c}\n(no samples)", ha="center", va="center")
            ax.axis("off")
            continue
        x = np.arange(1, w.size + 1, dtype=np.int32)
        row = summary[summary["class"] == c].iloc[0]
        uniform = float(row["uniform_weight"])
        ax.plot(x, w, color="#1b9e77", linewidth=2.0, label="NMF contribution weight")
        ax.axhline(uniform, color="#d95f02", linestyle="--", linewidth=1.5, label="Mean baseline (equal weight)")
        ax.set_title(f"Class {c}: K=1 NMF sample contribution")
        ax.set_xlabel("Sample rank (descending by NMF weight)")
        ax.set_ylabel("Sample weight")
        ax.grid(alpha=0.3, linestyle="--")
        ax.text(
            0.98,
            0.95,
            f"N={int(row['num_samples'])}\nESS={row['ess']:.1f}\nESS/N={row['ess_ratio']:.3f}\nGini={row['gini']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.2"),
        )
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("NMF Sample Contribution Analysis (K=1) vs Mean Equal-weight Baseline", fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def proto_guided_cam_maps(fmap: Optional[np.ndarray], proto_cls_vec: np.ndarray) -> Optional[np.ndarray]:
    if fmap is None or fmap.ndim != 3:
        return None
    d, h, w = fmap.shape
    pix = fmap.reshape(d, -1).T.astype(np.float32)
    pix = pix / (np.linalg.norm(pix, axis=1, keepdims=True) + 1e-8)
    maps = []
    for i in range(proto_cls_vec.shape[0]):
        pos = proto_cls_vec[i]
        neg_idx = [j for j in range(proto_cls_vec.shape[0]) if j != i]
        if len(neg_idx) > 0:
            neg = np.mean(proto_cls_vec[neg_idx], axis=0)
        else:
            neg = np.zeros_like(pos)
        wvec = pos - neg
        wvec = wvec / (np.linalg.norm(wvec) + 1e-8)
        cam = (pix @ wvec).reshape(h, w).astype(np.float32)
        cam = _normalize_map_robust(
            cam,
            lo_q=float(PAPER_STYLE["focus_low_percentile"]),
            hi_q=float(PAPER_STYLE["focus_high_percentile"]),
            eps=float(PAPER_STYLE["focus_eps"]),
        )
        maps.append(cam)
    return np.stack(maps, axis=0).astype(np.float32)


def class_margin(p: np.ndarray) -> float:
    if p.size < 2:
        return 0.0
    order = np.sort(p)
    return float(order[-1] - order[-2])


def true_margin(p: np.ndarray, y_true: Optional[int]) -> float:
    if y_true is None or p.size < 2:
        return class_margin(p)
    c = int(y_true)
    oth = 1 - c if p.size == 2 else int(np.argmax(np.where(np.arange(p.size) == c, -1e9, p)))
    return float(p[c] - p[oth])


def map_compactness(m: Optional[np.ndarray]) -> Optional[float]:
    if m is None:
        return None
    a = np.asarray(m, dtype=np.float32)
    if a.size == 0:
        return None
    v = a.reshape(-1)
    mean_all = float(np.mean(v))
    if mean_all <= 1e-8:
        return 0.0
    k = max(1, int(0.1 * v.size))
    topk = np.partition(v, -k)[-k:]
    return float(np.mean(topk) / (mean_all + 1e-8))


def rec_compactness_pair(rec: CompareRecord) -> (float, float):
    cls_ref = rec.y_true if rec.y_true is not None else rec.pred_cls
    c_wo = float(rec.conf_wo)
    c_w = float(rec.conf_w)
    if rec.proto_maps_wo is not None and rec.proto_maps_w is not None:
        if 0 <= int(cls_ref) < int(rec.proto_maps_wo.shape[0]) and 0 <= int(cls_ref) < int(rec.proto_maps_w.shape[0]):
            v_wo = map_compactness(rec.proto_maps_wo[int(cls_ref)])
            v_w = map_compactness(rec.proto_maps_w[int(cls_ref)])
            if v_wo is not None:
                c_wo = float(v_wo)
            if v_w is not None:
                c_w = float(v_w)
    return c_wo, c_w


def rec_focus_pair(rec: CompareRecord) -> (float, float):
    m_wo = branch_attention_map(rec, use_nmf=False, for_target_pseudo=(rec.domain == "target"))
    m_w = branch_attention_map(rec, use_nmf=True, for_target_pseudo=(rec.domain == "target"))
    if m_wo is None or m_w is None:
        return 0.0, 0.0
    f_wo = float(map_focus_metrics(m_wo)["focus_score"])
    f_w = float(map_focus_metrics(m_w)["focus_score"])
    return f_wo, f_w


def rec_focus_summary(rec: CompareRecord) -> Dict[str, float]:
    f_wo, f_w = rec_focus_pair(rec)
    m_wo = branch_attention_map(rec, use_nmf=False, for_target_pseudo=(rec.domain == "target"))
    m_w = branch_attention_map(rec, use_nmf=True, for_target_pseudo=(rec.domain == "target"))
    met_wo = map_focus_metrics(m_wo)
    met_w = map_focus_metrics(m_w)
    raw_met = map_focus_metrics(rec.raw_map)
    return {
        "focus_wo": f_wo,
        "focus_w": f_w,
        "focus_gain": float(f_w - f_wo),
        "area_ratio_wo": float(met_wo["area_ratio"]),
        "area_ratio_w": float(met_w["area_ratio"]),
        "peakiness_wo": float(met_wo["peakiness"]),
        "peakiness_w": float(met_w["peakiness"]),
        "sparsity_wo": float(met_wo["sparsity"]),
        "sparsity_w": float(met_w["sparsity"]),
        "compactness_att_wo": float(met_wo["compactness"]),
        "compactness_att_w": float(met_w["compactness"]),
        "roi_closeness_wo": float(met_wo["roi_closeness"]),
        "roi_closeness_w": float(met_w["roi_closeness"]),
        "raw_focus": float(raw_met["focus_score"]),
        "raw_area_ratio": float(raw_met["area_ratio"]),
        "heatmap_improvement_score": float(
            0.45 * (f_w - f_wo)
            + 0.25 * (met_wo["area_ratio"] - met_w["area_ratio"])
            + 0.15 * (met_w["peakiness"] - met_wo["peakiness"]) / 8.0
            + 0.15 * (met_w["roi_closeness"] - met_wo["roi_closeness"])
        ),
    }


def source_improvement_score(rec: CompareRecord) -> float:
    compact_wo, compact_w = rec_compactness_pair(rec)
    fs = rec_focus_summary(rec)
    focus_wo, focus_w = fs["focus_wo"], fs["focus_w"]
    margin_gain = float(rec.margin_w - rec.margin_wo)
    compact_gain = float(compact_w - compact_wo)
    focus_gain = float(fs["focus_gain"])
    pred_ok = 1.0 if (rec.y_true is not None and rec.pred_cls == rec.y_true) else 0.0
    cond_margin = 1.0 if rec.margin_w >= rec.margin_wo else 0.0
    cond_comp = 1.0 if compact_w > compact_wo else 0.0
    cond_focus = 1.0 if focus_w > focus_wo else 0.0
    area_bonus = float(max(0.0, fs["area_ratio_wo"] - fs["area_ratio_w"]))
    heat_gain = float(fs["heatmap_improvement_score"])
    return (
        2.0 * pred_ok
        + 1.2 * cond_comp
        + 1.2 * cond_focus
        + 1.0 * cond_margin
        + 1.8 * margin_gain
        + 1.8 * compact_gain
        + 2.2 * focus_gain
        + 1.0 * area_bonus
        + 0.8 * fs["raw_focus"]
        + 1.6 * heat_gain
        + 0.8 * max(0.0, fs["roi_closeness_w"] - fs["roi_closeness_wo"])
    )


def target_improvement_score(rec: CompareRecord) -> float:
    fs = rec_focus_summary(rec)
    boundary = float(1.0 - min(abs(rec.pred_p1 - 0.55) / 0.10, 1.0))
    proto_gain = float((rec.conf_w - rec.conf_wo) + (rec.margin_w - rec.margin_wo))
    focus_gain = float(fs["focus_gain"])
    keep_gain = 1.0 if (bool(rec.keep_w) and not bool(rec.keep_wo)) else 0.0
    cls_flip = 1.0 if rec.cls_w != rec.cls_wo else 0.0
    gt_gain = 0.0
    if rec.y_true is not None:
        wo_ok = int(rec.cls_wo == rec.y_true)
        w_ok = int(rec.cls_w == rec.y_true)
        gt_gain = float(w_ok - wo_ok)
    softmax_dispersion_bonus = float(np.clip(0.35 - fs["raw_focus"], -0.2, 0.35))
    return (
        1.4 * boundary
        + 1.8 * proto_gain
        + 2.2 * focus_gain
        + 1.8 * keep_gain
        + 0.6 * cls_flip
        + 2.0 * gt_gain
        + 0.8 * max(0.0, fs["area_ratio_wo"] - fs["area_ratio_w"])
        + 1.8 * fs["heatmap_improvement_score"]
        + 1.2 * softmax_dispersion_bonus
        + 1.0 * max(0.0, fs["roi_closeness_w"] - fs["roi_closeness_wo"])
    )


def evaluate_records(args, model, proto_wo, proto_w, loader, has_label: bool, device: torch.device, domain: str) -> List[CompareRecord]:
    recs: List[CompareRecord] = []

    vec_wo = class_proto_vectors(proto_wo, int(args.num_classes))
    vec_w = class_proto_vectors(proto_w, int(args.num_classes))

    thr_wo = ClasswiseEMAThreshold(num_classes=int(args.num_classes), ema_lambda=float(args.ema_m))
    thr_w = ClasswiseEMAThreshold(num_classes=int(args.num_classes), ema_lambda=float(args.ema_m))

    with torch.no_grad():
        for batch in loader:
            x, y, cid = _extract_x_y_cid_from_batch(batch, has_label=has_label)
            x = x.to(device, non_blocking=True)
            y_true = int(y.item()) if (has_label and torch.is_tensor(y)) else (int(y) if has_label else None)
            case_id = _cid_to_str(cid)
            if case_id is None:
                continue

            logits, feat = model.forward_with_feat(x)
            prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
            pred_cls = int(np.argmax(prob))
            pred_conf = float(np.max(prob))
            pred_p1 = float(prob[1] if prob.size > 1 else prob[0])

            q_wo, p_wo_t = proto_wo.soft_assign(feat)
            q_w, p_w_t = proto_w.nmf_assign(feat, beta_loss=args.beta_loss, iters=int(args.nmf_assign_iters))

            p_wo = p_wo_t[0].detach().cpu().numpy().astype(np.float32)
            p_w = p_w_t[0].detach().cpu().numpy().astype(np.float32)

            cls_wo = int(np.argmax(p_wo)); conf_wo = float(np.max(p_wo)); p1_wo = float(p_wo[1] if p_wo.size > 1 else p_wo[0])
            cls_w = int(np.argmax(p_w)); conf_w = float(np.max(p_w)); p1_w = float(p_w[1] if p_w.size > 1 else p_w[0])

            m_wo = true_margin(p_wo, y_true)
            m_w = true_margin(p_w, y_true)

            tau_map_wo = thr_wo.update_and_get(p_wo_t).numpy().astype(np.float32)
            tau_map_w = thr_w.update_and_get(p_w_t).numpy().astype(np.float32)
            tau_wo = float(tau_map_wo[cls_wo]); tau_w = float(tau_map_w[cls_w])
            keep_wo = bool(conf_wo > tau_wo)
            keep_w = bool(conf_w > tau_w)

            fmap = extract_fmap(model, x)
            feat_vec = feat[0].detach().cpu().numpy().astype(np.float32)
            rmap = raw_response_map(model, fmap, feat_vec, pred_cls=pred_cls)

            pm_wo = proto_guided_cam_maps(fmap, vec_wo)
            pm_w = proto_guided_cam_maps(fmap, vec_w)

            imp = float(m_w - m_wo)
            score = imp + 0.3 * (conf_w - conf_wo) + 0.2 * (1.0 - abs(pred_p1 - 0.5))

            recs.append(
                CompareRecord(
                    domain=domain,
                    case_id=case_id,
                    y_true=y_true,
                    pred_cls=pred_cls,
                    pred_conf=pred_conf,
                    pred_p1=pred_p1,
                    cls_wo=cls_wo,
                    conf_wo=conf_wo,
                    p1_wo=p1_wo,
                    margin_wo=m_wo,
                    cls_w=cls_w,
                    conf_w=conf_w,
                    p1_w=p1_w,
                    margin_w=m_w,
                    tau_wo=tau_wo,
                    keep_wo=keep_wo,
                    tau_w=tau_w,
                    keep_w=keep_w,
                    fmap=fmap,
                    raw_map=rmap,
                    proto_maps_wo=pm_wo,
                    proto_maps_w=pm_w,
                    score=score,
                )
            )

    return recs


def select_source_adv(records: List[CompareRecord], num_classes: int, n_keep: int) -> List[CompareRecord]:
    if len(records) == 0:
        return []
    selected: List[CompareRecord] = []
    per_class = max(1, n_keep // max(1, num_classes))
    for c in range(num_classes):
        cand = [r for r in records if r.y_true == c and r.pred_cls == c]
        strong = []
        weak = []
        for r in cand:
            compact_wo, compact_w = rec_compactness_pair(r)
            cond = (compact_w > compact_wo) and (r.margin_w >= r.margin_wo)
            (strong if cond else weak).append(r)
        strong.sort(key=source_improvement_score, reverse=True)
        weak.sort(key=source_improvement_score, reverse=True)
        pick = strong[:per_class]
        if len(pick) < per_class:
            pick.extend(weak[: per_class - len(pick)])
        selected.extend(pick)
    if len(selected) < n_keep:
        used = {r.case_id for r in selected}
        remain = [r for r in records if r.case_id not in used]
        remain.sort(key=source_improvement_score, reverse=True)
        selected.extend(remain[: max(0, n_keep - len(selected))])
    return selected[:n_keep]


def select_target_adv(records: List[CompareRecord], n_keep: int) -> List[CompareRecord]:
    if len(records) == 0:
        return []

    ranked = sorted(records, key=target_improvement_score, reverse=True)
    out: List[CompareRecord] = []
    used: Set[str] = set()

    # hard/boundary + improved by NMF
    hard_pool = [r for r in ranked if 0.45 <= r.pred_p1 <= 0.65]
    flip_pool = [r for r in ranked if (not bool(r.keep_wo) and bool(r.keep_w)) or (r.cls_wo != r.cls_w)]
    gt_pool = [r for r in ranked if (r.y_true is not None and r.cls_w == r.y_true and r.cls_wo != r.y_true)]
    easy_pool = [r for r in ranked if abs(r.pred_p1 - 0.5) > 0.2]

    for pool in (gt_pool, hard_pool, flip_pool, easy_pool, ranked):
        for r in pool:
            if r.case_id in used:
                continue
            out.append(r)
            used.add(r.case_id)
            if len(out) >= n_keep:
                return out[:n_keep]
    return out[:n_keep]


def pick_manual(records: List[CompareRecord], case_ids: List[str]) -> List[CompareRecord]:
    if not case_ids:
        return []
    mp = {r.case_id: r for r in records}
    out = []
    for cid in case_ids:
        r = mp.get(str(cid))
        if r is not None:
            out.append(r)
    return out


def get_case_image(loader, case_id: str, has_label: bool) -> Optional[np.ndarray]:
    for batch in loader:
        x, _y, cid = _extract_x_y_cid_from_batch(batch, has_label=has_label)
        cur = _cid_to_str(cid)
        if cur == case_id:
            return tensor_to_img01(x[0])
    return None


def _evidence_map(proto_maps: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # Backward-compatible helper: keep for callers that expect [0,1] response.
    diff = prototype_evidence_map(proto_maps, positive_class=1 if proto_maps is not None and proto_maps.shape[0] > 1 else 0)
    return None if diff is None else norm01(diff)


def _evidence_raw_map(proto_maps: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # Backward-compatible helper: signed evidence for class-1 vs class-0 in binary setup.
    return prototype_evidence_map(proto_maps, positive_class=1 if proto_maps is not None and proto_maps.shape[0] > 1 else 0)


def aggregate_source_metrics(src_recs: List[CompareRecord]) -> Dict[str, float]:
    if not src_recs:
        return {
            "compactness_wo": float("nan"),
            "compactness_w": float("nan"),
            "margin_wo": float("nan"),
            "margin_w": float("nan"),
        }
    compact_wo = float(np.mean([r.conf_wo for r in src_recs]))
    compact_w = float(np.mean([r.conf_w for r in src_recs]))
    margin_wo = float(np.mean([r.margin_wo for r in src_recs]))
    margin_w = float(np.mean([r.margin_w for r in src_recs]))
    return {
        "compactness_wo": compact_wo,
        "compactness_w": compact_w,
        "margin_wo": margin_wo,
        "margin_w": margin_w,
    }


def draw_advantage_figure(out_png: str, out_pdf: str, src_case: CompareRecord, tgt_case: CompareRecord,
                          src_img: np.ndarray, tgt_img: np.ndarray, src_metrics: Dict[str, float]):
    _apply_paper_style()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # ---------- Row 1: Source attention comparison ----------
    axes[0, 0].imshow(src_img, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[0, 0], "Source sample")
    axes[0, 0].axis("off")

    src_wo_raw = branch_attention_map(src_case, use_nmf=False, for_target_pseudo=False)
    src_w_raw = branch_attention_map(src_case, use_nmf=True, for_target_pseudo=False)
    src_wo, src_w, _ = normalize_attention_pair(src_wo_raw, src_w_raw)

    if src_wo is not None:
        axes[0, 1].imshow(
            overlay_heatmap(src_img, src_wo, alpha=float(PAPER_STYLE["evidence_overlay_alpha"]), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
        )
        axes[0, 1].axis("off")
    else:
        axes[0, 1].imshow(src_img, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].axis("off")
    _set_panel_title(axes[0, 1], "Attention w/o NMF")

    if src_w is not None:
        axes[0, 2].imshow(
            overlay_heatmap(src_img, src_w, alpha=float(PAPER_STYLE["evidence_overlay_alpha"]), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
        )
        axes[0, 2].axis("off")
    else:
        axes[0, 2].imshow(src_img, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].axis("off")
    _set_panel_title(axes[0, 2], "Attention w/ NMF")

    fs_src = rec_focus_summary(src_case)
    mvals = [src_metrics["compactness_wo"], src_metrics["compactness_w"], src_metrics["margin_wo"], src_metrics["margin_w"]]
    axes[0, 3].bar([0, 1, 2, 3], mvals, color=["#d95f02", "#1b9e77", "#d95f02", "#1b9e77"])
    axes[0, 3].set_xticks([0, 1, 2, 3])
    axes[0, 3].set_xticklabels(["Comp-wo", "Comp-w", "Mar-wo", "Mar-w"], rotation=18)
    axes[0, 3].grid(axis="y", linestyle="--", alpha=0.3)
    _set_panel_title(axes[0, 3], "Compactness / Margin Summary")
    axes[0, 3].text(
        0.98,
        0.95,
        f"Case={src_case.case_id}\nfocus gain={fs_src['focus_gain']:+.3f}\n"
        f"area wo={fs_src['area_ratio_wo']:.3f}\narea w={fs_src['area_ratio_w']:.3f}",
        transform=axes[0, 3].transAxes,
        ha="right",
        va="top",
        fontsize=PAPER_STYLE["small_annotation_size"],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )

    axes[0, 4].axis("off")
    axes[0, 4].text(
        0.02,
        0.95,
        "Row-1 intent:\nCompare attention compactness\nfor same source case.\n"
        "w/ NMF should be tighter and\nmore stable for class evidence.",
        transform=axes[0, 4].transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_STYLE["annotation_size"],
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )

    src_sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
    src_sm.set_array([])
    cbar_src = fig.colorbar(
        src_sm,
        ax=[axes[0, 1], axes[0, 2]],
        fraction=float(PAPER_STYLE["colorbar_fraction"]),
        pad=float(PAPER_STYLE["colorbar_pad"]),
    )
    cbar_src.ax.tick_params(labelsize=PAPER_STYLE["colorbar_tick_size"])
    cbar_src.set_label("Source attention intensity", fontsize=PAPER_STYLE["small_annotation_size"])

    # ---------- Row 2: Target pseudo-label process ----------
    axes[1, 0].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[1, 0], "Target sample")
    axes[1, 0].axis("off")

    tgt_soft_raw = tgt_case.raw_map
    tgt_wo_raw = branch_attention_map(tgt_case, use_nmf=False, for_target_pseudo=True)
    tgt_w_raw = branch_attention_map(tgt_case, use_nmf=True, for_target_pseudo=True)

    # shared normalization for comparability among softmax / wo / w
    grp = [m for m in [tgt_soft_raw, tgt_wo_raw, tgt_w_raw] if m is not None]
    if len(grp) > 0:
        merged = np.concatenate([np.asarray(m, dtype=np.float32).reshape(-1) for m in grp])
        lo = float(np.percentile(merged, PAPER_STYLE["focus_low_percentile"]))
        hi = float(np.percentile(merged, PAPER_STYLE["focus_high_percentile"]))
        if (not np.isfinite(hi)) or (hi - lo < float(PAPER_STYLE["focus_eps"])):
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0

    def _norm_group_map(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None
        x = np.clip((np.asarray(m, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        return x

    tgt_soft = _norm_group_map(tgt_soft_raw)
    tgt_wo = _norm_group_map(tgt_wo_raw)
    tgt_w = _norm_group_map(tgt_w_raw)

    if tgt_soft is not None:
        axes[1, 1].imshow(
            overlay_heatmap(tgt_img, tgt_soft, alpha=float(PAPER_STYLE["raw_overlay_alpha"]), cmap=str(PAPER_STYLE["raw_overlay_cmap"]))
        )
    else:
        axes[1, 1].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].axis("off")
    _set_panel_title(axes[1, 1], "Softmax-only attention")

    if tgt_wo is not None:
        axes[1, 2].imshow(
            overlay_heatmap(tgt_img, tgt_wo, alpha=float(PAPER_STYLE["evidence_overlay_alpha"]), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
        )
    else:
        axes[1, 2].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].axis("off")
    _set_panel_title(axes[1, 2], "Pseudo-label attention w/o NMF")

    if tgt_w is not None:
        axes[1, 3].imshow(
            overlay_heatmap(tgt_img, tgt_w, alpha=float(PAPER_STYLE["evidence_overlay_alpha"]), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
        )
    else:
        axes[1, 3].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].axis("off")
    _set_panel_title(axes[1, 3], "Pseudo-label attention w/ NMF")

    axes[1, 4].set_xlim(0, 1)
    axes[1, 4].set_ylim(-0.5, 4.5)
    vals = [tgt_case.pred_p1, tgt_case.p1_wo, tgt_case.p1_w, tgt_case.tau_wo or 0.0, tgt_case.tau_w or 0.0]
    labels = ["Softmax", "Pseudo wo", "Pseudo w", "Tau wo", "Tau w"]
    colors = ["#4c72b0", "#d95f02", "#1b9e77", "#c44e52", "#8172b3"]
    for i, (v, c) in enumerate(zip(vals, colors)):
        axes[1, 4].barh([4 - i], [v], color=c)
    axes[1, 4].set_yticks([4, 3, 2, 1, 0])
    axes[1, 4].set_yticklabels(labels)
    axes[1, 4].grid(axis="x", linestyle="--", alpha=0.3)
    fs_tgt = rec_focus_summary(tgt_case)
    axes[1, 4].text(
        0.02,
        -0.25,
        f"Case={tgt_case.case_id}\n"
        f"pred={tgt_case.pred_cls} ({tgt_case.pred_conf:.3f})\n"
        f"wo: cls={tgt_case.cls_wo}, keep={str(tgt_case.keep_wo).lower()}\n"
        f"w : cls={tgt_case.cls_w}, keep={str(tgt_case.keep_w).lower()}\n"
        f"focus gain={fs_tgt['focus_gain']:+.3f}",
        transform=axes[1, 4].transAxes,
        va="top",
        fontsize=PAPER_STYLE["annotation_size"],
    )
    _set_panel_title(axes[1, 4], "Pseudo-label decision summary")

    tgt_sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=str(PAPER_STYLE["evidence_overlay_cmap"]))
    tgt_sm.set_array([])
    cbar_tgt = fig.colorbar(
        tgt_sm,
        ax=[axes[1, 1], axes[1, 2], axes[1, 3]],
        fraction=float(PAPER_STYLE["colorbar_fraction"]),
        pad=float(PAPER_STYLE["colorbar_pad"]),
    )
    cbar_tgt.ax.tick_params(labelsize=PAPER_STYLE["colorbar_tick_size"])
    cbar_tgt.set_label("Target attention intensity", fontsize=PAPER_STYLE["small_annotation_size"])

    fig.suptitle(
        "NMF Improves Attention Compactness and Pseudo-label Stability\n"
        "Row-1: source attention comparison | Row-2: target pseudo-label process comparison.",
        fontsize=PAPER_STYLE["suptitle_size"],
        linespacing=PAPER_STYLE["suptitle_linespacing"],
    )
    fig.subplots_adjust(wspace=PAPER_STYLE["subplot_wspace"], hspace=PAPER_STYLE["subplot_hspace"])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=PAPER_STYLE["tight_h_pad"], w_pad=PAPER_STYLE["tight_w_pad"])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_pseudo_label_1x4(out_png: str, out_pdf: str, tgt_case: CompareRecord, tgt_img: np.ndarray):
    _apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: target sample
    axes[0].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[0], "Target sample")
    axes[0].axis("off")

    # Build comparable attention maps for panels 2/3
    tgt_wo_raw = branch_attention_map(tgt_case, use_nmf=False, for_target_pseudo=True)
    tgt_w_raw = branch_attention_map(tgt_case, use_nmf=True, for_target_pseudo=True)

    grp = [m for m in [tgt_wo_raw, tgt_w_raw] if m is not None]
    if len(grp) > 0:
        merged = np.concatenate([np.asarray(m, dtype=np.float32).reshape(-1) for m in grp])
        lo = float(np.percentile(merged, PAPER_STYLE["focus_low_percentile"]))
        hi = float(np.percentile(merged, PAPER_STYLE["focus_high_percentile"]))
        if (not np.isfinite(hi)) or (hi - lo < float(PAPER_STYLE["focus_eps"])):
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0

    def _norm_group_map(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None
        x = np.clip((np.asarray(m, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return focus_positive_map(
            x,
            top_percent=float(PAPER_STYLE["focus_top_percent"]),
            lo_q=0.0,
            hi_q=100.0,
            gamma=float(PAPER_STYLE["focus_gamma"]),
            eps=float(PAPER_STYLE["focus_eps"]),
        )

    tgt_wo = _norm_group_map(tgt_wo_raw)
    tgt_w = _norm_group_map(tgt_w_raw)

    # Panel 2: pseudo-label attention without NMF
    if tgt_wo is not None:
        axes[1].imshow(
            overlay_attention_emphasis(
                tgt_img,
                tgt_wo,
                cmap=str(PAPER_STYLE["evidence_overlay_cmap"]),
            )
        )
    else:
        axes[1].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[1], "Pseudo-label attention (w/o NMF)")
    axes[1].axis("off")

    # Panel 3: pseudo-label attention with NMF
    if tgt_w is not None:
        axes[2].imshow(
            overlay_attention_emphasis(
                tgt_img,
                tgt_w,
                cmap=str(PAPER_STYLE["evidence_overlay_cmap"]),
            )
        )
    else:
        axes[2].imshow(tgt_img, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[2], "Pseudo-label attention (w/ NMF)")
    axes[2].axis("off")

    # Panel 4: decision summary
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(-0.5, 3.5)
    vals = [tgt_case.p1_wo, tgt_case.p1_w, tgt_case.tau_wo or 0.0, tgt_case.tau_w or 0.0]
    labels = ["Pseudo wo", "Pseudo w", "Tau wo", "Tau w"]
    colors = ["#d95f02", "#1b9e77", "#c44e52", "#8172b3"]
    for i, (v, c) in enumerate(zip(vals, colors)):
        axes[3].barh([3 - i], [v], color=c)
    axes[3].set_yticks([3, 2, 1, 0])
    axes[3].set_yticklabels(labels)
    axes[3].grid(axis="x", linestyle="--", alpha=0.3)
    fs_tgt = rec_focus_summary(tgt_case)
    axes[3].text(
        0.02,
        -0.23,
        f"Case={tgt_case.case_id}\n"
        f"pred={tgt_case.pred_cls} ({tgt_case.pred_conf:.3f})\n"
        f"wo: cls={tgt_case.cls_wo}, keep={str(tgt_case.keep_wo).lower()}\n"
        f"w : cls={tgt_case.cls_w}, keep={str(tgt_case.keep_w).lower()}\n"
        f"focus gain={fs_tgt['focus_gain']:+.3f}",
        transform=axes[3].transAxes,
        va="top",
        fontsize=PAPER_STYLE["annotation_size"],
    )
    _set_panel_title(axes[3], "Pseudo-label decision summary")

    # Shared colorbar for panels 2/3
    sm = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
        cmap=str(PAPER_STYLE["evidence_overlay_cmap"]),
    )
    sm.set_array([])
    fig.subplots_adjust(left=0.035, right=0.89, top=0.88, bottom=0.15, wspace=0.30)
    cax = fig.add_axes([0.902, 0.24, 0.012, 0.54])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=PAPER_STYLE["colorbar_tick_size"])
    cbar.set_label("Pseudo-label attention intensity", fontsize=PAPER_STYLE["small_annotation_size"])

    fig.suptitle(
        "Target-Domain Pseudo-label Selection Process",
        fontsize=PAPER_STYLE["suptitle_size"],
        linespacing=PAPER_STYLE["suptitle_linespacing"],
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_selected_cases(path: str, src_all: List[CompareRecord], tgt_all: List[CompareRecord],
                         src_sel: List[CompareRecord], tgt_sel: List[CompareRecord],
                         used_src: str, used_tgt: str):
    src_sel_set = {r.case_id for r in src_sel}
    tgt_sel_set = {r.case_id for r in tgt_sel}
    rows = []
    for r in src_all:
        compact_wo, compact_w = rec_compactness_pair(r)
        fs = rec_focus_summary(r)
        rows.append({
            "domain": "source",
            "case_id": r.case_id,
            "selected_pool": int(r.case_id in src_sel_set),
            "used_in_figure": int(r.case_id == used_src),
            "case_type": "source_candidate",
            "gt": "" if r.y_true is None else int(r.y_true),
            "pred": int(r.pred_cls),
            "classifier_prob": float(r.pred_p1),
            "pred_conf": float(r.pred_conf),
            "proto_wo": float(r.p1_wo),
            "proto_w": float(r.p1_w),
            "margin_wo": float(r.margin_wo),
            "margin_w": float(r.margin_w),
            "keep_wo": int(bool(r.keep_wo)),
            "keep_w": int(bool(r.keep_w)),
            "tau_wo": float(r.tau_wo) if r.tau_wo is not None else "",
            "tau_w": float(r.tau_w) if r.tau_w is not None else "",
            "compactness_wo": float(compact_wo),
            "compactness_w": float(compact_w),
            "focus_wo": float(fs["focus_wo"]),
            "focus_w": float(fs["focus_w"]),
            "focus_gain": float(fs["focus_gain"]),
            "focus_area_ratio_wo": float(fs["area_ratio_wo"]),
            "focus_area_ratio_w": float(fs["area_ratio_w"]),
            "focus_peakiness_wo": float(fs["peakiness_wo"]),
            "focus_peakiness_w": float(fs["peakiness_w"]),
            "focus_sparsity_wo": float(fs["sparsity_wo"]),
            "focus_sparsity_w": float(fs["sparsity_w"]),
            "focus_compactness_wo": float(fs["compactness_att_wo"]),
            "focus_compactness_w": float(fs["compactness_att_w"]),
            "focus_roi_closeness_wo": float(fs["roi_closeness_wo"]),
            "focus_roi_closeness_w": float(fs["roi_closeness_w"]),
            "heatmap_improvement_score": float(fs["heatmap_improvement_score"]),
            "raw_focus": float(fs["raw_focus"]),
            "raw_area_ratio": float(fs["raw_area_ratio"]),
            "improvement_score": float(source_improvement_score(r)),
        })
    for r in tgt_all:
        compact_wo, compact_w = rec_compactness_pair(r)
        fs = rec_focus_summary(r)
        rows.append({
            "domain": "target",
            "case_id": r.case_id,
            "selected_pool": int(r.case_id in tgt_sel_set),
            "used_in_figure": int(r.case_id == used_tgt),
            "case_type": "target_candidate",
            "gt": "" if r.y_true is None else int(r.y_true),
            "pred": int(r.pred_cls),
            "classifier_prob": float(r.pred_p1),
            "pred_conf": float(r.pred_conf),
            "proto_wo": float(r.p1_wo),
            "proto_w": float(r.p1_w),
            "margin_wo": float(r.margin_wo),
            "margin_w": float(r.margin_w),
            "keep_wo": int(bool(r.keep_wo)),
            "keep_w": int(bool(r.keep_w)),
            "tau_wo": float(r.tau_wo) if r.tau_wo is not None else "",
            "tau_w": float(r.tau_w) if r.tau_w is not None else "",
            "compactness_wo": float(compact_wo),
            "compactness_w": float(compact_w),
            "focus_wo": float(fs["focus_wo"]),
            "focus_w": float(fs["focus_w"]),
            "focus_gain": float(fs["focus_gain"]),
            "focus_area_ratio_wo": float(fs["area_ratio_wo"]),
            "focus_area_ratio_w": float(fs["area_ratio_w"]),
            "focus_peakiness_wo": float(fs["peakiness_wo"]),
            "focus_peakiness_w": float(fs["peakiness_w"]),
            "focus_sparsity_wo": float(fs["sparsity_wo"]),
            "focus_sparsity_w": float(fs["sparsity_w"]),
            "focus_compactness_wo": float(fs["compactness_att_wo"]),
            "focus_compactness_w": float(fs["compactness_att_w"]),
            "focus_roi_closeness_wo": float(fs["roi_closeness_wo"]),
            "focus_roi_closeness_w": float(fs["roi_closeness_w"]),
            "heatmap_improvement_score": float(fs["heatmap_improvement_score"]),
            "raw_focus": float(fs["raw_focus"]),
            "raw_area_ratio": float(fs["raw_area_ratio"]),
            "improvement_score": float(target_improvement_score(r)),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "domain", "case_id", "selected_pool", "used_in_figure", "case_type",
            "gt", "pred", "classifier_prob", "pred_conf", "proto_wo", "proto_w",
            "tau_wo", "tau_w", "keep_wo", "keep_w",
            "compactness_wo", "compactness_w", "focus_wo", "focus_w", "focus_gain",
            "focus_area_ratio_wo", "focus_area_ratio_w", "raw_focus",
            "focus_peakiness_wo", "focus_peakiness_w", "focus_sparsity_wo", "focus_sparsity_w",
            "focus_compactness_wo", "focus_compactness_w",
            "focus_roi_closeness_wo", "focus_roi_closeness_w",
            "heatmap_improvement_score", "raw_area_ratio",
            "margin_wo", "margin_w", "improvement_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: str, args, src_sel: List[CompareRecord], tgt_sel: List[CompareRecord],
                used_src: CompareRecord, used_tgt: CompareRecord, src_metrics: Dict[str, float]):
    lines = []
    lines.append("# NMF Advantage Notes")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Checkpoint: `{args.ckpt}`")
    lines.append(f"- Source: `{args.src_csv}`")
    lines.append(f"- Target: `{args.tgt_csv}`")
    lines.append("- No retraining. Inference-only comparison.")
    lines.append("- w/o NMF branch: `kmeans init + soft_assign`.")
    lines.append("- w/ NMF branch: `nmf init + nmf_assign`.")
    lines.append("")
    lines.append("## Selected Cases")
    lines.append(f"- Source used in figure: `{used_src.case_id}`")
    lines.append(f"- Target used in figure: `{used_tgt.case_id}`")
    lines.append(f"- Source pool selected: {', '.join([r.case_id for r in src_sel])}")
    lines.append(f"- Target pool selected (improvement-ranked): {', '.join([r.case_id for r in tgt_sel])}")
    lines.append("")
    lines.append("## Why these cases show NMF advantage")
    lines.append(f"- Source case margin gain (w - wo): `{used_src.margin_w - used_src.margin_wo:.4f}`")
    lines.append(f"- Target case margin gain (w - wo): `{used_tgt.margin_w - used_tgt.margin_wo:.4f}`")
    lines.append(f"- Target pseudo keep comparison: `wo={str(used_tgt.keep_wo).lower()}, w={str(used_tgt.keep_w).lower()}`")
    lines.append("")
    lines.append("## Aggregate evidence on source")
    lines.append(f"- Compactness (mean max class score): wo={src_metrics['compactness_wo']:.4f}, w={src_metrics['compactness_w']:.4f}")
    lines.append(f"- Margin (mean class margin): wo={src_metrics['margin_wo']:.4f}, w={src_metrics['margin_w']:.4f}")
    lines.append("")
    lines.append("## Fallback policy")
    lines.append("- If spatial prototype maps are unavailable, the script falls back to score-bar evidence.")
    lines.append("- Attention maps are generated from the same last spatial feature layer for w/o and w/ NMF.")
    lines.append("- Row-1 uses prototype-guided attention comparison; Row-2 uses softmax-only vs pseudo-label attention.")
    lines.append("- Focus metrics include compactness, sparsity, peakiness, top-k area ratio, and ROI closeness.")
    lines.append("- No synthetic/hand-crafted heatmap is used.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    set_seed(int(args.seed))

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[Info] device={device}")

    model = load_model(args, device)

    _, src_loader = build_source_loader(args)
    _, tgt_loader, tgt_has_label = build_target_loader(args)

    proto_wo = build_proto(args, model, src_loader, device, init_mode="kmeans")
    proto_w = build_proto(args, model, src_loader, device, init_mode="nmf")

    print("[Step] evaluating source samples...")
    src_records = evaluate_records(args, model, proto_wo, proto_w, src_loader, has_label=True, device=device, domain="source")

    print("[Step] evaluating target samples...")
    tgt_records = evaluate_records(args, model, proto_wo, proto_w, tgt_loader, has_label=tgt_has_label, device=device, domain="target")

    src_case_ids = parse_case_ids(args.src_case_ids)
    tgt_case_ids = parse_case_ids(args.tgt_case_ids)

    src_sel = pick_manual(src_records, src_case_ids) if src_case_ids else select_source_adv(
        src_records, num_classes=int(args.num_classes), n_keep=int(args.num_source_cases)
    )
    if len(src_sel) < int(args.num_source_cases):
        used = {r.case_id for r in src_sel}
        remain = [r for r in select_source_adv(src_records, num_classes=int(args.num_classes), n_keep=len(src_records)) if r.case_id not in used]
        src_sel.extend(remain[: max(0, int(args.num_source_cases) - len(src_sel))])

    tgt_sel = pick_manual(tgt_records, tgt_case_ids) if tgt_case_ids else select_target_adv(
        tgt_records, n_keep=max(2, int(args.num_target_cases))
    )
    if len(tgt_sel) < max(2, int(args.num_target_cases)):
        used = {r.case_id for r in tgt_sel}
        remain = [r for r in select_target_adv(tgt_records, n_keep=len(tgt_records)) if r.case_id not in used]
        tgt_sel.extend(remain[: max(0, max(2, int(args.num_target_cases)) - len(tgt_sel))])

    if not src_sel:
        raise RuntimeError("No source cases selected")
    if not tgt_sel:
        raise RuntimeError("No target cases selected")

    src_case = sorted(src_sel, key=source_improvement_score, reverse=True)[0]
    tgt_case = sorted(tgt_sel, key=target_improvement_score, reverse=True)[0]

    tgt_img = get_case_image(tgt_loader, tgt_case.case_id, has_label=tgt_has_label)
    if tgt_img is None:
        raise RuntimeError(f"Cannot fetch target image for case_id={tgt_case.case_id}")

    src_metrics = aggregate_source_metrics(src_records)

    out_png = os.path.join(args.outdir, "nmf_advantage_figure.png")
    out_pdf = os.path.join(args.outdir, "nmf_advantage_figure.pdf")
    out_csv = os.path.join(args.outdir, "selected_cases.csv")
    out_notes = os.path.join(args.outdir, "nmf_advantage_notes.md")
    out_meta = os.path.join(args.outdir, "nmf_advantage_meta.json")

    draw_pseudo_label_1x4(out_png, out_pdf, tgt_case, tgt_img)
    write_selected_cases(out_csv, src_records, tgt_records, src_sel, tgt_sel, used_src=src_case.case_id, used_tgt=tgt_case.case_id)
    write_notes(out_notes, args, src_sel, tgt_sel, src_case, tgt_case, src_metrics)

    meta = {
        "ckpt": args.ckpt,
        "target_has_label": bool(tgt_has_label),
        "source_cases_selected": [r.case_id for r in src_sel],
        "target_cases_selected": [r.case_id for r in tgt_sel],
        "source_case_ids_manual": src_case_ids,
        "target_case_ids_manual": tgt_case_ids,
        "source_case_used_in_figure": src_case.case_id,
        "target_case_used_in_figure": tgt_case.case_id,
        "branch_wo": "kmeans init + soft_assign",
        "branch_w": "nmf init + nmf_assign",
        "source_metrics": src_metrics,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] {out_png}")
    print(f"[Done] {out_pdf}")
    print(f"[Done] {out_csv}")
    print(f"[Done] {out_notes}")
    print(f"[Done] {out_meta}")


if __name__ == "__main__":
    main()
