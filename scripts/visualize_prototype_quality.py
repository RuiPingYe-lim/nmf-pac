#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize prototype quality improvement from NMF (inference only, no retraining).

Priority figure layout (1x4):
1) Shared 2D feature distribution + both prototype sets + shift arrows + inset
2) Prototype shift and center-alignment summary
3) Prototype quality metrics summary
4) Prototype similarity heatmap comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from custom_net import build_custom_model
from data import NPYSliceDataset
from uda_core.prototypes import PrototypeBank


@dataclass
class SampleFeat:
    case_id: str
    y: int
    feat: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_case_ids(text: str) -> List[str]:
    if text is None:
        return []
    chunks = []
    for tok in str(text).replace(";", ",").split(","):
        t = tok.strip()
        if t:
            chunks.append(t)
    return chunks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize prototype quality (w/o NMF vs w/ NMF)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--src_root", type=str, required=True)
    ap.add_argument("--src_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--num_cases", type=int, default=300, help="Number of source samples to visualize/evaluate.")
    ap.add_argument("--class_id", type=int, default=None, help="Optional class id to emphasize in outputs.")
    ap.add_argument("--example_case_id", type=str, default=None, help="Optional fixed example case id shown in panel annotation.")
    ap.add_argument("--enable_mode_b", action="store_true", default=False, help="Optional extra figure: raw feature distribution vs NMF-transformed distribution.")

    ap.add_argument("--plane", type=str, default="sagittal", choices=["sagittal", "coronal", "axial"])
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--id_col_src", type=str, default="case_id")
    ap.add_argument("--label_col_src", type=str, default="label")
    ap.add_argument("--single_file_case_src", action="store_true", default=True)
    ap.add_argument("--id_zero_pad_src", type=int, default=0)

    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")

    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--Kmax", type=int, default=1)
    ap.add_argument("--tau_proto", type=float, default=0.07)
    ap.add_argument("--proto_m", type=float, default=0.97)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def _extract_x_y_cid_from_batch(batch):
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")))
        cid = batch.get("case_id", batch.get("cid", batch.get("id")))
        if x is None or y is None:
            raise ValueError("Cannot extract x/y from dict batch")
        return x, y, cid

    if isinstance(batch, (tuple, list)):
        if len(batch) < 3:
            raise ValueError("Expected batch with at least (x, y, case_id)")
        return batch[0], batch[1], batch[2]

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def _cid_to_str(cid) -> str:
    if isinstance(cid, (list, tuple)):
        return str(cid[0]) if len(cid) > 0 else ""
    if torch.is_tensor(cid):
        if cid.numel() == 0:
            return ""
        if cid.numel() == 1:
            return str(cid.detach().cpu().item())
        return str(cid.detach().cpu().flatten()[0].item())
    return str(cid)


class XYOnlyLoader:
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            x, y, _ = _extract_x_y_cid_from_batch(batch)
            yield x, y

    def __len__(self):
        return len(self.base_loader)


def build_source_loader(args: argparse.Namespace):
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


def load_model(args: argparse.Namespace, device: torch.device):
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


def build_proto(args: argparse.Namespace, model, src_loader, device: torch.device, init_mode: str) -> PrototypeBank:
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


def class_proto_vectors(proto: PrototypeBank, num_classes: int) -> np.ndarray:
    mu = proto.mu.detach().cpu().numpy().astype(np.float32)  # [sumK, D]
    out = []
    for c in range(num_classes):
        s, e = proto.offsets[c]
        vc = mu[s:e].mean(axis=0)
        vc = vc / (np.linalg.norm(vc) + 1e-8)
        out.append(vc.astype(np.float32))
    return np.stack(out, axis=0)


def extract_source_features(model, src_loader, device: torch.device) -> List[SampleFeat]:
    out: List[SampleFeat] = []
    with torch.no_grad():
        for batch in src_loader:
            x, y, cid = _extract_x_y_cid_from_batch(batch)
            x = x.to(device, non_blocking=True)
            y_int = int(y.item()) if torch.is_tensor(y) else int(y)
            case_id = _cid_to_str(cid)
            _logits, feat = model.forward_with_feat(x)
            f = feat[0].detach().cpu().numpy().astype(np.float32)
            f = f / (np.linalg.norm(f) + 1e-8)
            out.append(SampleFeat(case_id=case_id, y=y_int, feat=f))
    return out


def balanced_subset_indices(labels: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    n = int(labels.shape[0])
    if n_keep <= 0 or n_keep >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    picks: List[int] = []
    quota = max(1, n_keep // max(1, len(classes)))
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        k = min(len(idx), quota)
        picks.extend(rng.choice(idx, size=k, replace=False).tolist())

    if len(picks) < n_keep:
        remain = [i for i in range(n) if i not in set(picks)]
        extra = rng.choice(remain, size=min(len(remain), n_keep - len(picks)), replace=False).tolist()
        picks.extend(extra)

    picks = sorted(set(picks))
    if len(picks) > n_keep:
        picks = picks[:n_keep]
    return np.asarray(picks, dtype=np.int64)


def compute_similarity(feats_norm: np.ndarray, protos_norm: np.ndarray) -> np.ndarray:
    return feats_norm @ protos_norm.T


def _relu(v: float) -> float:
    return float(max(0.0, v))


def compute_metrics(feats_norm: np.ndarray, labels: np.ndarray, protos_norm: np.ndarray) -> Dict[str, float]:
    num_classes = protos_norm.shape[0]
    sim = compute_similarity(feats_norm, protos_norm)
    dists = 1.0 - sim

    own_d = []
    margins = []
    for i in range(feats_norm.shape[0]):
        c = int(labels[i])
        c = int(np.clip(c, 0, num_classes - 1))
        own = float(dists[i, c])
        own_d.append(own)
        others = [j for j in range(num_classes) if j != c]
        if len(others) == 0:
            margins.append(0.0)
        else:
            best_other = float(np.max(sim[i, others]))
            margins.append(float(sim[i, c] - best_other))

    compactness = float(np.mean(own_d)) if len(own_d) > 0 else float("nan")
    if num_classes >= 2:
        sep = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                sep.append(float(1.0 - np.dot(protos_norm[i], protos_norm[j])))
        separation = float(np.mean(sep)) if len(sep) > 0 else float("nan")
    else:
        separation = float("nan")
    margin = float(np.mean(margins)) if len(margins) > 0 else float("nan")

    # Purity: for each prototype, top-k responding samples should belong to that class.
    k = max(1, int(0.1 * feats_norm.shape[0]))
    pur = []
    for c in range(num_classes):
        idx = np.argsort(-sim[:, c])[:k]
        pur.append(float(np.mean((labels[idx] == c).astype(np.float32))))
    purity = float(np.mean(pur)) if len(pur) > 0 else float("nan")

    return {
        "compactness": compactness,   # lower is better
        "separation": separation,     # higher is better
        "margin": margin,             # higher is better
        "purity": purity,             # higher is better
    }


def build_case_quality_table(
    feats_norm: np.ndarray,
    labels: np.ndarray,
    case_ids: List[str],
    proto_wo: np.ndarray,
    proto_w: np.ndarray,
) -> pd.DataFrame:
    sim_wo = compute_similarity(feats_norm, proto_wo)
    sim_w = compute_similarity(feats_norm, proto_w)
    n, cnum = sim_wo.shape

    # Class centers from normalized features: proxy of high-density class region.
    centers = np.zeros((cnum, feats_norm.shape[1]), dtype=np.float32)
    for c in range(cnum):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        vc = feats_norm[idx].mean(axis=0).astype(np.float32)
        vc = vc / (np.linalg.norm(vc) + 1e-8)
        centers[c] = vc
    proto_center_align_wo = np.sum(proto_wo * centers, axis=1)
    proto_center_align_w = np.sum(proto_w * centers, axis=1)

    rows: List[Dict[str, object]] = []
    for i in range(n):
        y = int(np.clip(labels[i], 0, cnum - 1))
        others = [j for j in range(cnum) if j != y]
        if len(others) == 0:
            continue
        pred_wo = int(np.argmax(sim_wo[i]))
        pred_w = int(np.argmax(sim_w[i]))

        true_wo = float(sim_wo[i, y])
        true_w = float(sim_w[i, y])
        max_other_wo = float(np.max(sim_wo[i, others]))
        max_other_w = float(np.max(sim_w[i, others]))
        margin_wo = true_wo - max_other_wo
        margin_w = true_w - max_other_w
        margin_gain = margin_w - margin_wo

        mean_other_wo = float(np.mean(sim_wo[i, others]))
        mean_other_w = float(np.mean(sim_w[i, others]))
        correct_wo = int(pred_wo == y)
        correct_w = int(pred_w == y)
        flip_to_correct = int(correct_wo == 0 and correct_w == 1)
        neg_to_pos = int(margin_wo < 0.0 and margin_w > 0.0)

        # Intra-class closeness gain (higher is better): distance reduction to own prototype.
        compactness_gain = (1.0 - true_wo) - (1.0 - true_w)
        # Inter-class suppression gain (higher is better): strongest wrong-class response decreases.
        inter_class_margin_gain = max_other_wo - max_other_w
        # Prototype-center alignment gain for the sample's class.
        center_align_gain = float(proto_center_align_w[y] - proto_center_align_wo[y])

        # Visualization-oriented contrast gain (higher = easier to interpret in heatmap).
        contrast_wo = true_wo - mean_other_wo
        contrast_w = true_w - mean_other_w
        contrast_gain = contrast_w - contrast_wo

        visualization_score = (
            4.0 * float(flip_to_correct)
            + 2.5 * float(neg_to_pos)
            + 2.0 * _relu(margin_gain)
            + 1.5 * _relu(compactness_gain)
            + 1.5 * _relu(inter_class_margin_gain)
            + 1.0 * _relu(center_align_gain)
            + 1.0 * _relu(contrast_gain)
            - 0.6 * _relu(-margin_w)
        )

        rows.append(
            {
                "sample_index": i,
                "case_id": str(case_ids[i]),
                "y_true": y,
                "pred": pred_w,
                "proto_wo": pred_wo,
                "proto_w": pred_w,
                "margin_wo": margin_wo,
                "margin_w": margin_w,
                "margin_gain": margin_gain,
                "correct_wo": correct_wo,
                "correct_w": correct_w,
                "flip_to_correct": flip_to_correct,
                "neg_to_pos": neg_to_pos,
                "own_sim_wo": true_wo,
                "own_sim_w": true_w,
                "wrong_max_sim_wo": max_other_wo,
                "wrong_max_sim_w": max_other_w,
                "compactness_metric": 1.0 - true_w,
                "compactness_gain": compactness_gain,
                "separation_metric": margin_w,
                "inter_class_margin_gain": inter_class_margin_gain,
                "prototype_center_alignment_score": center_align_gain,
                "proto_center_align_wo": float(proto_center_align_wo[y]),
                "proto_center_align_w": float(proto_center_align_w[y]),
                "contrast_gain": contrast_gain,
                "visualization_score": visualization_score,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(
        by=[
            "flip_to_correct",              # 1) wrong -> correct
            "neg_to_pos",                   # 2) margin negative -> positive
            "margin_gain",                  # 3) larger margin gain
            "inter_class_margin_gain",      # 4) wrong-class suppression
            "visualization_score",          # 5) visually interpretable
            "prototype_center_alignment_score",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


def select_visual_subset(case_df: pd.DataFrame, n_keep: int, forced_case_id: Optional[str]) -> np.ndarray:
    if case_df.empty:
        return np.zeros((0,), dtype=np.int64)
    n = len(case_df)
    if n_keep <= 0 or n_keep >= n:
        return case_df["sample_index"].to_numpy(dtype=np.int64)

    selected: List[int] = []
    classes = sorted(case_df["y_true"].unique().tolist())
    quota = max(1, n_keep // max(1, 2 * len(classes)))
    for cls in classes:
        cls_df = case_df[case_df["y_true"] == cls].sort_values("final_rank", ascending=True)
        selected.extend(cls_df.head(quota)["sample_index"].astype(int).tolist())

    used = set(selected)
    if forced_case_id is not None and str(forced_case_id).strip():
        forced = case_df[case_df["case_id"] == str(forced_case_id)]
        if not forced.empty:
            selected.append(int(forced.iloc[0]["sample_index"]))
            used.add(int(forced.iloc[0]["sample_index"]))

    ranked_all = case_df.sort_values("final_rank", ascending=True)
    for _, row in ranked_all.iterrows():
        si = int(row["sample_index"])
        if si in used:
            continue
        selected.append(si)
        used.add(si)
        if len(selected) >= n_keep:
            break

    # Keep best-ranked examples in final subset.
    picked = case_df[case_df["sample_index"].isin(selected)].sort_values("final_rank", ascending=True)
    if len(picked) > n_keep:
        picked = picked.head(n_keep).copy()
        if forced_case_id is not None and str(forced_case_id).strip():
            forced = case_df[case_df["case_id"] == str(forced_case_id)]
            if not forced.empty:
                forced_idx = int(forced.iloc[0]["sample_index"])
                if forced_idx not in set(picked["sample_index"].astype(int).tolist()):
                    picked = pd.concat([picked.iloc[:-1], forced.iloc[[0]]], axis=0)
                    picked = picked.sort_values("final_rank", ascending=True)
    return picked["sample_index"].to_numpy(dtype=np.int64)


def select_example_case(case_df: pd.DataFrame, example_case_id: Optional[str]) -> Optional[pd.Series]:
    if case_df.empty:
        return None
    if example_case_id is not None and str(example_case_id).strip():
        hit = case_df[case_df["case_id"] == str(example_case_id)]
        if not hit.empty:
            return hit.iloc[0]
    return case_df.sort_values("final_rank", ascending=True).iloc[0]


def select_heatmap_order(case_df: pd.DataFrame, max_rows: int) -> np.ndarray:
    if case_df.empty:
        return np.zeros((0,), dtype=np.int64)
    ranked = case_df.sort_values(["final_rank", "visualization_score"], ascending=[True, False])
    if len(ranked) <= max_rows:
        view = ranked.copy()
    else:
        classes = sorted(ranked["y_true"].unique().tolist())
        quota = max(1, max_rows // max(1, len(classes)))
        picks: List[int] = []
        for cls in classes:
            cls_rows = ranked[ranked["y_true"] == cls].head(quota)
            picks.extend(cls_rows["sample_index"].astype(int).tolist())
        used = set(picks)
        for _, row in ranked.iterrows():
            si = int(row["sample_index"])
            if si in used:
                continue
            picks.append(si)
            used.add(si)
            if len(picks) >= max_rows:
                break
        view = ranked[ranked["sample_index"].isin(picks)]
    view = view.sort_values(["y_true", "final_rank"], ascending=[True, True])
    return view["sample_index"].to_numpy(dtype=np.int64)


def draw_figure(
    out_png: str,
    feats_norm: np.ndarray,
    labels: np.ndarray,
    case_ids: List[str],
    proto_wo: np.ndarray,
    proto_w: np.ndarray,
    metrics_wo: Dict[str, float],
    metrics_w: Dict[str, float],
    case_df: pd.DataFrame,
    example_case_id: Optional[str],
    class_id: Optional[int],
) -> None:
    # Shared 2D embedding for fair visual comparison.
    all_mat = np.concatenate([feats_norm, proto_wo, proto_w], axis=0)
    pca = PCA(n_components=2, random_state=0)
    z_all = pca.fit_transform(all_mat).astype(np.float32)
    n = feats_norm.shape[0]
    c = proto_wo.shape[0]
    z_feat = z_all[:n]
    z_pwo = z_all[n:n + c]
    z_pw = z_all[n + c:n + 2 * c]

    sim_wo = compute_similarity(feats_norm, proto_wo)
    sim_w = compute_similarity(feats_norm, proto_w)

    # Heatmap samples: pick most interpretable rows under ranking, then group by class.
    max_rows = min(140, n)
    order = select_heatmap_order(case_df, max_rows=max_rows)
    if len(order) == 0:
        order = np.arange(max_rows, dtype=np.int64)
    labels_ord = labels[order]
    sim_cat = np.concatenate([sim_wo[order], sim_w[order]], axis=1)  # [N, 2C] (for binary => [N,4])

    # Project class centers into the same 2D plane.
    centers_2d = np.zeros((c, 2), dtype=np.float32)
    for cls in range(c):
        idx = np.where(labels == cls)[0]
        if len(idx) > 0:
            centers_2d[cls] = z_feat[idx].mean(axis=0).astype(np.float32)

    shift_dist = np.linalg.norm(z_pw - z_pwo, axis=1)
    dist_center_wo = np.linalg.norm(z_pwo - centers_2d, axis=1)
    dist_center_w = np.linalg.norm(z_pw - centers_2d, axis=1)
    align_gain = dist_center_wo - dist_center_w  # positive: w/ NMF closer to own center

    # Nearest-class alignment in 2D (higher is better).
    align_wo = []
    align_w = []
    for cls in range(c):
        own_wo = np.linalg.norm(z_pwo[cls] - centers_2d[cls])
        own_w = np.linalg.norm(z_pw[cls] - centers_2d[cls])
        others = [j for j in range(c) if j != cls]
        if len(others) == 0:
            align_wo.append(0.0)
            align_w.append(0.0)
        else:
            best_other_wo = min(np.linalg.norm(z_pwo[cls] - centers_2d[j]) for j in others)
            best_other_w = min(np.linalg.norm(z_pw[cls] - centers_2d[j]) for j in others)
            align_wo.append(float(best_other_wo - own_wo))
            align_w.append(float(best_other_w - own_w))
    align_wo = np.asarray(align_wo, dtype=np.float32)
    align_w = np.asarray(align_w, dtype=np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    cmap_pts = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}
    classes = np.unique(labels)

    # Panel 1: shared feature scatter + both prototypes + shift arrows
    ax = axes[0]
    for cls in classes:
        idx = np.where(labels == cls)[0]
        ax.scatter(z_feat[idx, 0], z_feat[idx, 1], s=11, alpha=0.28, c=cmap_pts.get(int(cls), "#666666"), label=f"class {int(cls)} samples")
    for cls in range(c):
        cc = cmap_pts.get(int(cls), "#000000")
        ax.scatter(centers_2d[cls, 0], centers_2d[cls, 1], s=90, marker="o", facecolors="none", edgecolors=cc, linewidths=1.2, zorder=4)
        ax.scatter(z_pwo[cls, 0], z_pwo[cls, 1], s=220, marker="X", facecolors="none", edgecolors=cc, linewidths=2.0, zorder=5)
        ax.scatter(z_pw[cls, 0], z_pw[cls, 1], s=220, marker="D", c=cc, edgecolors="black", linewidths=0.7, zorder=6)
        dx = z_pw[cls, 0] - z_pwo[cls, 0]
        dy = z_pw[cls, 1] - z_pwo[cls, 1]
        # Draw a thick black underlay + colored arrow for better visibility.
        ax.arrow(
            z_pwo[cls, 0], z_pwo[cls, 1], dx, dy,
            color="black", width=0.0024, head_width=0.065, head_length=0.085,
            length_includes_head=True, alpha=0.90, zorder=7,
        )
        ax.arrow(
            z_pwo[cls, 0], z_pwo[cls, 1], dx, dy,
            color=cc, width=0.0014, head_width=0.050, head_length=0.070,
            length_includes_head=True, alpha=1.0, zorder=8,
        )
        ax.text(
            z_pw[cls, 0], z_pw[cls, 1],
            f" c{cls} |d|={shift_dist[cls]:.3f}",
            fontsize=8.0, fontweight="bold", color=cc, ha="left", va="bottom", zorder=9,
        )
    ax.set_title("Mode A: Shared Features + Prototype Shift (wo -> w)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2, linestyle="--")

    # Inset zoom around prototypes.
    xmin = float(min(np.min(z_pwo[:, 0]), np.min(z_pw[:, 0])))
    xmax = float(max(np.max(z_pwo[:, 0]), np.max(z_pw[:, 0])))
    ymin = float(min(np.min(z_pwo[:, 1]), np.min(z_pw[:, 1])))
    ymax = float(max(np.max(z_pwo[:, 1]), np.max(z_pw[:, 1])))
    padx = max(0.04, 0.25 * (xmax - xmin + 1e-8))
    pady = max(0.04, 0.25 * (ymax - ymin + 1e-8))
    inset = inset_axes(ax, width="43%", height="43%", loc="lower left", borderpad=1.0)
    for cls in classes:
        idx = np.where(labels == cls)[0]
        inset.scatter(z_feat[idx, 0], z_feat[idx, 1], s=6, alpha=0.12, c=cmap_pts.get(int(cls), "#666666"))
    for cls in range(c):
        cc = cmap_pts.get(int(cls), "#000000")
        inset.scatter(z_pwo[cls, 0], z_pwo[cls, 1], s=120, marker="X", facecolors="none", edgecolors=cc, linewidths=1.4, zorder=3)
        inset.scatter(z_pw[cls, 0], z_pw[cls, 1], s=110, marker="D", c=cc, edgecolors="black", linewidths=0.5, zorder=4)
        inset.arrow(
            z_pwo[cls, 0], z_pwo[cls, 1], z_pw[cls, 0] - z_pwo[cls, 0], z_pw[cls, 1] - z_pwo[cls, 1],
            color="black", width=0.0015, head_width=0.038, head_length=0.038, length_includes_head=True, alpha=0.90, zorder=5,
        )
        inset.arrow(
            z_pwo[cls, 0], z_pwo[cls, 1], z_pw[cls, 0] - z_pwo[cls, 0], z_pw[cls, 1] - z_pwo[cls, 1],
            color=cc, width=0.0009, head_width=0.030, head_length=0.030, length_includes_head=True, alpha=1.0, zorder=6,
        )
    inset.set_xlim(xmin - padx, xmax + padx)
    inset.set_ylim(ymin - pady, ymax + pady)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Prototype Zoom", fontsize=8)
    inset.grid(alpha=0.15, linestyle="--")

    # Optional emphasis class
    if class_id is not None and int(class_id) in classes:
        cls = int(class_id)
        idx = np.where(labels == cls)[0]
        axes[0].scatter(z_feat[idx, 0], z_feat[idx, 1], s=16, alpha=0.5, facecolors="none", edgecolors="black", linewidths=0.6)

    # Panel 2: quant summary for prototype movement and center alignment.
    ax = axes[1]
    x_cls = np.arange(c, dtype=np.float32)
    bw = 0.22
    ax.bar(x_cls - bw, shift_dist, bw, label="shift ||p_w-p_wo||", color="#7570b3", alpha=0.9)
    ax.bar(x_cls, dist_center_wo, bw, label="dist(wo, center)", color="#d95f02", alpha=0.85)
    ax.bar(x_cls + bw, dist_center_w, bw, label="dist(w, center)", color="#1b9e77", alpha=0.85)
    ax.set_xticks(x_cls)
    ax.set_xticklabels([f"c{j}" for j in range(c)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Distance in shared 2D space")
    ax.set_title("Prototype Shift & Center Distance")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax_r = ax.twinx()
    ax_r.plot(x_cls, align_wo, marker="o", linestyle="--", color="#e7298a", label="nearest-align wo")
    ax_r.plot(x_cls, align_w, marker="o", linestyle="-", color="#66a61e", label="nearest-align w")
    ax_r.set_ylabel("Nearest-class alignment (higher better)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="best")

    # Compact per-class distance gain text.
    gain_lines = []
    for cls in range(c):
        gain_lines.append(f"c{cls}: center {dist_center_wo[cls]:.3f}->{dist_center_w[cls]:.3f} (gain {align_gain[cls]:+.3f})")
    ax.text(
        0.02, 0.98, "\n".join(gain_lines[:8]),
        transform=ax.transAxes, va="top", ha="left", fontsize=7.4,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="#cccccc", boxstyle="round,pad=0.22"),
    )

    # Panel 3: metrics summary
    ax = axes[2]
    metrics = ["compactness", "separation", "margin", "purity"]
    wo_vals = [metrics_wo[m] for m in metrics]
    w_vals = [metrics_w[m] for m in metrics]
    x = np.arange(len(metrics), dtype=np.float32)
    bw_m = 0.36
    ax.bar(x - bw_m / 2, wo_vals, bw_m, label="w/o NMF", color="#d95f02")
    ax.bar(x + bw_m / 2, w_vals, bw_m, label="w/ NMF", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(["compactness\n(lower)", "separation\n(higher)", "margin\n(higher)", "purity\n(higher)"], rotation=0)
    ax.set_title("Prototype Metric Summary")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")

    ex = select_example_case(case_df, example_case_id=example_case_id)
    if ex is None:
        ex_text = "Example case: n/a"
    else:
        ex_text = (
            f"Example case: {str(ex['case_id'])}\n"
            f"GT class: {int(ex['y_true'])}\n"
            f"proto wo->w: {int(ex['proto_wo'])}->{int(ex['proto_w'])}\n"
            f"margin wo={float(ex['margin_wo']):.3f}\n"
            f"margin w ={float(ex['margin_w']):.3f}\n"
            f"margin gain={float(ex['margin_gain']):+.3f}\n"
            f"rank={int(ex['final_rank'])} | vis={float(ex['visualization_score']):.2f}"
        )
    ax.text(0.02, 0.98, ex_text, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc", boxstyle="round,pad=0.25"))

    # Panel 4: similarity heatmap comparison
    ax = axes[3]
    im = ax.imshow(sim_cat, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title("Prototype Similarity Heatmap\n(w/o vs w/ NMF)")
    xt = []
    for j in range(c):
        xt.append(f"wo:c{j}")
    for j in range(c):
        xt.append(f"w:c{j}")
    ax.set_xticks(np.arange(len(xt)))
    ax.set_xticklabels(xt, rotation=30, ha="right")
    ax.set_ylabel("Samples (grouped by class)")
    ax.axvline(c - 0.5, color="white", linewidth=1.0, alpha=0.9)
    # Class boundary lines
    for cls in classes:
        cls_idx = np.where(labels_ord == cls)[0]
        if len(cls_idx) > 0:
            ax.axhline(cls_idx[-1] + 0.5, color="white", linewidth=0.6, alpha=0.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("cosine similarity")

    # Explicit legend for panel 1 marker semantics.
    sem_handles = [
        plt.Line2D([0], [0], marker="X", color="none", markeredgecolor="black", markerfacecolor="none", markersize=8, linestyle="None", label="prototype w/o NMF"),
        plt.Line2D([0], [0], marker="D", color="black", markeredgecolor="black", markerfacecolor="black", markersize=7, linestyle="None", label="prototype w/ NMF"),
        plt.Line2D([0], [0], marker="o", color="none", markeredgecolor="black", markerfacecolor="none", markersize=7, linestyle="None", label="class center"),
    ]
    fig.legend(sem_handles, [h.get_label() for h in sem_handles], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("NMF Improves Prototype Construction Quality", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_mode_b_distribution(
    out_png: str,
    feats_norm: np.ndarray,
    nmf_repr: np.ndarray,
    labels: np.ndarray,
) -> None:
    # Mode B (optional): compare raw features vs NMF coefficients (assignment activations).
    z_raw = PCA(n_components=2, random_state=0).fit_transform(feats_norm).astype(np.float32)
    z_nmf = PCA(n_components=2, random_state=0).fit_transform(nmf_repr).astype(np.float32)
    classes = np.unique(labels)
    cmap_pts = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, zz, title in [
        (axes[0], z_raw, "Raw Feature Distribution"),
        (axes[1], z_nmf, "NMF-Transformed Distribution (assignment coefficients)"),
    ]:
        for cls in classes:
            idx = np.where(labels == cls)[0]
            ax.scatter(zz[idx, 0], zz[idx, 1], s=12, alpha=0.45, c=cmap_pts.get(int(cls), "#666666"), label=f"class {int(cls)}")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2, linestyle="--")
    handles, labels_lgd = axes[1].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels_lgd, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(path: str, metrics_wo: Dict[str, float], metrics_w: Dict[str, float]) -> None:
    rows = [
        {"metric": "compactness", "wo_nmf": metrics_wo["compactness"], "w_nmf": metrics_w["compactness"], "better": "lower"},
        {"metric": "separation", "wo_nmf": metrics_wo["separation"], "w_nmf": metrics_w["separation"], "better": "higher"},
        {"metric": "margin", "wo_nmf": metrics_wo["margin"], "w_nmf": metrics_w["margin"], "better": "higher"},
        {"metric": "purity", "wo_nmf": metrics_wo["purity"], "w_nmf": metrics_w["purity"], "better": "higher"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "wo_nmf", "w_nmf", "better"])
        writer.writeheader()
        writer.writerows(rows)


def save_candidates_csv(path: str, case_df: pd.DataFrame) -> None:
    if case_df.empty:
        pd.DataFrame(
            columns=[
                "case_id", "y_true", "pred", "proto_wo", "proto_w",
                "margin_wo", "margin_w", "margin_gain", "correct_wo", "correct_w",
                "compactness_metric", "separation_metric", "prototype_center_alignment_score",
                "visualization_score", "final_rank",
            ]
        ).to_csv(path, index=False, encoding="utf-8")
        return
    cols = [
        "case_id",
        "y_true",
        "pred",
        "proto_wo",
        "proto_w",
        "margin_wo",
        "margin_w",
        "margin_gain",
        "correct_wo",
        "correct_w",
        "compactness_metric",
        "separation_metric",
        "prototype_center_alignment_score",
        "visualization_score",
        "final_rank",
        "flip_to_correct",
        "neg_to_pos",
        "compactness_gain",
        "inter_class_margin_gain",
        "own_sim_wo",
        "own_sim_w",
        "wrong_max_sim_wo",
        "wrong_max_sim_w",
    ]
    keep = [c for c in cols if c in case_df.columns]
    case_df.sort_values("final_rank", ascending=True)[keep].to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    set_seed(int(args.seed))

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[Info] device={device}")

    model = load_model(args, device)
    _src_ds, src_loader = build_source_loader(args)

    print("[Step] build prototypes...")
    proto_bank_wo = build_proto(args, model, src_loader, device, init_mode="kmeans")
    proto_bank_w = build_proto(args, model, src_loader, device, init_mode="nmf")
    proto_wo = class_proto_vectors(proto_bank_wo, int(args.num_classes))
    proto_w = class_proto_vectors(proto_bank_w, int(args.num_classes))

    print("[Step] extract source features...")
    recs = extract_source_features(model, src_loader, device)
    labels = np.asarray([r.y for r in recs], dtype=np.int64)
    case_ids = [r.case_id for r in recs]
    feats = np.asarray([r.feat for r in recs], dtype=np.float32)

    # Ensure prototypes are unit vectors.
    proto_wo = proto_wo / (np.linalg.norm(proto_wo, axis=1, keepdims=True) + 1e-8)
    proto_w = proto_w / (np.linalg.norm(proto_w, axis=1, keepdims=True) + 1e-8)

    case_df_all = build_case_quality_table(
        feats_norm=feats,
        labels=labels,
        case_ids=case_ids,
        proto_wo=proto_wo,
        proto_w=proto_w,
    )
    idx_keep = select_visual_subset(
        case_df=case_df_all,
        n_keep=int(args.num_cases),
        forced_case_id=args.example_case_id,
    )
    feats = feats[idx_keep]
    labels = labels[idx_keep]
    case_ids = [case_ids[i] for i in idx_keep.tolist()]
    case_df = case_df_all[case_df_all["sample_index"].isin(idx_keep)].copy()
    case_df = case_df.sort_values("final_rank", ascending=True).reset_index(drop=True)
    # Re-map global sample_index -> local index in the plotted subset.
    idx_map = {int(src_idx): int(local_idx) for local_idx, src_idx in enumerate(idx_keep.tolist())}
    case_df["sample_index"] = case_df["sample_index"].map(idx_map).astype(np.int64)

    metrics_wo = compute_metrics(feats, labels, proto_wo)
    metrics_w = compute_metrics(feats, labels, proto_w)

    out_png = os.path.join(args.outdir, "prototype_quality_figure.png")
    out_png_mode_b = os.path.join(args.outdir, "prototype_quality_modeB_distribution.png")
    out_csv = os.path.join(args.outdir, "prototype_quality_metrics.csv")
    out_candidates = os.path.join(args.outdir, "prototype_quality_candidates.csv")
    out_meta = os.path.join(args.outdir, "prototype_quality_meta.json")

    draw_figure(
        out_png=out_png,
        feats_norm=feats,
        labels=labels,
        case_ids=case_ids,
        proto_wo=proto_wo,
        proto_w=proto_w,
        metrics_wo=metrics_wo,
        metrics_w=metrics_w,
        case_df=case_df,
        example_case_id=args.example_case_id,
        class_id=args.class_id,
    )
    save_metrics_csv(out_csv, metrics_wo, metrics_w)
    save_candidates_csv(out_candidates, case_df_all)

    mode_b_written = False
    if bool(args.enable_mode_b) and feats.shape[0] > 2:
        with torch.no_grad():
            q_nmf, _ = proto_bank_w.nmf_assign(torch.from_numpy(feats).to(device).float(), iters=60)
        nmf_repr = q_nmf.detach().cpu().numpy().astype(np.float32)
        draw_mode_b_distribution(
            out_png=out_png_mode_b,
            feats_norm=feats,
            nmf_repr=nmf_repr,
            labels=labels,
        )
        mode_b_written = True

    meta = {
        "ckpt": args.ckpt,
        "src_csv": args.src_csv,
        "src_root": args.src_root,
        "num_cases": int(args.num_cases),
        "class_id": args.class_id,
        "num_classes": int(args.num_classes),
        "example_case_id": args.example_case_id,
        "enable_mode_b": bool(args.enable_mode_b),
        "proto_wo_init": "kmeans",
        "proto_w_init": "nmf",
        "metrics_wo": metrics_wo,
        "metrics_w": metrics_w,
        "mode_b_output": out_png_mode_b if mode_b_written else None,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] figure: {out_png}")
    print(f"[Done] metrics: {out_csv}")
    print(f"[Done] candidates: {out_candidates}")
    if mode_b_written:
        print(f"[Done] modeB: {out_png_mode_b}")
    print(f"[Done] meta: {out_meta}")


if __name__ == "__main__":
    main()
