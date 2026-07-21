#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of the role of NMF in prototype construction and pseudo-label assignment.

This script does not retrain. It only loads an existing checkpoint and runs inference-time
analysis with project-native logic:
- model: custom_net.build_custom_model
- data: data.NPYSliceDataset / data.NPYInferDataset
- prototype construction: uda_core.prototypes.PrototypeBank.from_source_init
- pseudo-label assignment evidence: PrototypeBank.nmf_assign + ClasswiseEMAThreshold
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition._nmf import non_negative_factorization
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "suptitle_size": 13,
    "suptitle_linespacing": 1.15,
    "subplot_wspace": 0.24,
    "subplot_hspace": 0.28,
    "tight_h_pad": 1.1,
    "tight_w_pad": 1.0,
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


@dataclass
class SampleRecord:
    domain: str
    case_id: str
    y_true: Optional[int]
    pred_cls: int
    pred_conf: float
    pred_p1: float
    nmf_cls: int
    nmf_conf: float
    nmf_p1: float
    keep_pseudo: Optional[bool]
    tau_used: Optional[float]
    score: float
    raw_map: Optional[np.ndarray]
    fmap: Optional[np.ndarray]  # [C,H,W]
    proto_sim_maps: Optional[np.ndarray]  # [C_cls,H,W]
    proto_class_scores: np.ndarray  # [C_cls]
    nmf_q_proto: np.ndarray  # [sumK]


def _extract_xy_from_batch(batch):
    """
    Return (x, y) from heterogeneous batch structures.
    Supports tuple/list/dict, extra fields are ignored.
    """
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")))
        if x is None or y is None:
            raise ValueError("Cannot extract (x, y) from dict batch")
        return x, y

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list has fewer than 2 elements; cannot extract (x, y)")
        return batch[0], batch[1]

    raise ValueError(f"Unsupported batch type for (x, y) extraction: {type(batch)}")


def _extract_x_y_cid_from_batch(batch, has_label: bool):
    """
    Return (x, y_or_none, cid_str_or_none) from heterogeneous batch structures.
    """
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")) if has_label else None)
        cid = batch.get("case_id", batch.get("cid", batch.get("id")))
        if x is None:
            raise ValueError("Cannot extract x from dict batch")
        if has_label and y is None:
            raise ValueError("Cannot extract y from dict batch while has_label=True")
        return x, y, cid

    if isinstance(batch, (tuple, list)):
        if has_label:
            if len(batch) < 3:
                raise ValueError("Expected labeled batch with at least 3 items: (x, y, cid, ...)")
            return batch[0], batch[1], batch[2]
        if len(batch) < 2:
            raise ValueError("Expected unlabeled batch with at least 2 items: (x, cid, ...)")
        return batch[0], None, batch[1]

    raise ValueError(f"Unsupported batch type for (x, y, cid) extraction: {type(batch)}")


def _batch_cid_to_str(cid):
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
    """
    Lightweight loader wrapper that yields only (x, y), so callers that assume
    2-tuple batches remain compatible.
    """

    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            yield _extract_xy_from_batch(batch)

    def __len__(self):
        return len(self.base_loader)


def parse_args():
    ap = argparse.ArgumentParser("Visualize NMF effects from existing checkpoint (no retraining)")

    ap.add_argument("--ckpt", type=str,
                    default="outputs/checkpoint.pth")
    ap.add_argument("--src_csv", type=str, required=True)
    ap.add_argument("--tgt_csv", type=str, required=True)
    ap.add_argument("--src_root", type=str, required=True)
    ap.add_argument("--tgt_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--num_components", type=int, default=4)
    ap.add_argument("--num_source_cases", type=int, default=2)
    ap.add_argument("--num_target_cases", type=int, default=2)

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

    # Model/inference defaults aligned with MRNet->Knee runs in this repo.
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--num_classes", type=int, default=2)

    # Prototype/NMF assignment settings.
    ap.add_argument("--proto_init", type=str, default="nmf", choices=["kmeans", "nmf"])
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--Kmax", type=int, default=1)
    ap.add_argument("--tau_proto", type=float, default=0.07)
    ap.add_argument("--proto_m", type=float, default=0.97)
    ap.add_argument("--nmf_assign_iters", type=int, default=100)
    ap.add_argument("--beta_loss", type=str, default="frobenius",
                    choices=["frobenius", "kullback-leibler", "itakura-saito"])

    # FreeMatch-style threshold config used by project code.
    ap.add_argument("--ema_m", type=float, default=0.95)

    # Device and reproducibility.
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    return ap.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def tensor_to_img01(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W], normalized by mean/std=0.5 in data.py
    t = x.detach().cpu().float().clone()
    t = t * 0.5 + 0.5
    t = t.clamp(0.0, 1.0)
    img = t.permute(1, 2, 0).numpy()
    if img.shape[2] == 3:
        gray = img.mean(axis=2)
    else:
        gray = img[..., 0]
    return norm01(gray)


def overlay_heatmap(img_gray01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    # base image -> float32 RGB in [0,1]
    base = np.asarray(img_gray01, dtype=np.float32)
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

    # heatmap -> 2D float32 in [0,1]
    hm = np.asarray(heat01, dtype=np.float32)
    hm = np.squeeze(hm)
    if hm.ndim == 3:
        # (C,H,W) or (H,W,C) -> (H,W)
        if hm.shape[0] in (1, 3) and hm.shape[0] != hm.shape[-1]:
            hm = hm.mean(axis=0)
        else:
            hm = hm.mean(axis=-1)
    if hm.ndim != 2:
        raise ValueError(f"overlay_heatmap: expected 2D heatmap after squeeze, got shape={hm.shape}")
    hm = norm01(hm)

    # resize heatmap to base spatial size when needed
    h, w = int(base.shape[0]), int(base.shape[1])
    if hm.shape != (h, w):
        hm_t = torch.from_numpy(hm).view(1, 1, hm.shape[0], hm.shape[1]).float()
        hm_t = F.interpolate(hm_t, size=(h, w), mode="bilinear", align_corners=False)
        hm = hm_t[0, 0].cpu().numpy().astype(np.float32)
        hm = norm01(hm)

    cm = plt.get_cmap(cmap)
    heat_rgb = cm(np.clip(hm, 0, 1))[..., :3].astype(np.float32)
    out = (1.0 - float(alpha)) * base + float(alpha) * heat_rgb
    return np.clip(out, 0.0, 1.0)


def load_model_and_ckpt(args, device: torch.device):
    method = args.backbone.replace("custom_", "") if args.backbone.startswith("custom_") else args.backbone
    model = build_custom_model(
        method=method,
        num_classes=int(args.num_classes),
        pretrained=args.pretrained,
        device=str(device),
    )
    try:
        sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]

    if not isinstance(sd, dict):
        raise RuntimeError("Unsupported checkpoint format: expected state_dict-like object")

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


def build_prototype_bank(args, model, src_loader, device: torch.device):
    feat_dim = int(getattr(model, "feat_dim", 2048))
    proto = PrototypeBank(
        num_classes=int(args.num_classes),
        feat_dim=feat_dim,
        K=int(args.K) if args.K is not None else None,
        Kmax=int(args.Kmax),
        proto_m=float(args.proto_m),
        temp_proto=float(args.tau_proto),
        device=str(device),
    )
    # Adapter layer: from_source_init expects (x, y) batch.
    src_xy_loader = XYOnlyLoader(src_loader)
    proto.from_source_init(
        model=model,
        dl_src=src_xy_loader,
        K=int(args.K) if args.K is not None else None,
        Kmax=int(args.Kmax),
        searchK=(args.K is None),
        init_mode=str(args.proto_init),
    )
    return proto


def extract_feature_maps(model, x: torch.Tensor) -> Optional[torch.Tensor]:
    if hasattr(model, "extract_feature_map"):
        try:
            return model.extract_feature_map(x)
        except Exception:
            return None
    return None


def compute_raw_response_map(fmap: Optional[np.ndarray], feat_vec: np.ndarray) -> Optional[np.ndarray]:
    if fmap is not None and fmap.ndim == 3:
        # [C,H,W] -> [H,W]
        raw = np.mean(np.abs(fmap), axis=0)
        return norm01(raw)

    # vector fallback
    if feat_vec.ndim == 1:
        vec = np.abs(feat_vec)
        if vec.size == 0:
            return None
        side = int(np.ceil(np.sqrt(vec.size)))
        pad = side * side - vec.size
        if pad > 0:
            vec = np.pad(vec, (0, pad), mode="constant")
        return norm01(vec.reshape(side, side))
    return None


def compute_nmf_component_maps(fmap: Optional[np.ndarray], num_components: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      comp_maps: [K,H,W] or None
      coeff_strength: [K] or None
    """
    if fmap is None or fmap.ndim != 3:
        return None, None

    c, h, w = fmap.shape
    X = fmap.transpose(1, 2, 0).reshape(-1, c)  # [HW,C]
    X = norm01(X)
    X = np.clip(X, 0.0, None)

    k = int(max(1, min(num_components, X.shape[0], X.shape[1])))
    if k < 1:
        return None, None

    try:
        W, H, _ = non_negative_factorization(
            X,
            n_components=k,
            init="nndsvda",
            solver="mu",
            beta_loss="frobenius",
            max_iter=200,
            tol=1e-6,
            random_state=0,
        )
        W = np.asarray(W, dtype=np.float32)
        comp_maps = W.reshape(h, w, k).transpose(2, 0, 1)
        comp_maps = np.stack([norm01(m) for m in comp_maps], axis=0)
        coeff = W.mean(axis=0)
        coeff = coeff / (coeff.sum() + 1e-8)
        return comp_maps, coeff.astype(np.float32)
    except Exception as e:
        print(f"[Warn] NMF component decomposition failed: {e}")
        return None, None


def compute_class_proto_vectors(proto: PrototypeBank, num_classes: int) -> np.ndarray:
    mu = proto.mu.detach().cpu().numpy().astype(np.float32)  # [sumK,D]
    out = []
    for c in range(num_classes):
        s, e = proto.offsets[c]
        vc = mu[s:e].mean(axis=0)
        n = np.linalg.norm(vc) + 1e-8
        out.append((vc / n).astype(np.float32))
    return np.stack(out, axis=0)  # [C,D]


def compute_proto_similarity_maps(fmap: Optional[np.ndarray], proto_class_vec: np.ndarray) -> Optional[np.ndarray]:
    """
    fmap: [D,H,W], proto_class_vec: [C,D]
    return [C,H,W] cosine maps normalized to [0,1] per class
    """
    if fmap is None or fmap.ndim != 3:
        return None
    d, h, w = fmap.shape
    pix = fmap.reshape(d, -1).T.astype(np.float32)  # [HW,D]
    pix = pix / (np.linalg.norm(pix, axis=1, keepdims=True) + 1e-8)

    maps = []
    for c in range(proto_class_vec.shape[0]):
        v = proto_class_vec[c]
        sim = pix @ v
        sim = sim.reshape(h, w)
        maps.append(norm01(sim))
    return np.stack(maps, axis=0).astype(np.float32)


def evaluate_source_samples(args, model, proto: PrototypeBank, src_loader, device: torch.device) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    proto_vec = compute_class_proto_vectors(proto, int(args.num_classes))

    with torch.no_grad():
        for batch in src_loader:
            xb, yb, cid = _extract_x_y_cid_from_batch(batch, has_label=True)
            xb = xb.to(device, non_blocking=True)
            y_true = int(yb.item()) if torch.is_tensor(yb) else int(yb)
            case_id = _batch_cid_to_str(cid)
            if case_id is None:
                raise RuntimeError("Missing case_id in source batch")

            logits, feat = model.forward_with_feat(xb)
            prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
            pred_cls = int(np.argmax(prob))
            pred_conf = float(np.max(prob))
            p1 = float(prob[1] if prob.shape[0] > 1 else prob[0])

            fmap_t = extract_feature_maps(model, xb)
            fmap = None if fmap_t is None else fmap_t[0].detach().cpu().numpy().astype(np.float32)
            feat_vec = feat[0].detach().cpu().numpy().astype(np.float32)

            raw_map = compute_raw_response_map(fmap, feat_vec)
            proto_maps = compute_proto_similarity_maps(fmap, proto_vec)

            q_proto, p_cls = proto.nmf_assign(feat, beta_loss=args.beta_loss, iters=int(args.nmf_assign_iters))
            p_cls_np = p_cls[0].detach().cpu().numpy().astype(np.float32)
            nmf_cls = int(np.argmax(p_cls_np))
            nmf_conf = float(np.max(p_cls_np))
            nmf_p1 = float(p_cls_np[1] if p_cls_np.shape[0] > 1 else p_cls_np[0])

            clear_score = float(np.std(raw_map)) if raw_map is not None else 0.0
            score = float(pred_conf + nmf_conf + clear_score)

            records.append(
                SampleRecord(
                    domain="source",
                    case_id=case_id,
                    y_true=y_true,
                    pred_cls=pred_cls,
                    pred_conf=pred_conf,
                    pred_p1=p1,
                    nmf_cls=nmf_cls,
                    nmf_conf=nmf_conf,
                    nmf_p1=nmf_p1,
                    keep_pseudo=None,
                    tau_used=None,
                    score=score,
                    raw_map=raw_map,
                    fmap=fmap,
                    proto_sim_maps=proto_maps,
                    proto_class_scores=p_cls_np,
                    nmf_q_proto=q_proto[0].detach().cpu().numpy().astype(np.float32),
                )
            )
    return records


def evaluate_target_samples(args, model, proto: PrototypeBank, tgt_loader, has_label: bool, device: torch.device) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    proto_vec = compute_class_proto_vectors(proto, int(args.num_classes))
    thresh = ClasswiseEMAThreshold(num_classes=int(args.num_classes), ema_lambda=float(args.ema_m))

    with torch.no_grad():
        for batch in tgt_loader:
            xb, yb, cid = _extract_x_y_cid_from_batch(batch, has_label=has_label)
            y_true = int(yb.item()) if (has_label and torch.is_tensor(yb)) else (int(yb) if has_label else None)

            xb = xb.to(device, non_blocking=True)
            case_id = _batch_cid_to_str(cid)
            if case_id is None:
                raise RuntimeError("Missing case_id in target batch")

            logits, feat = model.forward_with_feat(xb)
            prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
            pred_cls = int(np.argmax(prob))
            pred_conf = float(np.max(prob))
            p1 = float(prob[1] if prob.shape[0] > 1 else prob[0])

            fmap_t = extract_feature_maps(model, xb)
            fmap = None if fmap_t is None else fmap_t[0].detach().cpu().numpy().astype(np.float32)
            feat_vec = feat[0].detach().cpu().numpy().astype(np.float32)

            raw_map = compute_raw_response_map(fmap, feat_vec)
            proto_maps = compute_proto_similarity_maps(fmap, proto_vec)

            q_proto, p_cls = proto.nmf_assign(feat, beta_loss=args.beta_loss, iters=int(args.nmf_assign_iters))
            p_cls_t = p_cls.detach()
            p_cls_np = p_cls_t[0].detach().cpu().numpy().astype(np.float32)
            nmf_cls = int(np.argmax(p_cls_np))
            nmf_conf = float(np.max(p_cls_np))
            nmf_p1 = float(p_cls_np[1] if p_cls_np.shape[0] > 1 else p_cls_np[0])

            tau_map = thresh.update_and_get(p_cls_t).numpy().astype(np.float32)
            tau_used = float(tau_map[nmf_cls])
            keep = bool(nmf_conf > tau_used)

            agree = 1.0 if pred_cls == nmf_cls else 0.0
            consistency = 1.0 - abs(p1 - nmf_p1)
            score = float(min(pred_conf, nmf_conf) + 0.5 * agree + 0.5 * consistency)

            records.append(
                SampleRecord(
                    domain="target",
                    case_id=case_id,
                    y_true=y_true,
                    pred_cls=pred_cls,
                    pred_conf=pred_conf,
                    pred_p1=p1,
                    nmf_cls=nmf_cls,
                    nmf_conf=nmf_conf,
                    nmf_p1=nmf_p1,
                    keep_pseudo=keep,
                    tau_used=tau_used,
                    score=score,
                    raw_map=raw_map,
                    fmap=fmap,
                    proto_sim_maps=proto_maps,
                    proto_class_scores=p_cls_np,
                    nmf_q_proto=q_proto[0].detach().cpu().numpy().astype(np.float32),
                )
            )
    return records


def select_source_cases(records: List[SampleRecord], num_classes: int, num_source_cases: int) -> List[SampleRecord]:
    per_class_quota = max(1, num_source_cases // max(1, num_classes))
    selected: List[SampleRecord] = []

    for c in range(num_classes):
        cand = [r for r in records if r.y_true == c and r.pred_cls == c]
        cand.sort(key=lambda r: r.score, reverse=True)
        selected.extend(cand[:per_class_quota])

    # Fill to requested count if needed.
    if len(selected) < num_source_cases:
        used = {r.case_id for r in selected}
        remain = [r for r in records if r.pred_cls == r.y_true and r.case_id not in used]
        remain.sort(key=lambda r: r.score, reverse=True)
        for r in remain:
            selected.append(r)
            if len(selected) >= num_source_cases:
                break

    # Final fallback.
    if len(selected) < num_source_cases:
        used = {r.case_id for r in selected}
        remain = [r for r in records if r.case_id not in used]
        remain.sort(key=lambda r: r.score, reverse=True)
        for r in remain:
            selected.append(r)
            if len(selected) >= num_source_cases:
                break

    return selected[:num_source_cases]


def select_target_cases(records: List[SampleRecord], num_target_cases: int) -> List[SampleRecord]:
    # Prefer reliable pseudo-labels with agreement.
    cand = [
        r for r in records
        if (r.keep_pseudo is True) and (r.pred_cls == r.nmf_cls)
    ]
    if len(cand) < num_target_cases:
        cand = [r for r in records if (r.keep_pseudo is True)] + [r for r in records if (r.keep_pseudo is not True)]

    dedup = {}
    for r in cand:
        if (r.case_id not in dedup) or (r.score > dedup[r.case_id].score):
            dedup[r.case_id] = r

    final = list(dedup.values())
    final.sort(key=lambda r: r.score, reverse=True)
    return final[:num_target_cases]


def build_component_montage(comp_maps: np.ndarray) -> np.ndarray:
    # [K,H,W] -> [H, K*W]
    chunks = [comp_maps[k] for k in range(comp_maps.shape[0])]
    return np.concatenate(chunks, axis=1)


def draw_figure(
    out_png: str,
    out_pdf: str,
    source_case: SampleRecord,
    target_case: SampleRecord,
    src_img_gray: np.ndarray,
    tgt_img_gray: np.ndarray,
    num_components: int,
):
    _apply_paper_style()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # ---------- Row 1: Source ----------
    axes[0, 0].imshow(src_img_gray, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[0, 0], "Source sample")
    axes[0, 0].axis("off")

    if source_case.raw_map is not None:
        ov = overlay_heatmap(src_img_gray, source_case.raw_map)
        axes[0, 1].imshow(ov, vmin=0, vmax=1)
    else:
        axes[0, 1].imshow(src_img_gray, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[0, 1], "Raw feature response")
    axes[0, 1].axis("off")

    comp_maps, coeff = compute_nmf_component_maps(source_case.fmap, num_components=num_components)
    if comp_maps is not None:
        montage = build_component_montage(comp_maps)
        axes[0, 2].imshow(montage, cmap="viridis", vmin=0, vmax=1)
        _set_panel_title(axes[0, 2], f"NMF components (K={comp_maps.shape[0]})")
        axes[0, 2].axis("off")
    else:
        v = source_case.nmf_q_proto
        axes[0, 2].bar(np.arange(len(v)), v, color="#4c72b0")
        _set_panel_title(axes[0, 2], "NMF coefficients")

    if source_case.proto_sim_maps is not None and source_case.y_true is not None:
        c = int(source_case.y_true)
        proto_map = source_case.proto_sim_maps[c]
        ovp = overlay_heatmap(src_img_gray, proto_map)
        axes[0, 3].imshow(ovp, vmin=0, vmax=1)
        _set_panel_title(axes[0, 3], "Prototype response")
        axes[0, 3].axis("off")
    else:
        v = source_case.proto_class_scores
        axes[0, 3].bar(np.arange(len(v)), v, color="#55a868")
        _set_panel_title(axes[0, 3], "Prototype matching")

    axes[0, 4].axis("off")
    lines_src = [
        "source features",
        "   -> NMF components",
        "   -> class prototype",
        "",
        f"Case: {source_case.case_id}",
        f"GT={source_case.y_true}  Pred={source_case.pred_cls}",
        f"P(cls1)={source_case.pred_p1:.3f}",
        f"NMF P(cls1)={source_case.nmf_p1:.3f}",
    ]
    axes[0, 4].text(0.02, 0.98, "\n".join(lines_src), va="top", ha="left", fontsize=PAPER_STYLE["annotation_size"])
    _set_panel_title(axes[0, 4], "Summary panel")

    # ---------- Row 2: Target ----------
    axes[1, 0].imshow(tgt_img_gray, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[1, 0], "Target sample")
    axes[1, 0].axis("off")

    if target_case.raw_map is not None:
        ovt = overlay_heatmap(tgt_img_gray, target_case.raw_map)
        axes[1, 1].imshow(ovt, vmin=0, vmax=1)
    else:
        axes[1, 1].imshow(tgt_img_gray, cmap="gray", vmin=0, vmax=1)
    _set_panel_title(axes[1, 1], "Raw feature response")
    axes[1, 1].axis("off")

    if target_case.proto_sim_maps is not None and target_case.proto_sim_maps.shape[0] >= 2:
        ov0 = overlay_heatmap(tgt_img_gray, target_case.proto_sim_maps[0])
        ov1 = overlay_heatmap(tgt_img_gray, target_case.proto_sim_maps[1])
        axes[1, 2].imshow(ov0, vmin=0, vmax=1)
        _set_panel_title(axes[1, 2], "Similarity to healthy prototype")
        axes[1, 2].axis("off")
        axes[1, 3].imshow(ov1, vmin=0, vmax=1)
        _set_panel_title(axes[1, 3], "Similarity to tear prototype")
        axes[1, 3].axis("off")
    else:
        v = target_case.proto_class_scores
        axes[1, 2].bar(np.arange(len(v)), v, color="#8172b3")
        _set_panel_title(axes[1, 2], "Similarity to class-0")
        axes[1, 3].bar(np.arange(len(v)), v, color="#ccb974")
        _set_panel_title(axes[1, 3], "Similarity to class-1")

    # Pseudo-label evidence panel
    _set_panel_title(axes[1, 4], "Pseudo-label evidence")
    axes[1, 4].set_xlim(0, 1.0)
    axes[1, 4].set_ylim(-0.5, 2.5)
    axes[1, 4].barh([2], [target_case.pred_p1], color="#4c72b0", label="Classifier P(cls1)")
    axes[1, 4].barh([1], [target_case.nmf_p1], color="#55a868", label="NMF score P(cls1)")
    tau = 0.0 if target_case.tau_used is None else float(target_case.tau_used)
    axes[1, 4].barh([0], [tau], color="#c44e52", label="Threshold")
    axes[1, 4].set_yticks([2, 1, 0])
    axes[1, 4].set_yticklabels(["Classifier", "NMF", "Tau"])
    axes[1, 4].grid(axis="x", linestyle="--", alpha=0.3)

    pseudo_label = target_case.nmf_cls
    keep_txt = "yes" if target_case.keep_pseudo else "no"
    txt = (
        f"Case: {target_case.case_id}\n"
        f"Cls pred={target_case.pred_cls} (conf={target_case.pred_conf:.3f})\n"
        f"NMF pred={target_case.nmf_cls} (conf={target_case.nmf_conf:.3f})\n"
        f"Pseudo label={pseudo_label}, selected={keep_txt}"
    )
    axes[1, 4].text(0.02, -0.35, txt, fontsize=PAPER_STYLE["annotation_size"], va="top")

    fig.suptitle(
        "Visualization of the role of NMF in prototype construction and pseudo-label assignment",
        fontsize=PAPER_STYLE["suptitle_size"],
        linespacing=PAPER_STYLE["suptitle_linespacing"],
    )
    fig.subplots_adjust(wspace=PAPER_STYLE["subplot_wspace"], hspace=PAPER_STYLE["subplot_hspace"])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=PAPER_STYLE["tight_h_pad"], w_pad=PAPER_STYLE["tight_w_pad"])

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_selected_csv(path: str, selected_source: List[SampleRecord], selected_target: List[SampleRecord],
                       used_source_case: str, used_target_case: str):
    rows = []
    for r in selected_source:
        rows.append({
            "domain": "source",
            "case_id": r.case_id,
            "used_in_main_figure": int(r.case_id == used_source_case),
            "y_true": "" if r.y_true is None else int(r.y_true),
            "pred_cls": int(r.pred_cls),
            "pred_conf": float(r.pred_conf),
            "pred_p1": float(r.pred_p1),
            "nmf_cls": int(r.nmf_cls),
            "nmf_conf": float(r.nmf_conf),
            "nmf_p1": float(r.nmf_p1),
            "keep_pseudo": "",
            "tau_used": "",
            "selection_score": float(r.score),
        })
    for r in selected_target:
        rows.append({
            "domain": "target",
            "case_id": r.case_id,
            "used_in_main_figure": int(r.case_id == used_target_case),
            "y_true": "" if r.y_true is None else int(r.y_true),
            "pred_cls": int(r.pred_cls),
            "pred_conf": float(r.pred_conf),
            "pred_p1": float(r.pred_p1),
            "nmf_cls": int(r.nmf_cls),
            "nmf_conf": float(r.nmf_conf),
            "nmf_p1": float(r.nmf_p1),
            "keep_pseudo": int(bool(r.keep_pseudo)),
            "tau_used": "" if r.tau_used is None else float(r.tau_used),
            "selection_score": float(r.score),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_image_by_case_id(loader, case_id: str, has_label: bool) -> Optional[np.ndarray]:
    for batch in loader:
        x, _y, cid = _extract_x_y_cid_from_batch(batch, has_label=has_label)
        cur = _batch_cid_to_str(cid)
        if cur == case_id:
            return tensor_to_img01(x[0])
    return None


def main():
    args = parse_args()
    ensure_dir(args.outdir)
    set_seed(int(args.seed))

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[Info] device={device}")

    model = load_model_and_ckpt(args, device)

    src_ds, src_loader = build_source_loader(args)
    tgt_ds, tgt_loader, tgt_has_label = build_target_loader(args)

    proto = build_prototype_bank(args, model, src_loader, device)

    print("[Step] evaluating source samples...")
    src_records = evaluate_source_samples(args, model, proto, src_loader, device)

    print("[Step] evaluating target samples...")
    tgt_records = evaluate_target_samples(args, model, proto, tgt_loader, tgt_has_label, device)

    selected_source = select_source_cases(src_records, int(args.num_classes), int(args.num_source_cases))
    selected_target = select_target_cases(tgt_records, int(args.num_target_cases))

    if len(selected_source) == 0:
        raise RuntimeError("No source case selected. Please check source data / labels.")
    if len(selected_target) == 0:
        raise RuntimeError("No target case selected. Please check target data.")

    source_case = selected_source[0]
    target_case = selected_target[0]

    src_img = get_image_by_case_id(src_loader, source_case.case_id, has_label=True)
    tgt_img = get_image_by_case_id(tgt_loader, target_case.case_id, has_label=tgt_has_label)
    if src_img is None:
        raise RuntimeError(f"Cannot fetch source image for case_id={source_case.case_id}")
    if tgt_img is None:
        raise RuntimeError(f"Cannot fetch target image for case_id={target_case.case_id}")

    out_png = os.path.join(args.outdir, "nmf_effect_figure.png")
    out_pdf = os.path.join(args.outdir, "nmf_effect_figure.pdf")
    out_csv = os.path.join(args.outdir, "selected_cases.csv")
    out_json = os.path.join(args.outdir, "vis_metadata.json")

    draw_figure(
        out_png=out_png,
        out_pdf=out_pdf,
        source_case=source_case,
        target_case=target_case,
        src_img_gray=src_img,
        tgt_img_gray=tgt_img,
        num_components=int(args.num_components),
    )

    write_selected_csv(
        path=out_csv,
        selected_source=selected_source,
        selected_target=selected_target,
        used_source_case=source_case.case_id,
        used_target_case=target_case.case_id,
    )

    meta = {
        "ckpt": args.ckpt,
        "src_csv": args.src_csv,
        "tgt_csv": args.tgt_csv,
        "src_root": args.src_root,
        "tgt_root": args.tgt_root,
        "num_components": int(args.num_components),
        "num_source_cases": int(args.num_source_cases),
        "num_target_cases": int(args.num_target_cases),
        "target_has_label": bool(tgt_has_label),
        "source_case_used_in_figure": source_case.case_id,
        "target_case_used_in_figure": target_case.case_id,
        "selected_source_cases": [r.case_id for r in selected_source],
        "selected_target_cases": [r.case_id for r in selected_target],
        "note": "No retraining. Prototype/NMF/pseudo-label evidence computed in inference mode with project-native logic.",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] figure png: {out_png}")
    print(f"[Done] figure pdf: {out_pdf}")
    print(f"[Done] selected cases: {out_csv}")
    print(f"[Done] metadata: {out_json}")


if __name__ == "__main__":
    main()
