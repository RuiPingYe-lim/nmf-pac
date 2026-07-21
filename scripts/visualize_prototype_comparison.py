#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure X: Comparison between Mean Prototypes and NMF Prototypes.

Main design:
1) Fix ONE source-trained feature extractor checkpoint.
2) Build prototypes only from source-domain features.
3) Compare Mean prototype vs NMF prototype construction.
4) Optionally evaluate prototype behaviors on target-domain features.

This script prioritizes project-native reuse:
- model construction/loading: custom_net.build_custom_model
- data loading: data.NPYSliceDataset / data.NPYInferDataset
- project NMF prototype core: uda_core.prototypes.PrototypeBank.from_source_init(init_mode="nmf")
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.decomposition import NMF
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from custom_net import build_custom_model
from data import NPYInferDataset, NPYSliceDataset
from uda_core.prototypes import PrototypeBank


DEFAULT_CKPT = "outputs/checkpoint.pth"


@dataclass
class ExtractedSet:
    features: np.ndarray
    labels: Optional[np.ndarray]
    ids: List[str]
    images: Optional[np.ndarray] = None
    feature_maps: Optional[np.ndarray] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize Mean-vs-NMF prototype comparison from a fixed source checkpoint")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--src_root", type=str, required=True)
    ap.add_argument("--src_csv", type=str, required=True)
    ap.add_argument("--tgt_root", type=str, default=None)
    ap.add_argument("--tgt_csv", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--plane", type=str, default="sagittal", choices=["sagittal", "coronal", "axial"])
    ap.add_argument("--id_col", type=str, default="case_id")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--nmf_components", type=int, default=1)
    ap.add_argument("--topk_dims", type=int, default=128)
    ap.add_argument("--feature_layer", type=str, default=None)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--single_file_case", action="store_true", default=True)
    ap.add_argument("--id_zero_pad", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _cid_to_list(cid, bsz: int) -> List[str]:
    if cid is None:
        return [f"idx_{i}" for i in range(bsz)]
    if isinstance(cid, (list, tuple)):
        out = [str(x) for x in cid]
        if len(out) < bsz:
            out.extend([out[-1] if out else ""] * (bsz - len(out)))
        return out[:bsz]
    if torch.is_tensor(cid):
        if cid.ndim == 0:
            return [str(cid.detach().cpu().item()) for _ in range(bsz)]
        flat = cid.detach().cpu().reshape(-1).tolist()
        out = [str(x) for x in flat]
        if len(out) < bsz:
            out.extend([out[-1] if out else ""] * (bsz - len(out)))
        return out[:bsz]
    return [str(cid) for _ in range(bsz)]


def _extract_batch_labeled_or_unlabeled(batch):
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("img")))
        y = batch.get("y", batch.get("label", batch.get("target")))
        cid = batch.get("case_id", batch.get("cid", batch.get("id")))
        if x is None:
            raise ValueError("Cannot extract x from dict batch")
        return x, y, cid
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], None, batch[1]
    raise ValueError(f"Unsupported batch type: {type(batch)}")


class XYOnlyLoader:
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            x, y, _ = _extract_batch_labeled_or_unlabeled(batch)
            if y is None:
                continue
            yield x, y

    def __len__(self):
        return len(self.base_loader)


class FeatureHook:
    def __init__(self, model, layer_name: Optional[str]):
        self.model = model
        self.layer_name = layer_name
        self.last = None
        self.handle = None
        self.enabled = False

        if layer_name is None or str(layer_name).strip() == "":
            return
        module = self._resolve_module(model, layer_name)
        if module is None:
            print(f"[Warn] feature_layer='{layer_name}' not found; feature-map export is disabled.")
            return
        self.handle = module.register_forward_hook(self._hook_fn)
        self.enabled = True
        print(f"[Info] feature hook attached at layer: {layer_name}")

    def _resolve_module(self, root, name: str):
        cur = root
        for part in str(name).split("."):
            if part.isdigit():
                idx = int(part)
                if isinstance(cur, (torch.nn.Sequential, list, tuple)) and idx < len(cur):
                    cur = cur[idx]
                else:
                    return None
            else:
                if not hasattr(cur, part):
                    return None
                cur = getattr(cur, part)
        return cur

    def _hook_fn(self, _mod, _inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        self.last = out

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.enabled = False


def load_model(args: argparse.Namespace, device: torch.device):
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
        raise RuntimeError("Unsupported checkpoint format")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Init] ckpt={args.ckpt}")
    print(f"[Init] missing_keys={len(getattr(msg, 'missing_keys', []))} unexpected_keys={len(getattr(msg, 'unexpected_keys', []))}")
    model.eval()
    return model


def build_dataloader(
    root: str,
    csv_path: str,
    args: argparse.Namespace,
    is_target: bool = False,
) -> Tuple[DataLoader, bool]:
    if root is None or csv_path is None:
        raise ValueError("root/csv cannot be None when building dataloader")
    df = pd.read_csv(csv_path)
    has_label = args.label_col in df.columns
    if has_label:
        ds = NPYSliceDataset(
            npy_root=root,
            csv_file=csv_path,
            plane=args.plane,
            id_col=args.id_col,
            label_col=args.label_col,
            resize=args.resize,
            single_file_case=bool(args.single_file_case),
            id_zero_pad=int(args.id_zero_pad),
            augment=False,
            return_case_id=True,
        )
    else:
        case_ids = df[args.id_col].astype(str).tolist()
        ds = NPYInferDataset(
            npy_root=root,
            case_ids=case_ids,
            plane=args.plane,
            resize=args.resize,
            single_file_case=bool(args.single_file_case),
            id_zero_pad=int(args.id_zero_pad),
        )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
    )
    print(f"[Data] {'target' if is_target else 'source'} samples={len(ds)} has_label={has_label}")
    return dl, has_label


def extract_features(
    model,
    loader: DataLoader,
    device: torch.device,
    has_label: bool,
    feature_hook: Optional[FeatureHook] = None,
    export_images_and_fmaps: bool = False,
) -> ExtractedSet:
    feats: List[np.ndarray] = []
    labs: List[int] = []
    ids: List[str] = []
    imgs: List[np.ndarray] = []
    fmaps: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x, y, cid = _extract_batch_labeled_or_unlabeled(batch)
            x = x.to(device, non_blocking=True)
            logits, feat = model.forward_with_feat(x)

            bsz = int(x.shape[0])
            ids_b = _cid_to_list(cid, bsz)
            ids.extend(ids_b)
            feats.append(feat.detach().cpu().numpy().astype(np.float32))

            if has_label and y is not None:
                if torch.is_tensor(y):
                    labs.extend([int(v) for v in y.detach().cpu().numpy().reshape(-1).tolist()])
                else:
                    labs.extend([int(v) for v in np.asarray(y).reshape(-1).tolist()])

            if export_images_and_fmaps:
                x_img = ((x.detach().cpu() * 0.5) + 0.5).clamp(0.0, 1.0)  # [B,3,H,W]
                x_gray = x_img.mean(dim=1)  # [B,H,W]
                imgs.append(x_gray.numpy().astype(np.float32))

                if feature_hook is not None and feature_hook.enabled and feature_hook.last is not None:
                    fm = feature_hook.last
                    if torch.is_tensor(fm):
                        if fm.ndim == 4 and fm.shape[0] == bsz:
                            fmaps.append(fm.detach().cpu().numpy().astype(np.float32))
                        else:
                            fmaps.append(np.zeros((bsz, 1, 1, 1), dtype=np.float32))
                    else:
                        fmaps.append(np.zeros((bsz, 1, 1, 1), dtype=np.float32))
                else:
                    fmaps.append(np.zeros((bsz, 1, 1, 1), dtype=np.float32))

    features = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    labels = np.asarray(labs, dtype=np.int64) if has_label else None
    images = np.concatenate(imgs, axis=0) if len(imgs) > 0 else None
    feature_maps = np.concatenate(fmaps, axis=0) if len(fmaps) > 0 else None
    print(f"[Feat] features shape={features.shape}")
    if labels is not None:
        uniq, cnts = np.unique(labels, return_counts=True)
        print(f"[Feat] label counts: {dict(zip([int(u) for u in uniq], [int(c) for c in cnts]))}")
    if images is not None:
        print(f"[Feat] exported images shape={images.shape}")
    if feature_maps is not None:
        print(f"[Feat] exported feature_maps shape={feature_maps.shape}")
    return ExtractedSet(features=features, labels=labels, ids=ids, images=images, feature_maps=feature_maps)


def make_features_nonnegative(features: np.ndarray, mode: str = "shift", eps: float = 1e-6) -> Tuple[np.ndarray, Dict[str, float]]:
    x = np.asarray(features, dtype=np.float32)
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    info: Dict[str, float] = {"mode": mode, "input_min": min_v, "input_max": max_v}
    if mode == "none":
        info["shift"] = 0.0
        return x, info
    if mode == "shift":
        shift = 0.0 if min_v >= 0.0 else (-min_v + float(eps))
        out = x + shift
        info["shift"] = float(shift)
        info["output_min"] = float(np.min(out))
        info["output_max"] = float(np.max(out))
        return out.astype(np.float32), info
    raise ValueError(f"Unknown nonnegative mode: {mode}")


def build_mean_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    d = int(features.shape[1])
    out = np.zeros((num_classes, d), dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            out[c] = features[idx].mean(axis=0).astype(np.float32)
    return out


def build_nmf_prototypes_from_project_code(
    args: argparse.Namespace,
    model,
    src_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    proto = PrototypeBank(
        num_classes=int(num_classes),
        feat_dim=int(getattr(model, "feat_dim", 2048)),
        K=int(args.nmf_components),
        Kmax=int(max(1, args.nmf_components)),
        proto_m=0.97,
        temp_proto=0.07,
        device=str(device),
    )
    proto.from_source_init(
        model=model,
        dl_src=XYOnlyLoader(src_loader),
        K=int(args.nmf_components),
        Kmax=int(max(1, args.nmf_components)),
        searchK=False,
        init_mode="nmf",
    )
    mu = proto.mu.detach().cpu().numpy().astype(np.float32)
    out = []
    for c in range(num_classes):
        s, e = proto.offsets[c]
        vc = mu[s:e].mean(axis=0).astype(np.float32)
        out.append(vc)
    return np.stack(out, axis=0)


def build_nmf_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_components: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    # Audit branch using sklearn.decomposition.NMF exactly as requested.
    feats_nonneg, nn_info = make_features_nonnegative(features, mode="shift")
    d = int(feats_nonneg.shape[1])
    out = np.zeros((num_classes, d), dtype=np.float32)
    cls_info: Dict[str, object] = {"nonnegative_transform": nn_info, "class_status": {}}
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            cls_info["class_status"][str(c)] = {"n_samples": 0, "status": "empty"}
            continue
        x_c = feats_nonneg[idx]
        k = max(1, int(n_components))
        nmf = NMF(
            n_components=k,
            init="nndsvda",
            solver="mu",
            beta_loss="frobenius",
            max_iter=400,
            random_state=int(random_state),
        )
        w = nmf.fit_transform(x_c)
        h = nmf.components_.astype(np.float32)  # [k, D]
        out[c] = h.mean(axis=0).astype(np.float32)
        cls_info["class_status"][str(c)] = {
            "n_samples": int(len(idx)),
            "status": "ok",
            "n_components": int(k),
            "n_iter": int(getattr(nmf, "n_iter_", -1)),
            "reconstruction_err": float(getattr(nmf, "reconstruction_err_", float("nan"))),
            "w_mean": float(np.mean(w)),
        }
    return out, cls_info


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def compute_similarity_stats(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    method_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x = _l2n(features.astype(np.float32))
    p = _l2n(prototypes.astype(np.float32))
    sim = x @ p.T
    n = x.shape[0]
    rows: List[Dict[str, object]] = []
    intra_list = []
    inter_list = []
    margin_list = []
    for i in range(n):
        y = int(labels[i])
        other = 1 - y if p.shape[0] == 2 else int(np.argmax(np.where(np.arange(p.shape[0]) != y, sim[i], -1e9)))
        intra = float(sim[i, y])
        inter = float(sim[i, other])
        margin = intra - inter
        intra_list.append(intra)
        inter_list.append(inter)
        margin_list.append(margin)
        rows.append({"Method": method_name, "Type": "Intra", "Value": intra})
        rows.append({"Method": method_name, "Type": "Inter", "Value": inter})
        rows.append({"Method": method_name, "Type": "Margin", "Value": margin})
    df = pd.DataFrame(rows)
    stats = {
        "mean_intra_similarity": float(np.mean(intra_list)) if len(intra_list) > 0 else float("nan"),
        "mean_inter_similarity": float(np.mean(inter_list)) if len(inter_list) > 0 else float("nan"),
        "mean_margin": float(np.mean(margin_list)) if len(margin_list) > 0 else float("nan"),
    }
    return df, stats


def classify_by_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    x = _l2n(features.astype(np.float32))
    p = _l2n(prototypes.astype(np.float32))
    sim = x @ p.T
    pred = np.argmax(sim, axis=1).astype(np.int64)
    # Binary metrics as requested; fall back to macro if label set is not strictly {0,1}.
    uniq = sorted(list(set([int(v) for v in np.unique(labels).tolist()])))
    avg_mode = "binary" if uniq == [0, 1] else "macro"
    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, average=avg_mode, zero_division=0)),
        "recall": float(recall_score(labels, pred, average=avg_mode, zero_division=0)),
        "f1": float(f1_score(labels, pred, average=avg_mode, zero_division=0)),
    }
    cm = confusion_matrix(labels, pred, labels=[0, 1] if max(uniq) <= 1 else uniq)
    # For activation-example sampling.
    if p.shape[0] == 2:
        margins = sim[:, 1] - sim[:, 0]
    else:
        top2 = np.sort(sim, axis=1)[:, -2:]
        margins = top2[:, 1] - top2[:, 0]
    return pred, metrics, cm, margins.astype(np.float32)


def hoyer_sparsity(v: np.ndarray) -> float:
    x = np.abs(np.asarray(v, dtype=np.float32).reshape(-1))
    n = float(x.size)
    if n <= 1:
        return 0.0
    l1 = float(np.sum(x))
    l2 = float(np.sqrt(np.sum(x * x)) + 1e-8)
    return float((np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0))


def plot_prototype_heatmap(
    ax,
    mean_prototypes: np.ndarray,
    nmf_prototypes: np.ndarray,
    topk_dims: int,
) -> Dict[str, object]:
    d = int(mean_prototypes.shape[1])
    delta_mean = np.abs(mean_prototypes[1] - mean_prototypes[0])
    delta_nmf = np.abs(nmf_prototypes[1] - nmf_prototypes[0])
    importance = np.maximum(delta_mean, delta_nmf)
    k = min(int(topk_dims), d) if int(topk_dims) > 0 else d
    idx = np.argsort(-importance)[:k]

    mat = np.stack(
        [
            mean_prototypes[0, idx],
            nmf_prototypes[0, idx],
            mean_prototypes[1, idx],
            nmf_prototypes[1, idx],
        ],
        axis=0,
    )
    vmin = float(np.min(mat))
    vmax = float(np.max(mat))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=["Class 0 - Mean", "Class 0 - NMF", "Class 1 - Mean", "Class 1 - NMF"],
    )
    ax.set_title("(a) Prototype Heatmap")
    ax.set_xlabel(f"Top-{k} discriminative feature dimensions")
    ax.set_ylabel("Prototype rows")
    return {"selected_dims": idx.astype(int).tolist(), "vmin": vmin, "vmax": vmax, "k": int(k)}


def plot_similarity_distribution(ax, df_long: pd.DataFrame) -> None:
    sns.violinplot(
        data=df_long,
        x="Type",
        y="Value",
        hue="Method",
        split=False,
        inner="quartile",
        palette={"Mean": "#d95f02", "NMF": "#1b9e77"},
        ax=ax,
    )
    ax.set_title("(b) Intra / Inter / Margin Distribution (Target)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")


def plot_confusion_matrices(
    ax_left,
    ax_right,
    cm_mean: np.ndarray,
    cm_nmf: np.ndarray,
    acc_mean: float,
    acc_nmf: float,
) -> None:
    sns.heatmap(cm_mean, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_left)
    sns.heatmap(cm_nmf, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_right)
    ax_left.set_title(f"Mean Prototype\nacc={acc_mean:.3f}")
    ax_right.set_title(f"NMF Prototype\nacc={acc_nmf:.3f}")
    for ax in [ax_left, ax_right]:
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)


def _resize_to_img(hm: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(hm[None, None]).float()
    out = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)


def _norm01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn + 1e-8)).astype(np.float32)


def _overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    color = cmap(np.clip(heat, 0.0, 1.0))[..., :3]
    base = np.stack([img, img, img], axis=-1)
    out = (1.0 - alpha) * base + alpha * color
    return np.clip(out, 0.0, 1.0)


def plot_activation_examples(
    fig,
    spec,
    tgt_images: np.ndarray,
    tgt_feature_maps: np.ndarray,
    tgt_labels: np.ndarray,
    mean_prototypes: np.ndarray,
    nmf_prototypes: np.ndarray,
    margins_nmf: np.ndarray,
    pred_nmf: np.ndarray,
) -> Dict[str, object]:
    n = int(tgt_images.shape[0])
    c = int(tgt_feature_maps.shape[1])
    d = int(mean_prototypes.shape[1])
    if d != c:
        ax = fig.add_subplot(spec)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "(d) Activation examples skipped:\nprototype dim != feature-map channels",
            ha="center",
            va="center",
            fontsize=10,
        )
        return {"status": "skipped_dim_mismatch", "D": d, "C": c}

    correct_idx = np.where(pred_nmf == tgt_labels)[0]
    if len(correct_idx) == 0:
        correct_idx = np.arange(n)
    abs_m = np.abs(margins_nmf)
    easy = int(correct_idx[np.argmax(abs_m[correct_idx])])
    hard = int(correct_idx[np.argmin(abs_m[correct_idx])])
    amb = int(np.argmin(abs_m))
    chosen = [easy, hard, amb]
    tags = ["easy", "hard", "ambiguous"]

    g = GridSpecFromSubplotSpec(3, 3, subplot_spec=spec, wspace=0.08, hspace=0.14)
    mean_p = _l2n(mean_prototypes.astype(np.float32))
    nmf_p = _l2n(nmf_prototypes.astype(np.float32))

    for r, idx in enumerate(chosen):
        y = int(tgt_labels[idx])
        img = tgt_images[idx]
        fmap = tgt_feature_maps[idx]  # [C,Hf,Wf]
        hm_mean = np.tensordot(mean_p[y], fmap, axes=(0, 0))
        hm_nmf = np.tensordot(nmf_p[y], fmap, axes=(0, 0))
        hm_mean = _norm01(_resize_to_img(hm_mean, img.shape[0], img.shape[1]))
        hm_nmf = _norm01(_resize_to_img(hm_nmf, img.shape[0], img.shape[1]))

        p0 = fig.add_subplot(g[r, 0])
        p1 = fig.add_subplot(g[r, 1])
        p2 = fig.add_subplot(g[r, 2])
        p0.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        p1.imshow(_overlay(img, hm_mean))
        p2.imshow(_overlay(img, hm_nmf))
        for ax in [p0, p1, p2]:
            ax.set_xticks([])
            ax.set_yticks([])
        if r == 0:
            p0.set_title("Image", fontsize=9)
            p1.set_title("Mean Response", fontsize=9)
            p2.set_title("NMF Response", fontsize=9)
        p0.set_ylabel(f"{tags[r]} (y={y})", fontsize=8)

    frame = fig.add_subplot(spec)
    frame.patch.set_alpha(0.0)
    frame.set_xticks([])
    frame.set_yticks([])
    frame.set_title("(d) Activation / Back-projection Examples", fontsize=11, pad=8)
    for spine in frame.spines.values():
        spine.set_visible(False)
    return {"status": "ok", "chosen_indices": [int(v) for v in chosen]}


def save_metrics(path: str, metrics: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"[Info] device={device}")
    model = load_model(args, device)

    src_loader, src_has_label = build_dataloader(args.src_root, args.src_csv, args, is_target=False)
    if not src_has_label:
        raise RuntimeError("Source CSV must contain label column for prototype construction.")

    hook = FeatureHook(model, args.feature_layer)
    src_set = extract_features(
        model=model,
        loader=src_loader,
        device=device,
        has_label=True,
        feature_hook=hook,
        export_images_and_fmaps=False,
    )
    src_features = src_set.features
    src_labels = src_set.labels
    src_ids = src_set.ids

    np.save(os.path.join(args.outdir, "src_features.npy"), src_features)
    np.save(os.path.join(args.outdir, "src_labels.npy"), src_labels)
    np.save(os.path.join(args.outdir, "src_ids.npy"), np.asarray(src_ids, dtype=object))
    print(f"[Save] source features/labels/ids -> {args.outdir}")

    num_classes = int(args.num_classes)
    if src_labels is not None:
        uniq = sorted([int(v) for v in np.unique(src_labels).tolist()])
        if len(uniq) <= num_classes:
            num_classes = max(num_classes, max(uniq) + 1 if len(uniq) > 0 else num_classes)
    if num_classes != 2:
        print(f"[Warn] This task expects binary classes; current num_classes={num_classes}.")

    mean_prototypes = build_mean_prototypes(src_features, src_labels, num_classes=num_classes)
    nmf_prototypes_project = build_nmf_prototypes_from_project_code(
        args=args,
        model=model,
        src_loader=src_loader,
        device=device,
        num_classes=num_classes,
    )
    # sklearn NMF audit branch (for explicit non-negative transform / convergence logging).
    nmf_prototypes_sklearn, nmf_sklearn_info = build_nmf_prototypes(
        features=src_features,
        labels=src_labels,
        num_classes=num_classes,
        n_components=int(args.nmf_components),
        random_state=int(args.seed),
    )

    # Use project-native NMF prototypes as the main comparison object.
    nmf_prototypes = nmf_prototypes_project
    np.save(os.path.join(args.outdir, "mean_prototypes.npy"), mean_prototypes.astype(np.float32))
    np.save(os.path.join(args.outdir, "nmf_prototypes.npy"), nmf_prototypes.astype(np.float32))

    # Optional: save the sklearn-audit prototype for transparency.
    np.save(os.path.join(args.outdir, "nmf_prototypes_sklearn_audit.npy"), nmf_prototypes_sklearn.astype(np.float32))

    tgt_set = None
    tgt_features = None
    tgt_labels = None
    tgt_ids = None
    if args.tgt_root and args.tgt_csv:
        tgt_loader, tgt_has_label = build_dataloader(args.tgt_root, args.tgt_csv, args, is_target=True)
        tgt_set = extract_features(
            model=model,
            loader=tgt_loader,
            device=device,
            has_label=tgt_has_label,
            feature_hook=hook,
            export_images_and_fmaps=bool(hook.enabled),
        )
        tgt_features = tgt_set.features
        tgt_labels = tgt_set.labels
        tgt_ids = tgt_set.ids
        np.save(os.path.join(args.outdir, "tgt_features.npy"), tgt_features)
        np.save(os.path.join(args.outdir, "tgt_ids.npy"), np.asarray(tgt_ids, dtype=object))
        if tgt_labels is not None:
            np.save(os.path.join(args.outdir, "tgt_labels.npy"), tgt_labels)
        if tgt_set.images is not None:
            np.save(os.path.join(args.outdir, "tgt_images.npy"), tgt_set.images.astype(np.float32))
        if tgt_set.feature_maps is not None and hook.enabled:
            np.save(os.path.join(args.outdir, "tgt_feature_maps.npy"), tgt_set.feature_maps.astype(np.float32))
        print(f"[Save] target features/ids (+optional labels/maps) -> {args.outdir}")

    if hook is not None:
        hook.close()

    # Figure X
    sns.set_theme(style="whitegrid", context="paper")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    outer = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.28)

    ax_a = fig.add_subplot(outer[0, 0])
    heat_meta = plot_prototype_heatmap(
        ax=ax_a,
        mean_prototypes=mean_prototypes,
        nmf_prototypes=nmf_prototypes,
        topk_dims=int(args.topk_dims),
    )

    # (b) similarity distribution
    ax_b = fig.add_subplot(outer[0, 1])
    sim_stats = {}
    df_long = None
    pred_mean = pred_nmf = None
    cm_mean = cm_nmf = None
    metrics_mean = metrics_nmf = None
    margins_mean = margins_nmf = None
    activation_meta = {"status": "skipped_no_target"}
    if tgt_features is not None and tgt_labels is not None:
        df_mean, stat_mean = compute_similarity_stats(tgt_features, tgt_labels, mean_prototypes, "Mean")
        df_nmf, stat_nmf = compute_similarity_stats(tgt_features, tgt_labels, nmf_prototypes, "NMF")
        df_long = pd.concat([df_mean, df_nmf], axis=0, ignore_index=True)
        plot_similarity_distribution(ax_b, df_long)
        sim_stats = {"Mean": stat_mean, "NMF": stat_nmf}
        print(f"[Stats] Mean similarity: {stat_mean}")
        print(f"[Stats] NMF similarity: {stat_nmf}")
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "(b) target labels unavailable\nsimilarity distribution skipped", ha="center", va="center")

    # (c) confusion matrix
    spec_c = outer[1, 0]
    if tgt_features is not None and tgt_labels is not None:
        pred_mean, metrics_mean, cm_mean, margins_mean = classify_by_prototypes(tgt_features, tgt_labels, mean_prototypes)
        pred_nmf, metrics_nmf, cm_nmf, margins_nmf = classify_by_prototypes(tgt_features, tgt_labels, nmf_prototypes)
        inner_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec_c, wspace=0.25)
        ax_c1 = fig.add_subplot(inner_c[0, 0])
        ax_c2 = fig.add_subplot(inner_c[0, 1])
        plot_confusion_matrices(
            ax_left=ax_c1,
            ax_right=ax_c2,
            cm_mean=cm_mean,
            cm_nmf=cm_nmf,
            acc_mean=metrics_mean["accuracy"],
            acc_nmf=metrics_nmf["accuracy"],
        )
        ax_c1.text(-0.40, 1.10, "(c) Prototype-based Confusion Matrix", transform=ax_c1.transAxes, fontsize=11, fontweight="bold")
        print(f"[Metrics] Mean prototype: {metrics_mean}")
        print(f"[Metrics] NMF prototype : {metrics_nmf}")
    else:
        ax_c = fig.add_subplot(spec_c)
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "(c) target labels unavailable\nconfusion matrix skipped", ha="center", va="center")

    # (d) activation examples (optional)
    spec_d = outer[1, 1]
    if (
        tgt_set is not None
        and tgt_set.images is not None
        and tgt_set.feature_maps is not None
        and tgt_labels is not None
        and pred_nmf is not None
        and margins_nmf is not None
        and tgt_set.feature_maps.ndim == 4
        and tgt_set.feature_maps.shape[1] > 1
    ):
        activation_meta = plot_activation_examples(
            fig=fig,
            spec=spec_d,
            tgt_images=tgt_set.images,
            tgt_feature_maps=tgt_set.feature_maps,
            tgt_labels=tgt_labels,
            mean_prototypes=mean_prototypes,
            nmf_prototypes=nmf_prototypes,
            margins_nmf=margins_nmf,
            pred_nmf=pred_nmf,
        )
    else:
        ax_d = fig.add_subplot(spec_d)
        ax_d.axis("off")
        ax_d.text(
            0.5,
            0.5,
            "(d) Activation examples skipped:\nfeature maps unavailable or incompatible",
            ha="center",
            va="center",
        )
        activation_meta = {"status": "skipped_unavailable_or_incompatible"}

    fig.suptitle("Comparison between Mean Prototypes and NMF Prototypes", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(args.outdir, "figureX_prototype_comparison.png")
    out_pdf = os.path.join(args.outdir, "figureX_prototype_comparison.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] figure png: {out_png}")
    print(f"[Done] figure pdf: {out_pdf}")

    # Extra metrics
    proto_mean_sparsity = float(np.mean([hoyer_sparsity(mean_prototypes[c]) for c in range(mean_prototypes.shape[0])]))
    proto_nmf_sparsity = float(np.mean([hoyer_sparsity(nmf_prototypes[c]) for c in range(nmf_prototypes.shape[0])]))
    mean_sep = float(1.0 - np.dot(_l2n(mean_prototypes)[0], _l2n(mean_prototypes)[1])) if mean_prototypes.shape[0] >= 2 else float("nan")
    nmf_sep = float(1.0 - np.dot(_l2n(nmf_prototypes)[0], _l2n(nmf_prototypes)[1])) if nmf_prototypes.shape[0] >= 2 else float("nan")

    metrics = {
        "config": {
            "ckpt": args.ckpt,
            "src_root": args.src_root,
            "src_csv": args.src_csv,
            "tgt_root": args.tgt_root,
            "tgt_csv": args.tgt_csv,
            "backbone": args.backbone,
            "resize": int(args.resize),
            "batch_size": int(args.batch_size),
            "plane": args.plane,
            "nmf_components": int(args.nmf_components),
            "feature_layer": args.feature_layer,
        },
        "shape": {
            "src_features": list(src_features.shape),
            "mean_prototypes": list(mean_prototypes.shape),
            "nmf_prototypes": list(nmf_prototypes.shape),
            "tgt_features": list(tgt_features.shape) if tgt_features is not None else None,
            "tgt_feature_maps": list(tgt_set.feature_maps.shape) if (tgt_set is not None and tgt_set.feature_maps is not None) else None,
        },
        "prototype_sparsity_hoyer": {"Mean": proto_mean_sparsity, "NMF": proto_nmf_sparsity},
        "class_separation_cosine_distance": {"Mean": mean_sep, "NMF": nmf_sep},
        "target_similarity_stats": sim_stats,
        "target_classification_metrics": {"Mean": metrics_mean, "NMF": metrics_nmf},
        "nonnegative_transform_for_sklearn_audit": nmf_sklearn_info,
        "project_vs_sklearn_proto_cosine": {
            "class_0": float(np.dot(_l2n(nmf_prototypes)[0], _l2n(nmf_prototypes_sklearn)[0])) if nmf_prototypes.shape[0] > 0 else float("nan"),
            "class_1": float(np.dot(_l2n(nmf_prototypes)[1], _l2n(nmf_prototypes_sklearn)[1])) if nmf_prototypes.shape[0] > 1 else float("nan"),
        },
        "figure_meta": {"heatmap": heat_meta, "activation_examples": activation_meta},
    }

    out_json = os.path.join(args.outdir, "prototype_metrics.json")
    save_metrics(out_json, metrics)
    print(f"[Done] metrics json: {out_json}")


if __name__ == "__main__":
    main()

