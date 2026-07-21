#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition._nmf import non_negative_factorization
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_net import build_custom_model
from data import NPYSliceDataset
from nmf_lib_assign import minmax_fit_transform_together
from uda_core.prototypes import PrototypeBank

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Visualize NMF responsibility vs correctness")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nmf_path", default="", help="Optional .npz prototype file. If empty, try loading from ckpt.")
    ap.add_argument("--tgt_root", required=True)
    ap.add_argument("--tgt_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--id_col", default="case_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--plane", default="sagittal", choices=["sagittal", "coronal", "axial"])
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--single_file_case", action="store_true", default=False)
    ap.add_argument("--id_zero_pad", type=int, default=0)
    ap.add_argument("--temperature", default="auto", help="auto or positive float")
    ap.add_argument("--beta_loss", default="frobenius", choices=["frobenius", "kullback-leibler", "itakura-saito"])
    ap.add_argument("--nmf_assign_iters", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pretrained", default="imagenet")
    return ap.parse_args()


def _to_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and "cuda" in str(device_str):
        return torch.device(device_str)
    return torch.device("cpu")


def load_model(ckpt_path: str, backbone: str, num_classes: int, pretrained: str, device: torch.device):
    method = backbone.replace("custom_", "") if backbone.startswith("custom_") else backbone
    model = build_custom_model(method=method, num_classes=num_classes, pretrained=pretrained, device=str(device))
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = sd["model_state_dict"]
    if not isinstance(sd, dict):
        raise RuntimeError("Unsupported checkpoint format: state_dict not found.")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Init] model loaded | missing={len(getattr(msg, 'missing_keys', []))} unexpected={len(getattr(msg, 'unexpected_keys', []))}")
    model.eval()
    return model


def _npz_get_first(npz_obj, names: Sequence[str]):
    for n in names:
        if n in npz_obj:
            return npz_obj[n]
    return None


def _infer_offsets(mu_rows: int, num_classes: int) -> List[Tuple[int, int]]:
    if mu_rows < num_classes:
        raise ValueError(f"Cannot infer per-class prototype slices: total prototypes={mu_rows} < num_classes={num_classes}.")
    if mu_rows % num_classes != 0:
        raise ValueError(
            f"Cannot infer per-class prototype slices: total prototypes={mu_rows} not divisible by num_classes={num_classes}. "
            "Provide offsets/per_class_K in nmf file."
        )
    k = mu_rows // num_classes
    offsets: List[Tuple[int, int]] = []
    s = 0
    for _ in range(num_classes):
        e = s + k
        offsets.append((s, e))
        s = e
    return offsets


def load_nmf_prototypes(nmf_path: str, ckpt_path: str, num_classes: int) -> Tuple[np.ndarray, List[Tuple[int, int]], str]:
    source = "unknown"
    mu = None
    offsets = None

    if nmf_path:
        if not os.path.isfile(nmf_path):
            raise FileNotFoundError(f"--nmf_path not found: {nmf_path}")
        npz = np.load(nmf_path, allow_pickle=True)
        mu = _npz_get_first(npz, ["mu", "prototypes", "centers", "H", "nmf_prototypes"])
        off = _npz_get_first(npz, ["offsets", "proto_offsets"])
        per_k = _npz_get_first(npz, ["per_class_K", "class_K", "k_per_class"])
        if off is not None:
            offsets = [(int(a), int(b)) for a, b in np.asarray(off)]
        elif per_k is not None:
            per_k = [int(x) for x in np.asarray(per_k).tolist()]
            s = 0
            offsets = []
            for k in per_k:
                offsets.append((s, s + int(k)))
                s += int(k)
        source = f"npz:{nmf_path}"
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for k in ["proto_mu", "mu", "prototype_mu", "nmf_mu"]:
                if k in ckpt:
                    v = ckpt[k]
                    mu = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                    break
            if "proto_offsets" in ckpt:
                offsets = [(int(a), int(b)) for a, b in np.asarray(ckpt["proto_offsets"])]
            elif "offsets" in ckpt:
                offsets = [(int(a), int(b)) for a, b in np.asarray(ckpt["offsets"])]
            elif "per_class_K" in ckpt:
                per_k = [int(x) for x in np.asarray(ckpt["per_class_K"]).tolist()]
                s = 0
                offsets = []
                for k in per_k:
                    offsets.append((s, s + int(k)))
                    s += int(k)
        source = f"ckpt:{ckpt_path}"

    if mu is None:
        raise RuntimeError("NMF prototypes not found. Please pass --nmf_path (.npz) or provide checkpoint containing prototype tensors.")
    mu = np.asarray(mu, dtype=np.float32)
    if mu.ndim != 2:
        raise ValueError(f"Prototype matrix must be 2D, got shape={mu.shape}.")
    if offsets is None:
        offsets = _infer_offsets(mu.shape[0], num_classes)
    if len(offsets) != num_classes:
        raise ValueError(f"Prototype class slices mismatch: len(offsets)={len(offsets)} vs num_classes={num_classes}.")
    return mu, offsets, source


def build_target_loader(args: argparse.Namespace):
    ds = NPYSliceDataset(
        npy_root=args.tgt_root,
        csv_file=args.tgt_csv,
        plane=args.plane,
        id_col=args.id_col,
        label_col=args.label_col,
        resize=args.resize,
        single_file_case=bool(args.single_file_case),
        id_zero_pad=(None if int(args.id_zero_pad) <= 0 else int(args.id_zero_pad)),
        augment=False,
        return_case_id=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return ds, dl


def classwise_recon_errors(
    feat_np: np.ndarray,
    mu: np.ndarray,
    offsets: Sequence[Tuple[int, int]],
    beta_loss: str,
    iters: int,
) -> np.ndarray:
    errs = []
    for (s, e) in offsets:
        Hc = mu[s:e]
        Xs, Cs = minmax_fit_transform_together(feat_np, Hc)
        W_init = np.maximum(Xs @ Cs.T, 1e-6)
        W, H_fix, _ = non_negative_factorization(
            Xs,
            W=W_init,
            H=Cs,
            init="custom",
            update_H=False,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=iters,
            tol=1e-6,
            random_state=0,
        )
        rec = W @ H_fix
        err = ((Xs - rec) ** 2).sum(axis=1)
        errs.append(err.astype(np.float32))
    return np.stack(errs, axis=1)


def stable_softmax_neg_err(errs: np.ndarray, temperature: float) -> np.ndarray:
    z = -errs / max(float(temperature), 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def _summary_vec(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan}
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.nanmedian(x)),
        "iqr": float(q3 - q1),
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if sp <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / sp)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def save_fig(fig, out_base: str):
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(df: pd.DataFrame, outdir: str):
    correct = df[df["correct"] == 1]
    wrong = df[df["correct"] == 0]

    fig1, ax1 = plt.subplots(figsize=(6.2, 4.6))
    ax1.boxplot([correct["nmf_resp_pred"], wrong["nmf_resp_pred"]], labels=["Correctly classified", "Misclassified"], showmeans=True)
    ax1.set_ylabel("NMF responsibility for predicted class")
    ax1.set_title("Predicted-class NMF Responsibility")
    save_fig(fig1, os.path.join(outdir, "nmf_resp_pred_correct_vs_wrong"))

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.6))
    ax2.boxplot([correct["nmf_margin"], wrong["nmf_margin"]], labels=["Correctly classified", "Misclassified"], showmeans=True)
    ax2.set_ylabel("NMF reconstruction margin")
    ax2.set_title("NMF Reconstruction Margin")
    save_fig(fig2, os.path.join(outdir, "nmf_margin_correct_vs_wrong"))

    fig3, ax3 = plt.subplots(figsize=(6.6, 5.0))
    ax3.scatter(correct["pred_prob"], correct["nmf_resp_pred"], s=16, alpha=0.65, c="#1f77b4", label="correct")
    ax3.scatter(wrong["pred_prob"], wrong["nmf_resp_pred"], s=16, alpha=0.75, c="#d62728", marker="x", label="wrong")
    ax3.set_xlabel("Classifier confidence")
    ax3.set_ylabel("NMF responsibility for predicted class")
    ax3.set_title("Classifier Confidence vs NMF Responsibility")
    ax3.legend(frameon=False)
    save_fig(fig3, os.path.join(outdir, "classifier_confidence_vs_nmf_responsibility"))

    classes = sorted(df["label"].unique().tolist())
    fig4, axes = plt.subplots(1, max(len(classes), 1), figsize=(6.3 * max(len(classes), 1), 4.6), squeeze=False)
    for i, c in enumerate(classes):
        sub = df[df["label"] == c]
        sub_ok = sub[sub["correct"] == 1]["nmf_resp_true"].to_numpy()
        sub_ng = sub[sub["correct"] == 0]["nmf_resp_true"].to_numpy()
        axes[0, i].boxplot([sub_ok, sub_ng], labels=["Correct", "Wrong"], showmeans=True)
        axes[0, i].set_title(f"True class {c}")
        axes[0, i].set_ylabel("NMF responsibility for true class")
    save_fig(fig4, os.path.join(outdir, "nmf_resp_true_by_class"))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = _to_device(args.device)

    model = load_model(args.ckpt, args.backbone, int(args.num_classes), args.pretrained, device)
    _, tgt_loader = build_target_loader(args)
    mu_np, offsets, proto_source = load_nmf_prototypes(args.nmf_path, args.ckpt, int(args.num_classes))

    feat_dim = int(getattr(model, "feat_dim", mu_np.shape[1]))
    if mu_np.shape[1] != feat_dim:
        raise ValueError(f"Feature/prototype dimension mismatch: model feat_dim={feat_dim}, prototype_dim={mu_np.shape[1]}")

    proto = PrototypeBank(
        num_classes=int(args.num_classes),
        feat_dim=feat_dim,
        K=None,
        Kmax=4,
        proto_m=0.95,
        temp_proto=0.1,
        device=str(device),
    )
    proto.mu = torch.from_numpy(mu_np).to(device)
    proto.offsets = [(int(s), int(e)) for (s, e) in offsets]
    proto.per_class_K = [int(e - s) for (s, e) in offsets]

    print(f"[Init] prototype source={proto_source} | mu={mu_np.shape} | offsets={proto.offsets}")

    rows: List[Dict] = []
    all_errs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tgt_loader:
            xb, yb, cid = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if hasattr(model, "forward_with_feat"):
                logits, feat = model.forward_with_feat(xb)
            else:
                logits = model(xb)
                feat = logits

            prob = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            pred = np.argmax(prob, axis=1).astype(int)
            pred_prob = np.max(prob, axis=1).astype(np.float32)
            label = yb.detach().cpu().numpy().astype(int)

            _, p_cls = proto.nmf_assign(feat, beta_loss=args.beta_loss, iters=int(args.nmf_assign_iters))
            _ = p_cls.detach().cpu().numpy()  # explicit reuse of project-native nmf assignment path

            feat_np = feat.detach().cpu().numpy().astype(np.float32)
            errs = classwise_recon_errors(feat_np, mu_np, offsets, beta_loss=args.beta_loss, iters=int(args.nmf_assign_iters))
            all_errs.append(errs)

            for i in range(feat_np.shape[0]):
                case_i = str(cid[i])
                row = {
                    "case_id": case_i,
                    "label": int(label[i]),
                    "pred_label": int(pred[i]),
                    "pred_prob": float(pred_prob[i]),
                    "correct": int(pred[i] == label[i]),
                }
                for c in range(int(args.num_classes)):
                    row[f"nmf_err_class{c}"] = float(errs[i, c])
                rows.append(row)

    if not rows:
        raise RuntimeError("No target samples were processed.")

    err_mat = np.concatenate(all_errs, axis=0)
    if str(args.temperature).lower() == "auto":
        temp = float(np.median(err_mat))
        if not np.isfinite(temp) or temp <= 1e-8:
            temp = float(np.mean(err_mat))
        if not np.isfinite(temp) or temp <= 1e-8:
            temp = 1.0
        temp_mode = "auto(median_err)"
    else:
        temp = float(args.temperature)
        if temp <= 0:
            raise ValueError("--temperature must be positive or 'auto'.")
        temp_mode = "manual"

    resp_mat = stable_softmax_neg_err(err_mat, temperature=temp)
    for i, row in enumerate(rows):
        pred = int(row["pred_label"])
        true = int(row["label"])
        row["nmf_resp_pred"] = float(resp_mat[i, pred])
        row["nmf_resp_true"] = float(resp_mat[i, true])
        row["nmf_pred_by_recon"] = int(np.argmin(err_mat[i]))
        row["nmf_correct"] = int(row["nmf_pred_by_recon"] == true)
        if int(args.num_classes) == 2:
            other = 1 - pred
            row["nmf_margin"] = float(err_mat[i, other] - err_mat[i, pred])
        else:
            other_min = np.min(np.delete(err_mat[i], pred))
            row["nmf_margin"] = float(other_min - err_mat[i, pred])
        row["responsibility_definition"] = "A: softmax(-classwise_reconstruction_error / T)"

    df = pd.DataFrame(rows)
    ordered_cols = ["case_id", "label", "pred_label", "pred_prob", "correct", "nmf_resp_pred", "nmf_resp_true"]
    ordered_cols += [f"nmf_err_class{c}" for c in range(int(args.num_classes))]
    ordered_cols += ["nmf_margin", "nmf_pred_by_recon", "nmf_correct"]
    df = df[ordered_cols]
    out_csv = os.path.join(args.outdir, "nmf_responsibility_per_case.csv")
    df.to_csv(out_csv, index=False)

    correct_mask = df["correct"].to_numpy(dtype=int) == 1
    wrong_mask = ~correct_mask
    x_resp_ok = df.loc[correct_mask, "nmf_resp_pred"].to_numpy(dtype=np.float64)
    x_resp_ng = df.loc[wrong_mask, "nmf_resp_pred"].to_numpy(dtype=np.float64)
    x_mgn_ok = df.loc[correct_mask, "nmf_margin"].to_numpy(dtype=np.float64)
    x_mgn_ng = df.loc[wrong_mask, "nmf_margin"].to_numpy(dtype=np.float64)

    p_resp = np.nan
    p_margin = np.nan
    if mannwhitneyu is not None and x_resp_ok.size > 0 and x_resp_ng.size > 0:
        p_resp = float(mannwhitneyu(x_resp_ok, x_resp_ng, alternative="two-sided").pvalue)
    if mannwhitneyu is not None and x_mgn_ok.size > 0 and x_mgn_ng.size > 0:
        p_margin = float(mannwhitneyu(x_mgn_ok, x_mgn_ng, alternative="two-sided").pvalue)

    auroc_resp = np.nan
    auroc_margin = np.nan
    y01 = df["correct"].to_numpy(dtype=int)
    try:
        if len(np.unique(y01)) == 2:
            auroc_resp = float(roc_auc_score(y01, df["nmf_resp_pred"].to_numpy(dtype=np.float64)))
            auroc_margin = float(roc_auc_score(y01, df["nmf_margin"].to_numpy(dtype=np.float64)))
    except Exception:
        pass

    summary = {
        "num_samples": int(len(df)),
        "num_correct": int(correct_mask.sum()),
        "num_wrong": int(wrong_mask.sum()),
        "temperature": float(temp),
        "temperature_mode": temp_mode,
        "prototype_source": proto_source,
        "prototype_shape": list(mu_np.shape),
        "prototype_offsets": [[int(s), int(e)] for s, e in offsets],
        "responsibility_definition": "A: resp_c = softmax(-err_c / T), err_c from class-wise nonnegative reconstruction with fixed class dictionary",
        "nmf_resp_pred_correct_stats": _summary_vec(x_resp_ok),
        "nmf_resp_pred_wrong_stats": _summary_vec(x_resp_ng),
        "nmf_margin_correct_stats": _summary_vec(x_mgn_ok),
        "nmf_margin_wrong_stats": _summary_vec(x_mgn_ng),
        "p_value_mannwhitney_resp_pred": p_resp,
        "p_value_mannwhitney_margin": p_margin,
        "cohens_d_resp_pred_correct_minus_wrong": cohen_d(x_resp_ok, x_resp_ng),
        "cohens_d_margin_correct_minus_wrong": cohen_d(x_mgn_ok, x_mgn_ng),
        "cliffs_delta_resp_pred_correct_vs_wrong": cliffs_delta(x_resp_ok, x_resp_ng),
        "cliffs_delta_margin_correct_vs_wrong": cliffs_delta(x_mgn_ok, x_mgn_ng),
        "auroc_resp_pred_for_correctness": auroc_resp,
        "auroc_margin_for_correctness": auroc_margin,
    }

    out_json = os.path.join(args.outdir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_plots(df, args.outdir)

    print("[Done] saved:")
    print(f"  - {out_csv}")
    print(f"  - {out_json}")
    print(f"  - {os.path.join(args.outdir, 'nmf_resp_pred_correct_vs_wrong.png/.pdf')}")
    print(f"  - {os.path.join(args.outdir, 'nmf_margin_correct_vs_wrong.png/.pdf')}")
    print(f"  - {os.path.join(args.outdir, 'classifier_confidence_vs_nmf_responsibility.png/.pdf')}")
    print(f"  - {os.path.join(args.outdir, 'nmf_resp_true_by_class.png/.pdf')}")
    if int(summary["num_wrong"]) < 10:
        print("[Warn] misclassified samples are few (<10); distribution plots/statistics may be unstable.")
    print("[Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
