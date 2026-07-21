# -*- coding: utf-8 -*-
"""
calib.py
温度缩放（Temperature Scaling）与带温度的阈值选择/评估工具。
可独立于训练脚本使用：在验证集上拟合 T，然后用该 T 在验证/测试集上评估。
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits_labels(model: torch.nn.Module,
                          ds,
                          device: str = "cuda",
                          batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集收集 logits 和 labels（放在 CPU），用于温度拟合。
    Returns:
        logits: FloatTensor [N, C]  (CPU)
        labels: LongTensor  [N]     (CPU)
    """
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)  # [B, C]
        all_logits.append(logits.detach().cpu())
        all_y.append(yb)
    logits = torch.cat(all_logits, dim=0)          # [N, C] on CPU
    labels = torch.cat(all_y, dim=0).long().cpu()  # [N]    on CPU
    return logits, labels


def learn_temperature(model: torch.nn.Module,
                      ds_val,
                      device: str = "cuda",
                      init_T: float = 1.0,
                      use_lbfgs: bool = True,
                      max_iter: int = 200,
                      adam_lr: float = 1e-2,
                      adam_steps: int = 200,
                      batch_size: int = 64) -> float:
    """
    在验证集上拟合“标量温度 T”（>=1e-3），最小化交叉熵（NLL）。
    返回：float T
    """
    logits, labels = collect_logits_labels(model, ds_val, device, batch_size)  # CPU tensors
    T = torch.tensor([float(init_T)], dtype=torch.float32, requires_grad=True)

    if use_lbfgs:
        opt = torch.optim.LBFGS([T], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels)
            loss.backward()
            return loss

        opt.step(closure)
    else:
        opt = torch.optim.Adam([T], lr=adam_lr)
        for _ in range(adam_steps):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels)
            loss.backward()
            opt.step()

    T_fit = float(T.detach().clamp_min(1e-3).item())
    print(f"[Calib] learned temperature T = {T_fit:.4f}")
    return T_fit


@torch.no_grad()
def evaluate_calibrated(model: torch.nn.Module,
                        ds,
                        device: str = "cuda",
                        threshold: float = 0.5,
                        tag: str = "val",
                        temperature: Optional[float] = None,
                        batch_size: int = 64) -> Dict[str, float]:
    """
    使用温度缩放后的概率进行评估（不修改原 evaluate，可直接替代调用）。
    返回与你原 evaluate 相同风格的字典，并额外不返回 p1/y（保持轻量）。
    """
    if ds is None or len(ds) == 0:
        return None

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ce = nn.CrossEntropyLoss()
    model.eval()

    all_p1, all_y = [], []
    loss_sum = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        if temperature is not None:
            logits = logits / float(temperature)
        loss = ce(logits, yb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        all_p1.append(p1.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
        loss_sum += float(loss.item())

    p1 = np.concatenate(all_p1)
    y  = np.concatenate(all_y).astype(int)

    # AUC
    try:
        auc = float(metrics.roc_auc_score(y, p1))
    except Exception:
        auc = 0.5

    pred = (p1 >= threshold).astype(int)
    acc  = float((pred == y).mean())
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)

    cm = metrics.confusion_matrix(y, pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0

    return {
        f"{tag}_loss": loss_sum / max(len(dl), 1),
        f"{tag}_auc":  auc,
        f"{tag}_acc":  acc,
        f"{tag}_prec": float(prec),
        f"{tag}_rec":  float(rec),
        f"{tag}_spec": spec,
        f"{tag}_f1":   float(f1),
    }


@torch.no_grad()
def choose_best_threshold_calibrated(model: torch.nn.Module,
                                     ds_val,
                                     device: str = "cuda",
                                     mode: str = "youden",   # 'acc' | 'f1' | 'youden' | 'custom'
                                     grid_step: float = 0.005,
                                     thr_min: float = 0.0,
                                     thr_max: float = 1.0,
                                     pos_weight: float = 1.0,
                                     spec_weight: float = 1.0,
                                     temperature: Optional[float] = None,
                                     batch_size: int = 64) -> Tuple[float, Dict[str, float]]:
    """
    使用“温度缩放后的概率”在验证集上选择最佳阈值。
    返回：(best_thr, stats_dict)
    """
    dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()

    all_p1, all_y = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        if temperature is not None:
            logits = logits / float(temperature)
        p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_p1.append(p1)
        all_y.append(yb.numpy())

    p1 = np.concatenate(all_p1)
    y  = np.concatenate(all_y).astype(int)

    thr_grid = np.arange(thr_min, thr_max + 1e-9, grid_step)
    best_thr, best_score, best_stats = 0.5, -1.0, {}

    for t in thr_grid:
        pred = (p1 >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()

        acc  = (tp + tn) / max(len(y), 1)
        rec  = tp / max(tp + fn, 1)            # TPR
        spec = tn / max(tn + fp, 1)            # TNR
        f1   = 0.0 if tp == 0 else (2 * tp) / max(2 * tp + fp + fn, 1)

        if mode == 'acc':
            score = acc
        elif mode == 'f1':
            score = f1
        elif mode == 'youden':
            score = rec + spec - 1.0
        else:  # 'custom'
            score = pos_weight * rec + spec_weight * spec

        if score > best_score:
            best_score = score
            best_thr = float(t)
            best_stats = dict(acc=float(acc), rec=float(rec), spec=float(spec), f1=float(f1))

    print(f"[Val-Calib] best_thr={best_thr:.3f} by {mode} (T={temperature if temperature is not None else 1.0}) | stats={best_stats}")
    return best_thr, best_stats
