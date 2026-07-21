# -*- coding: utf-8 -*-
"""
Full-supervised training on labeled Source + Target (train/val), then evaluate on Target test.
在源域与目标域均有标签的前提下：拼接训练集，优先使用“目标验证集”选择最佳模型（若提供），
否则退回使用源验证集。最终在目标域测试集上评估并打印完整指标。

可复现版本要点：
  - 统一随机种子与环境：PYTHONHASHSEED、CUBLAS_WORKSPACE_CONFIG、(O|MKL)MP 线程、torch/cudnn 设置
  - DataLoader：固定 generator + worker_init_fn（按 worker_id 偏移），并使用 persistent_workers
  - 如遇未实现确定性的算子，torch 将抛错，便于定位；必要时可将 use_deterministic_algorithms 改为 warn_only=True

阈值选择：
  - 训练完成后，若提供源域验证集（src_val），将基于 src_val 选阈值（youden/f1/sp95），并覆盖 args.thr，再在目标测试集评估。
"""

import os
import json
import argparse
import random
import torch
from copy import deepcopy
import numpy as np
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import (
    resnet18, resnet50, alexnet,
    ResNet18_Weights, ResNet50_Weights, AlexNet_Weights
)
from data import NPYSliceDataset  # 与项目一致
from custom_net import build_custom_model


def _prf_by_average(y, pred):
    out = {}
    for avg in ("binary", "macro", "weighted"):
        kwargs = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kwargs["pos_label"] = 1
        p, r, f1, _ = metrics.precision_recall_fscore_support(y, pred, **kwargs)
        out[f"Prec_{avg}"] = float(p)
        out[f"Rec_{avg}"] = float(r)
        out[f"F1_{avg}"] = float(f1)
    return out


# ----------------- Determinism helpers -----------------
def setup_determinism(seed: int = 42):
    """全局可复现设置：必须在任何 torch/cuda 操作前调用"""
    # 1) 环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS 确定性：在程序启动早期设置；:4096:8 一般更快，:16:8 更省显存
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # 避免 CPU/OpenMP 聚合的非确定性（可按需调大，但 1 是最稳妥）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 2) Python / NumPy / Torch 种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3) cuDNN / 算法层面
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 如遇未实现确定性的算子，这里会抛错，便于尽早发现问题
    torch.use_deterministic_algorithms(True)  # 如需兼容性，可改为 warn_only=True

    # 4) 限制 CPU 线程，进一步稳定（可按需调大）
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def make_dataloader(
    ds,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    base_seed: int,
):
    """使用固定 generator 与 worker_init_fn 的 DataLoader 构造器"""
    if ds is None:
        return None

    # 让 DataLoader 的打乱顺序可复现
    g = torch.Generator()
    g.manual_seed(base_seed)

    # 每个 worker 有不同但可复现的种子
    def seed_worker(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # 多进程时，避免每个 epoch 反复重建 worker 导致隐式随机性
    persistent = num_workers > 0

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent,
        drop_last=False,
    )


# ----------------- model -----------------
def build_model(backbone='resnet50', num_classes=2, pretrained='imagenet', device='cuda'):
    if backbone.startswith('custom_'):
        method = backbone.replace('custom_', '')
        return build_custom_model(method=method, num_classes=num_classes,
                                  pretrained=pretrained, device=device)
    use_imagenet = isinstance(pretrained, str) and pretrained.lower() == 'imagenet'
    if backbone == 'resnet50':
        m = resnet50(weights=ResNet50_Weights.DEFAULT if use_imagenet else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == 'resnet18':
        m = resnet18(weights=ResNet18_Weights.DEFAULT if use_imagenet else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone in ('alexnet_mrnet', 'alexnet'):
        base = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if use_imagenet else None)
        m = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m.to(device)


@torch.no_grad()
def evaluate_auc_acc(model, ds, device='cuda', thr=0.5, seed=42):
    """
    评估函数：计算数据集的 val_loss / AUC / ACC（用于选择best与调度器）。
    使用确定性 DataLoader（shuffle=False），仍传入固定 generator 以避免潜在随机算子。
    """
    if ds is None or len(ds) == 0:
        return None

    dl = make_dataloader(
        ds, batch_size=64, shuffle=False, num_workers=0,  # eval 用单线程最稳妥
        pin_memory=True, base_seed=seed
    )
    ce = nn.CrossEntropyLoss()
    model.eval()

    all_p1, all_y = [], []
    val_loss_sum = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        all_p1.append(p1.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
        val_loss_sum += float(loss.item())

    p1 = np.concatenate(all_p1); y = np.concatenate(all_y)
    try:
        auc = float(metrics.roc_auc_score(y, p1))
    except Exception as e:
        auc = 0.5
        print(f"Warning: AUC calculation failed. Error: {e}")
    acc = float(((p1 >= thr).astype(int) == y).mean())
    return {"loss": val_loss_sum / max(len(dl), 1), "auc": auc, "acc": acc}


@torch.no_grad()
def evaluate_full_metrics(model, ds, device='cuda', thr=0.5, seed=42):
    """
    在目标域测试集上输出完整指标：AUC/ACC/Prec/Rec/Spec/F1/Brier + 混淆矩阵。
    使用确定性 DataLoader（shuffle=False）。
    """
    if ds is None or len(ds) == 0:
        return None

    dl = make_dataloader(
        ds, batch_size=64, shuffle=False, num_workers=0,
        pin_memory=True, base_seed=seed
    )
    model.eval()
    all_p1, all_y = [], []
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1]
        all_p1.append(p1.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
    p1 = np.concatenate(all_p1); y = np.concatenate(all_y)

    # 主指标
    auc = float(metrics.roc_auc_score(y, p1)) if len(np.unique(y)) == 2 else np.nan
    yhat = (p1 >= thr).astype(int)
    acc  = metrics.accuracy_score(y, yhat)
    prf = _prf_by_average(y, yhat)

    brier = float(np.mean((p1 - y)**2))
    tn, fp, fn, tp = metrics.confusion_matrix(y, yhat, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "AUC": auc, "ACC": acc,
        "Prec": prf["Prec_weighted"], "Rec": prf["Rec_weighted"], "Spec": spec, "F1": prf["F1_weighted"],
        **prf,
        "Brier": brier, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }


# ----------------- 阈值选择（基于验证集） -----------------
@torch.no_grad()
def pick_threshold_on_val(model, ds_val, device='cuda',
                          mode='f1',           # 'youden' | 'f1' | 'sp95'
                          grid_step=0.005,         # F1 网格步长
                          target_spec=0.95,        # SP95 目标特异度
                          seed=42):
    """
    在验证集上为“当前模型+当前预处理”选阈值。
    返回: (best_thr, info_dict)
    """
    if ds_val is None or len(ds_val) == 0:
        return None, {"msg": "no validation set"}

    dl = make_dataloader(ds_val, batch_size=64, shuffle=False,
                         num_workers=0, pin_memory=True, base_seed=seed)
    model.eval()
    import numpy as np
    from sklearn import metrics
    p1s, ys = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        p1s.append(p1); ys.append(yb.numpy())
    p1 = np.concatenate(p1s); y = np.concatenate(ys).astype(int)

    info = {}
    if mode == 'youden':
        fpr, tpr, thr = metrics.roc_curve(y, p1)
        j = tpr - fpr
        best_idx = int(j.argmax())
        best_thr = float(thr[best_idx])
        info.update(dict(tpr=float(tpr[best_idx]), fpr=float(fpr[best_idx])))

    elif mode == 'f1':
        grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
        best_f1, best_thr = -1.0, 0.5
        for t in grid:
            yhat = (p1 >= t).astype(int)
            f1 = metrics.f1_score(y, yhat, average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
        info.update(dict(best_f1=float(best_f1), step=float(grid_step)))

    elif mode == 'sp95':
        fpr, tpr, thr = metrics.roc_curve(y, p1)
        spec = 1.0 - fpr
        ok = np.where(spec >= float(target_spec))[0]
        if len(ok) == 0:
            # 达不到目标特异度，取最保守阈值（极大阈值）
             best_idx = int(np.argmax(spec))   
        else:
            # 在满足 spec≥目标 的集合中选择 TPR 最大的
            best_idx = ok[np.argmax(tpr[ok])]
        best_thr = float(thr[best_idx])
        info.update(dict(spec=float(spec[best_idx]), tpr=float(tpr[best_idx]),
                         target_spec=float(target_spec)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    info['mode'] = mode
    info['best_thr'] = float(best_thr)
    return best_thr, info


def train_full_supervised(model,
                          ds_src_train,
                          ds_tgt_train,
                          ds_src_val=None,
                          ds_tgt_val=None,
                          batch_size=32, epochs=0,
                          lr=1e-4, weight_decay=1e-4,
                          device='cuda', save_dir=None, backbone_name='resnet50',
                          select_thr=0.5, seed=42):
    """
    全监督训练：将源域与目标域训练集拼接训练。
    优先用“目标验证集”选择最佳模型（若提供），否则用“源验证集”。
    所有随机性（shuffle/augment/初始化）均受 seed 控制。
    """
    # 拼接训练集
    if ds_tgt_train is None or len(ds_tgt_train) == 0:
        ds_train = ds_src_train
    elif ds_src_train is None or len(ds_src_train) == 0:
        ds_train = ds_tgt_train
    else:
        ds_train = ConcatDataset([ds_src_train, ds_tgt_train])

    # 训练 DataLoader：固定 generator + worker 种子；多进程且持久化 worker
    dl = make_dataloader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, base_seed=seed
    )

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 选择验证集：优先目标验证集
    val_pref = 'tgt' if (ds_tgt_val is not None and len(ds_tgt_val) > 0) else 'src'
    ds_val = ds_tgt_val if val_pref == 'tgt' else ds_src_val

    # 调度器基于“所选的验证集”的AUC
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=3, verbose=True)

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, steps = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()); steps += 1

        epoch_loss = loss_sum / max(steps, 1)
        print(f"[Train@FullSup] epoch {ep}/{epochs}  loss={epoch_loss:.4f}")

        # 验证
        if ds_val is not None:
            res = evaluate_auc_acc(model, ds_val, device=device, thr=select_thr, seed=seed)
            val_auc, val_acc, val_loss = res['auc'], res['acc'], res['loss']
            print(f"[Val/{val_pref}] epoch {ep}: val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}")
            scheduler.step(val_auc)

            # 记录best
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = deepcopy(model.state_dict())
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(save_dir, f"{backbone_name}_best.pth"))
                    print(f"[Save] update best model (val_auc={best_auc:.4f})")
        else:
            # 没有验证集则按训练 loss 的反向趋势简单保存（不推荐）
            if save_dir is not None and (ep == epochs):
                os.makedirs(save_dir, exist_ok=True)
                best_state = deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(save_dir, f"{backbone_name}_best.pth"))
                print(f"[Save] saved last as best (no val set)")

    # 结束后：如有best则加载
    if best_state is not None:
        model.load_state_dict(best_state)

    # 另存“最后一轮”权重
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{backbone_name}_last.pth"))
        print("[Save] last epoch weights saved.")
    return model


def main():
    ap = argparse.ArgumentParser("Full-supervised training on labeled Source+Target, then evaluate on Target test")

    # === 源域（必需：train；可选：val） ===
    ap.add_argument('--npyfile_src', type=str, required=True)
    ap.add_argument('--src_csv',     type=str, required=True)      # 源域训练集 CSV
    ap.add_argument('--src_val_csv', type=str, default=None)       # 源域验证集 CSV

    # === 目标域（必需：train+test；建议：val） ===
    ap.add_argument('--npyfile_tgt',   type=str, required=True)
    ap.add_argument('--tgt_csv',       type=str, required=True)    # 目标域训练集 CSV
    ap.add_argument('--tgt_val_csv',   type=str, default=None)     # 目标域验证集 CSV（建议提供）
    ap.add_argument('--tgt_test_csv',  type=str, required=True)    # 目标域测试集 CSV（评测用）

    # 与 NPYSliceDataset 对齐的字段
    ap.add_argument('--plane', type=str, default='sagittal', choices=['sagittal','coronal','axial'])

    ap.add_argument('--id_col_src',    type=str, default='case_id')
    ap.add_argument('--label_col_src', type=str, default='label')
    ap.add_argument('--single_file_case_src', action='store_true')
    ap.add_argument('--id_zero_pad_src', type=int, default=None)

    ap.add_argument('--id_col_tgt',    type=str, default='case_id')
    ap.add_argument('--label_col_tgt', type=str, default='label')
    ap.add_argument('--single_file_case_tgt', action='store_true')
    ap.add_argument('--id_zero_pad_tgt', type=int, default=None)

    # 训练与模型
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--pretrained', type=str, default='imagenet')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--device', type=str, default='cuda')

    # 阈值（用于ACC/混淆矩阵/Brier）
    ap.add_argument('--thr', type=float, default=0.5)

    # 阈值选择模式（新增）
    ap.add_argument('--thr_mode', type=str, default='youden',
                    choices=['youden','f1','sp95'],
                    help='在源域验证集上选阈值的准则')
    ap.add_argument('--grid_step', type=float, default=0.005,
                    help='F1 搜索的阈值步长')
    ap.add_argument('--target_spec', type=float, default=0.95,
                    help='SP95 模式的目标特异度')

    # 可复现
    ap.add_argument('--seed', type=int, default=42)

    # 输出
    ap.add_argument('--save_dir',   type=str, required=True)

    args = ap.parse_args()

    # ==== 可复现设置（务必最先执行）====
    setup_determinism(args.seed)

    device = args.device if (torch.cuda.is_available() and 'cuda' in args.device) else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # === 构建数据集 ===
    # 源域 train/val
    ds_src_train = NPYSliceDataset(
        args.npyfile_src, args.src_csv, args.plane,
        args.id_col_src, args.label_col_src,
        args.resize, args.single_file_case_src, args.id_zero_pad_src,
        augment=True
    )
    print(f"[Info] SRC train samples = {len(ds_src_train)}")
    ds_src_val = None
    if args.src_val_csv and os.path.isfile(args.src_val_csv):
        ds_src_val = NPYSliceDataset(
            args.npyfile_src, args.src_val_csv, args.plane,
            args.id_col_src, args.label_col_src,
            args.resize, args.single_file_case_src, args.id_zero_pad_src,
            augment=False
        )
        print(f"[Info] SRC val samples   = {len(ds_src_val)}")

    # 目标域 train/val/test
    ds_tgt_train = NPYSliceDataset(
        args.npyfile_tgt, args.tgt_csv, args.plane,
        args.id_col_tgt, args.label_col_tgt,
        args.resize, args.single_file_case_tgt, args.id_zero_pad_tgt,
        augment=True
    )
    print(f"[Info] TGT train samples = {len(ds_tgt_train)}")

    ds_tgt_val = None
    if args.tgt_val_csv and os.path.isfile(args.tgt_val_csv):
        ds_tgt_val = NPYSliceDataset(
            args.npyfile_tgt, args.tgt_val_csv, args.plane,
            args.id_col_tgt, args.label_col_tgt,
            args.resize, args.single_file_case_tgt, args.id_zero_pad_tgt,
            augment=False
        )
        print(f"[Info] TGT val samples   = {len(ds_tgt_val)}")

    ds_tgt_test = NPYSliceDataset(
        args.npyfile_tgt, args.tgt_test_csv, args.plane,
        args.id_col_tgt, args.label_col_tgt,
        args.resize, args.single_file_case_tgt, args.id_zero_pad_tgt,
        augment=False
    )
    print(f"[Info] TGT test samples  = {len(ds_tgt_test)}")

    # === 训练 ===
    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    model = train_full_supervised(
        model,
        ds_src_train=ds_src_train,
        ds_tgt_train=ds_tgt_train,
        ds_src_val=ds_src_val,
        ds_tgt_val=ds_tgt_val,
        batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        device=device, save_dir=args.save_dir, backbone_name=args.backbone,
        select_thr=args.thr, seed=args.seed
    )

    # === 基于“源域验证集”选阈值（若提供）===
    if ds_src_val is not None and len(ds_src_val) > 0:
        best_thr, thr_info = pick_threshold_on_val(
            model, ds_src_val, device=device,
            mode=args.thr_mode, grid_step=args.grid_step,
            target_spec=args.target_spec, seed=args.seed
        )
        if best_thr is not None:
            print(f"[Val/src] selected threshold by {args.thr_mode}: {best_thr:.3f} | info={thr_info}")
            # 覆盖测试用阈值
            args.thr = float(best_thr)
            # 落盘
            with open(os.path.join(args.save_dir, "calib_val.json"), "w") as f:
                json.dump(dict(source="src_val", **thr_info), f, indent=2)
    else:
        print("[Val/src] no src_val set; keep CLI --thr as-is.")

    # === 测试（目标域）===
    res = evaluate_full_metrics(model, ds_tgt_test, device=device, thr=args.thr, seed=args.seed)
    print("=== Target Test Metrics ===")
    print(f"AUC={res['AUC']:.4f} | ACC={res['ACC']:.4f} | Prec={res['Prec']:.4f} | Rec={res['Rec']:.4f} "
          f"| Spec={res['Spec']:.4f} | F1={res['F1']:.4f} | Brier={res['Brier']:.6f}")
    print(f"CM: tn, fp, fn, tp = {res['TN']} {res['FP']} {res['FN']} {res['TP']}")

    # 另存最终权重
    out_last = os.path.join(args.save_dir, f"{args.backbone}_last.pth")
    torch.save(model.state_dict(), out_last)
    print("[Save] last epoch weights saved.")


if __name__ == "__main__":
    main()
