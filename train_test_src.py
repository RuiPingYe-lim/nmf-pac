# -*- coding: utf-8 -*-
"""
（
已经修改以提高可复现性：固定 PYTHONHASHSEED、numpy、random、torch 等；
DataLoader 使用 torch.Generator；保存 repro.json 到 save_dir。
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

import time
import argparse
import random
import json

from copy import deepcopy
import numpy as np
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from calib import learn_temperature, evaluate_calibrated, choose_best_threshold_calibrated

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18, resnet50, alexnet,
    ResNet18_Weights, ResNet50_Weights, AlexNet_Weights
)
from data import NPYSliceDataset, set_seed
from custom_net import build_custom_model


def _prf_by_average(y, pred):
    out = {}
    for avg in ("binary", "macro", "weighted"):
        kwargs = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kwargs["pos_label"] = 1
        p, r, f1, _ = metrics.precision_recall_fscore_support(y, pred, **kwargs)
        out[f"prec_{avg}"] = float(p)
        out[f"rec_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f1)
    return out

# ----------------- reproducibility helpers -----------------
def make_reproducible(seed:int=42, cudnn_deterministic=True):
    """
    设置环境，尽量保证 PyTorch 实验可复现。
    Returns: dict of flags/info for logging.
    """
    info = {}
    # 1) OS env
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    info['seed'] = int(seed)

    # 2) torch generator commonly used for DataLoader shuffle
    gen = torch.Generator()
    gen.manual_seed(seed)
    info['generator_seed'] = int(seed)

    # 3) cudnn / deterministic flags
    try:
        # Newer PyTorch: use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        info['use_deterministic_algorithms'] = True
    except Exception:
        # fallback for older versions
        try:
            torch.set_deterministic(True)
            info['set_deterministic'] = True
        except Exception:
            info['set_deterministic'] = False

    # cudnn flags
    try:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = False
        info['cudnn_deterministic'] = bool(cudnn_deterministic)
        info['cudnn_benchmark'] = False
    except Exception:
        pass

    # record environment info
    info['torch_version'] = torch.__version__
    info['cuda_available'] = torch.cuda.is_available()
    return gen, info

def worker_init_fn(worker_id):
    # 每个 worker 再次种子化，避免 numpy/random 在 worker 里复用相同种子
    seed = torch.initial_seed() % (2**32 - 1)
    import numpy as _np, random as _random
    _np.random.seed(seed + worker_id)
    _random.seed(seed + worker_id)

# ----------------- model -----------------
def build_model(backbone='resnet50', num_classes=2, pretrained='imagenet', device='cuda'):
    if backbone.startswith('custom_'):
        method = backbone.replace('custom_', '')
        m = build_custom_model(method=method, num_classes=num_classes,
                                  pretrained=pretrained, device=device)
        if hasattr(m, 'rsa'):
            m.rsa = nn.Identity()
            print("[Hotfix] Disable m.rsa -> Identity() during source-only training")
        return m

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



def evaluate(model, ds, device='cuda', threshold=0.5, tag='val', batch_size=64, generator=None, num_workers=0):
    """
    评估函数，返回一组指标。
    DataLoader 使用外部传入的 generator 来保证 shuffle（若有）可复现。
    """
    if ds is None or len(ds) == 0:
        return None

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, worker_init_fn=worker_init_fn if num_workers>0 else None,
                    generator=generator)
    model.eval()

    all_p1, all_y = [], []
    loss_sum = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        
        # 在 logits 计算之后获取其设备
        weight = torch.tensor([1.0, 2.0]).to(logits.device)  # 将 weight 张量移动到与 logits 相同的设备
        ce = nn.CrossEntropyLoss(weight=weight)  # 确保损失函数使用正确的设备
        loss = ce(logits, yb)
        
        p1 = torch.softmax(logits, dim=1)[:, 1]  # 获取正类的概率
        all_p1.append(p1.detach().cpu().numpy())  # 将预测概率移到 CPU
        all_y.append(yb.detach().cpu().numpy())  # 将真实标签移到 CPU
        loss_sum += loss.item()

    p1 = np.concatenate(all_p1)
    y = np.concatenate(all_y)

    # AUC
    try:
        auc = float(metrics.roc_auc_score(y, p1))
    except Exception as e:
        auc = 0.5
        print(f"Warning: AUC calculation failed. Error: {e}")

    # 基于阈值的分类指标
    pred = (p1 >= threshold).astype(int)
    acc = float((pred == y).mean())
    
    # 精确率、召回率、F1
    prf = _prf_by_average(y, pred)
    
    # 特异度：TN / (TN + FP)
    cm = metrics.confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        f"{tag}_loss": loss_sum / max(len(dl), 1),
        f"{tag}_auc": auc,
        f"{tag}_acc": acc,
        f"{tag}_prec": prf["prec_weighted"],
        f"{tag}_rec": prf["rec_weighted"],
        f"{tag}_spec": spec,
        f"{tag}_f1": prf["f1_weighted"],
        f"{tag}_prec_binary": prf["prec_binary"],
        f"{tag}_rec_binary": prf["rec_binary"],
        f"{tag}_f1_binary": prf["f1_binary"],
        f"{tag}_prec_macro": prf["prec_macro"],
        f"{tag}_rec_macro": prf["rec_macro"],
        f"{tag}_f1_macro": prf["f1_macro"],
        f"{tag}_prec_weighted": prf["prec_weighted"],
        f"{tag}_rec_weighted": prf["rec_weighted"],
        f"{tag}_f1_weighted": prf["f1_weighted"],
    }


def choose_best_threshold(model, ds_val, device='cuda',
                          mode='youden',        # 'acc' | 'f1' | 'youden' | 'custom'
                          grid_step=0.005,      # 阈值步长：0.005 更细
                          thr_min=0.0, thr_max=1.0,
                          pos_weight=1.0, spec_weight=1.0,
                          batch_size=64, generator=None, num_workers=0):
    """
    返回: (best_thr, stats_dict)
    DataLoader 使用 generator 保证可复现 shuffle。
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn import metrics

    dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, worker_init_fn=worker_init_fn if num_workers>0 else None,
                    generator=generator)
    model.eval()
    all_p1, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_p1.append(p1)
            all_y.append(yb.numpy())
    p1 = np.concatenate(all_p1)
    y  = np.concatenate(all_y).astype(int)

    thr_grid = np.arange(thr_min, thr_max + 1e-9, grid_step)
    best_thr, best_score = 0.5, -1.0
    best_stats = {}

    for t in thr_grid:
        pred = (p1 >= t).astype(int)
        tp = ((pred==1)&(y==1)).sum(); tn = ((pred==0)&(y==0)).sum()
        fp = ((pred==1)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()

        acc  = (tp + tn) / max(len(y), 1)
        rec  = tp / max(tp + fn, 1)              # TPR / Recall
        spec = tn / max(tn + fp, 1)              # TNR / Specificity
        f1   = 0.0 if tp == 0 else (2*tp) / max(2*tp + fp + fn, 1)

        if mode == 'acc':
            score = acc
        elif mode == 'f1':
            score = f1
        elif mode == 'youden':
            score = rec + spec - 1.0             # Youden's J
        else:  # 'custom'：按需求权衡召回与特异度
            score = pos_weight * rec + spec_weight * spec

        if score > best_score:
            best_score = score
            best_thr = float(t)
            best_stats = dict(acc=float(acc), rec=float(rec), spec=float(spec), f1=float(f1))

    print(f"[Val] best_thr={best_thr:.3f} by {mode} | stats={best_stats}")
    return best_thr, best_stats


def train_source_only(model, ds_src, batch_size=32, epochs=8, lr=1e-4, weight_decay=1e-4,
                      device='cuda', save_dir=None, backbone_name='resnet50',
                      ds_val=None, ds_test=None, run_test_on_best=False,
                      plane='sagittal', task='cls', seed=42,
                      dl_num_workers=0, generator=None,
                      early_stop_patience=0, early_stop_min_epochs=0):
    # DataLoader 用 generator 来保证 shuffle 的可复现性
    dl = DataLoader(ds_src, batch_size=batch_size, shuffle=True, num_workers=dl_num_workers,
                    pin_memory=True, worker_init_fn=worker_init_fn if dl_num_workers>0 else None,
                    generator=generator)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=3, verbose=True)

    best_val_auc = -1.0
    best_state = None
    best_metrics = None
    early_stop_patience = max(0, int(early_stop_patience))
    early_stop_min_epochs = max(0, int(early_stop_min_epochs))
    no_improve_epochs = 0

    for ep in range(1, epochs + 1):
        print(f"\n=== Epoch {ep}/{epochs} ===")

        t0 = time.time()
        model.train()
        loss_sum, steps = 0.0, 0

        # 为了计算 train_auc，收集当轮训练 p1/y
        tr_p1_buf, tr_y_buf = [], []

        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            # 统计
            with torch.no_grad():
                p1 = torch.softmax(logits, dim=1)[:, 1]
                tr_p1_buf.append(p1.detach().cpu().numpy())
                tr_y_buf.append(yb.detach().cpu().numpy())

            loss_sum += float(loss.item()); steps += 1

        # 本轮训练损失与 AUC
        train_loss = loss_sum / max(steps, 1)
        try:
            train_auc = float(metrics.roc_auc_score(
                np.concatenate(tr_y_buf), np.concatenate(tr_p1_buf),average="weighted",
            ))
        except Exception:
            train_auc = 0.5

        # 验证集评估
        val_loss = val_auc = None
        if ds_val is not None:
            res = evaluate(model, ds_val, device, threshold=0.5, tag='val',
                           batch_size=64, generator=generator, num_workers=dl_num_workers)
            val_loss = res['val_loss']; val_auc = res['val_auc']

            # 调度器根据 AUC 调整 LR
            scheduler.step(val_auc)

            # 刷新 best → 保存并可选立刻在测试集评估
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = deepcopy(model.state_dict())
                if save_dir is not None:
                    torch.save(best_state, os.path.join(save_dir, f"{backbone_name}_best.pth"))
                    print(f"[Save] update best model (val_auc={best_val_auc:.4f})")

                # 选择最佳阈值并进行测试
                if run_test_on_best and (ds_test is not None):
                    # 1) 在验证集上学习温度 T
                    T_calib = learn_temperature(model, ds_val, device, init_T=1.0, use_lbfgs=True, max_iter=200)

                    # 2) 用“带温度”的概率在验证集上选阈值（建议 youden 或 f1）
                    best_threshold, best_stats = choose_best_threshold_calibrated(
                        model, ds_val, device,
                        mode='youden', grid_step=0.005, thr_min=0.0, thr_max=1.0,
                        temperature=T_calib
                    )

                    # 3) 在测试集用相同的 T 与阈值评估
                    test_res = evaluate_calibrated(
                        model, ds_test, device,
                        threshold=best_threshold, tag='test',
                        temperature=T_calib
                    )
                    best_metrics = {
                        "selection_metric": "src_val_auc",
                        "src_val_auc": float(val_auc),
                        "best_epoch": int(ep),
                        "tgt_test_auc_at_best": float(test_res["test_auc"]),
                        "tgt_test_acc_at_best": float(test_res["test_acc"]),
                        "tgt_test_prec_at_best": float(test_res["test_prec"]),
                        "tgt_test_rec_at_best": float(test_res["test_rec"]),
                        "tgt_test_spec_at_best": float(test_res["test_spec"]),
                        "tgt_test_f1_at_best": float(test_res["test_f1"]),
                        "threshold": float(best_threshold),
                        "temperature": float(T_calib),
                    }
                    metrics_path = os.path.join(save_dir, "best_metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
                    print(f"[Save] best metrics saved to {metrics_path}")
                                    
                    calib_info = {
                        "temperature": float(T_calib),
                        "threshold":   float(best_threshold),
                        "thr_mode":    "youden",
                        "grid_step":   0.005,
                        "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch":       int(ep),
                        "backbone":    backbone_name
                    }
                    json_path = os.path.join(save_dir, "calib.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(calib_info, f, ensure_ascii=False, indent=2)
                    print(f"[Save] calib params saved to {json_path} -> T={calib_info['temperature']:.4f}, thr={calib_info['threshold']:.3f}")


                    print(
                        "Epoch:{},  For task {} in plane {}, On the test_set, "
                        "test auc: {} | test loss: {} | test_acc : {} | test_prec: {} | "
                        "test_rec: {} | test_spec: {} | test_f1: {}".format(
                            ep, task, plane,
                            f"{test_res['test_auc']:.4f}",
                            f"{test_res['test_loss']:.4f}",
                            f"{test_res['test_acc']:.4f}",
                            f"{test_res['test_prec']:.4f}",
                            f"{test_res['test_rec']:.4f}",
                            f"{test_res['test_spec']:.4f}",
                            f"{test_res['test_f1']:.4f}",
                        )
                    )
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1


        # —— 你的训练/验证打印格式 ——（含耗时）
        delta = time.time() - t0
        print(
            "  Epoch: {} |train loss : {} | train auc {} | val loss {} | val auc {} | elapsed time {} s".format(
                ep,
                f"{train_loss:.4f}",
                f"{train_auc:.4f}",
                "NA" if val_loss is None else f"{val_loss:.4f}",
                "NA" if val_auc is None else f"{val_auc:.4f}",
                f"{delta:.1f}"
            )
        )
        if (
            ds_val is not None
            and early_stop_patience > 0
            and ep >= early_stop_min_epochs
            and no_improve_epochs >= early_stop_patience
        ):
            print(
                f"[Early Stop] no val_auc improvement for {no_improve_epochs} epochs "
                f"(patience={early_stop_patience}). Stop at epoch={ep}."
            )
            break

    # 训练结束保存最后一轮
    if save_dir is not None:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{backbone_name}_last.pth"))
        print("[Save] last epoch weights saved.")

    # 返回时加载 best（若有）
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser("Source-only training for model on source domain data")

    # 必需：数据路径
    ap.add_argument('--npyfile_src', type=str, required=True)
    ap.add_argument('--src_csv',     type=str, required=True)

    # 与 NPYSliceDataset 对齐
    ap.add_argument('--plane', type=str, default='sagittal', choices=['sagittal','coronal','axial'])
    ap.add_argument('--id_col_src',    type=str, default='case_id')
    ap.add_argument('--label_col_src', type=str, default='label')
    ap.add_argument('--single_file_case_src', action='store_true',
                    help='MRNet风格每病例一个.npy时需加此标志')
    ap.add_argument('--id_zero_pad_src', type=int, default=None,
                    help='若文件名需要零填充宽度（如 0001.npy），填位宽；否则留空')

    # 训练与模型
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--pretrained', type=str, default='imagenet')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--task', type=str, default='cls', help='用于日志打印的任务名')

    # 输出
    ap.add_argument('--save_dir',   type=str, required=True)

    # 可选：仅做路径连通性检查
    ap.add_argument('--dry_run', action='store_true', help='只做数据探测并打印前若干样本，不训练')

    # 源域验证集（可选）
    ap.add_argument('--src_val_csv', type=str, default=None)

    # 测试集（可选）+ 触发开关
    ap.add_argument('--test_csv', type=str, default=None)
    ap.add_argument('--npyfile_test', type=str, default=None,
                    help='若不指定，默认与 --npyfile_src 相同')
    ap.add_argument('--id_col_test',    type=str, default='case_id')
    ap.add_argument('--label_col_test', type=str, default='label')
    ap.add_argument('--single_file_case_test', action='store_true')
    ap.add_argument('--id_zero_pad_test', type=int, default=None)
    ap.add_argument('--run_test_on_best', action='store_true',default=True,
                    help='验证 AUC 创新高时，立即对测试集评估')
    ap.add_argument('--early_stop_patience', type=int, default=0,
                    help='Stop after this many source-val epochs without AUC improvement. 0 disables early stopping.')
    ap.add_argument('--early_stop_min_epochs', type=int, default=0,
                    help='Do not early-stop before this many epochs have completed.')

    # reproducibility arguments
    ap.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    ap.add_argument('--dl_num_workers', type=int, default=0, help='DataLoader num_workers (0 更易复现)')
    ap.add_argument('--force_deterministic', action='store_true', help='尝试启用 torch deterministic algorithms')

    args = ap.parse_args()

    # 准备 reproducible 环境
    os.makedirs(args.save_dir, exist_ok=True)
    gen, repro_info = make_reproducible(seed=args.seed, cudnn_deterministic=args.force_deterministic)
    repro_info['dl_num_workers'] = int(args.dl_num_workers)
    # save reproducibility metadata
    with open(os.path.join(args.save_dir, "repro.json"), "w", encoding="utf-8") as f:
        json.dump(repro_info, f, ensure_ascii=False, indent=2)

    # 准备
    set_seed(args.seed)  # 若 NPYSliceDataset 里有用到 set_seed 的地方
    device = args.device if (torch.cuda.is_available() and 'cuda' in args.device) else 'cpu'

    # 构建源域训练集
    ds_src = NPYSliceDataset(
        args.npyfile_src, args.src_csv, args.plane,
        args.id_col_src, args.label_col_src,
        args.resize, args.single_file_case_src, args.id_zero_pad_src,
        augment=True
    )
    print(f"[Info] src train samples = {len(ds_src)} (plane={args.plane}, single_file={args.single_file_case_src})")

    # 验证集（若提供）
    ds_val = None
    if args.src_val_csv and os.path.isfile(args.src_val_csv):
        ds_val = NPYSliceDataset(
            args.npyfile_src, args.src_val_csv, args.plane,
            args.id_col_src, args.label_col_src,
            args.resize, args.single_file_case_src, args.id_zero_pad_src,
            augment=False
        )
        print(f"[Info] src val samples  = {len(ds_val)}")

    # 测试集（若提供）
    ds_test = None
    if args.test_csv and os.path.isfile(args.test_csv):
        npy_root_test = args.npyfile_test or args.npyfile_src
        ds_test = NPYSliceDataset(
            npy_root_test, args.test_csv, args.plane,
            args.id_col_test, args.label_col_test,
            args.resize, args.single_file_case_test, args.id_zero_pad_test,
            augment=False
        )
        print(f"[Info] test samples     = {len(ds_test)}")

    # dry-run 模式：仅做探测
    if args.dry_run:
        print("[DryRun] Only dataset connectivity check done.")
        return

    # 构建模型并训练
    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    model = train_source_only(
        model, ds_src,
        batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        device=device, save_dir=args.save_dir, backbone_name=args.backbone,
        ds_val=ds_val, ds_test=ds_test, run_test_on_best=args.run_test_on_best,
        plane=args.plane, task=args.task,
        seed=args.seed, dl_num_workers=args.dl_num_workers, generator=gen,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
    )

    # 结束提示
    print("[Done] Training finished. Best weights already saved as *_best.pth; last epoch as *_last.pth.")


if __name__ == "__main__":
    main()
