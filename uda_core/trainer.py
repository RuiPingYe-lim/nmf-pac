# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from .repro import set_global_seed, repro_manifest, make_worker_init_fn, make_generator
from .data_builders import build_src_loader, build_src_proto_loader, build_tgt_loader, build_tgt_test_loader
from .pseudo_dataset import NPYPseudoDataset
from .thresholds import ClasswiseEMAThreshold
from .contrastive import symmetric_infoNCE
from .prototypes import PrototypeBank
from custom_net import build_custom_model
from data import NPYSliceDataset


def _nan_metrics():
    return {
        'auc': float('nan'),
        'acc': float('nan'),
        'prec': float('nan'),
        'rec': float('nan'),
        'f1': float('nan'),
        'prec_binary': float('nan'),
        'rec_binary': float('nan'),
        'f1_binary': float('nan'),
        'prec_macro': float('nan'),
        'rec_macro': float('nan'),
        'f1_macro': float('nan'),
        'prec_weighted': float('nan'),
        'rec_weighted': float('nan'),
        'f1_weighted': float('nan'),
        'spec': float('nan'),
    }


def _fmt_metrics(metrics):
    return ('AUC={auc:.4f} ACC={acc:.4f} PREC={prec:.4f} REC={rec:.4f} '
            'F1={f1:.4f} SPEC={spec:.4f}').format(**metrics)


def _evaluate_labeled_loader(model, loader, device, thr=0.5):
    if loader is None:
        return _nan_metrics()
    was_training = model.training
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob1 = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_score.append(prob1)
            y_true.append(yb.numpy())
    if was_training:
        model.train()
    if len(y_true) == 0:
        return _nan_metrics()
    y_true = np.concatenate(y_true).astype(int)
    y_score = np.concatenate(y_score)
    y_pred = (y_score >= thr).astype(int)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')
    acc = accuracy_score(y_true, y_pred)
    prf = {}
    for avg in ('binary', 'macro', 'weighted'):
        kwargs = {'average': avg, 'zero_division': 0}
        if avg == 'binary':
            kwargs['pos_label'] = 1
        p_avg, r_avg, f_avg, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
        prf[f'prec_{avg}'] = float(p_avg)
        prf[f'rec_{avg}'] = float(r_avg)
        prf[f'f1_{avg}'] = float(f_avg)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')
    return {
        'auc': float(auc),
        'acc': float(acc),
        # Keep legacy keys as weighted metrics for backward compatibility.
        'prec': prf['prec_weighted'],
        'rec': prf['rec_weighted'],
        'f1': prf['f1_weighted'],
        **prf,
        'spec': float(spec),
    }


def _build_src_val_loader(args):
    if not args.csv_src_val:
        return None
    ds = NPYSliceDataset(
        npy_root=args.npy_src,
        csv_file=args.csv_src_val,
        plane=args.plane,
        id_col=args.id_col_src,
        label_col=args.label_col_src,
        resize=args.resize,
        single_file_case=args.single_file_case_src,
        id_zero_pad=args.id_zero_pad_src,
        augment=False,
        volume_proj=args.volume_proj,
    )
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0),
        worker_init_fn=make_worker_init_fn(args.seed + 55),
        generator=make_generator(args.seed + 505),
    )


def _build_tgt_test_loader(args):
    if not args.csv_tgt_test:
        return None
    npy_tgt_test = args.npy_tgt_test if args.npy_tgt_test else args.npy_tgt
    return build_tgt_test_loader(
        npy_root=npy_tgt_test,
        csv=args.csv_tgt_test,
        plane=args.plane,
        resize=args.resize,
        id_col=args.id_col_tgt_test,
        label_col=args.label_col_tgt_test,
        single_file_case=args.single_file_case_tgt_test,
        id_zero_pad=args.id_zero_pad_tgt_test,
        batch_size=64,
        workers=args.workers,
        seed=args.seed,
        volume_proj=args.volume_proj,
    )


@torch.no_grad()
def _collect_target_feature_pool(model, tgt_base_ds, args, device, rd):
    was_training = model.training
    model.eval()
    feats = []
    loader = DataLoader(
        tgt_base_ds,
        batch_size=args.bs_tgt,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=make_worker_init_fn(args.seed + 707 + rd),
        persistent_workers=(args.workers > 0),
    )
    for xb_t, _cid_t in tqdm(loader, desc='post-B target feature refresh', ncols=100, leave=False):
        xb_t = xb_t.to(device, non_blocking=True)
        logits_t, feat_t = model.forward_with_feat(xb_t) if hasattr(model, 'forward_with_feat') else (model(xb_t), None)
        if feat_t is None:
            feat_t = logits_t
        feats.append(feat_t.detach().float().cpu().numpy())
    if was_training:
        model.train()
    if not feats:
        return None
    X = np.concatenate(feats, axis=0).astype(np.float32)
    if X.shape[0] > args.nmf_pool_size:
        rng = np.random.default_rng(args.seed + 909 + rd)
        idx = rng.choice(X.shape[0], size=args.nmf_pool_size, replace=False)
        X = X[np.sort(idx)]
    return X


def train(args):
    set_global_seed(args.seed, deterministic=args.deterministic)
    repro_manifest(args.save_dir, seed=args.seed, deterministic=args.deterministic)

    device = 'cuda' if (torch.cuda.is_available() and 'cuda' in args.device) else 'cpu'
    print(f"[Data] volume_proj={args.volume_proj}")

    dl_src = build_src_loader(
        args.npy_src, args.csv_src, args.plane, args.resize,
        args.id_col_src, args.label_col_src, args.single_file_case_src, args.id_zero_pad_src,
        args.bs_src, args.workers, args.seed, volume_proj=args.volume_proj
    )
    # Deterministic loader (no augmentation / no shuffle / no drop_last) used ONLY
    # for building the source NMF prototypes; the augmented dl_src is used for Stage B.
    dl_src_proto = build_src_proto_loader(
        args.npy_src, args.csv_src, args.plane, args.resize,
        args.id_col_src, args.label_col_src, args.single_file_case_src, args.id_zero_pad_src,
        args.bs_src, args.workers, args.seed, volume_proj=args.volume_proj
    )
    dl_tgt = build_tgt_loader(
        args.npy_tgt, args.csv_tgt, args.plane, args.resize,
        args.id_col_tgt, args.single_file_case_tgt, args.id_zero_pad_tgt,
        args.bs_tgt, args.workers, args.seed, volume_proj=args.volume_proj
    )
    src_val_loader = _build_src_val_loader(args)
    tgt_test_loader = _build_tgt_test_loader(args)
    tgt_base_ds = dl_tgt.dataset

    if src_val_loader is None:
        print('[Warn] src_val loader is missing; best-by-src_val selection will be unavailable.')
    if tgt_test_loader is None:
        print('[Warn] tgt_test loader is missing; tgt test metrics will be NaN.')

    model = build_custom_model(
        method=args.backbone.replace('custom_', '') if args.backbone.startswith('custom_') else args.backbone,
        num_classes=args.num_classes, pretrained=args.pretrained, device=device
    )
    if args.init_from and os.path.isfile(args.init_from):
        try:
            sd = torch.load(args.init_from, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(args.init_from, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        print(f"[Init] loaded from {args.init_from} | missing={len(getattr(res,'missing_keys',[]))} unexpected={len(getattr(res,'unexpected_keys',[]))}")
    model.train()

    feat_dim = getattr(model, 'feat_dim', 2048)
    projector = nn.Sequential(
        nn.Linear(feat_dim, args.proj_dim),
        nn.ReLU(inplace=True),
        nn.Linear(args.proj_dim, args.proj_dim)
    ).to(device)

    params = list(model.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.rounds * max(1, args.epochs_contrast))
    )

    proto = PrototypeBank(
        num_classes=args.num_classes, feat_dim=feat_dim,
        K=args.K, Kmax=args.Kmax, proto_m=args.proto_m, temp_proto=args.tau_proto, device=device
    )
    proto.from_source_init(
        model, dl_src_proto, K=args.K, Kmax=args.Kmax, searchK=(args.K is None),
        init_mode=args.proto_init, nmf_beta=args.nmf_init_beta, nmf_max_iter=args.nmf_init_max_iter,
        nmf_alphaH=args.nmf_init_alphaH, nmf_l1_ratio=args.nmf_init_l1_ratio, nmf_random_state=42
    )

    threshor = ClasswiseEMAThreshold(num_classes=args.num_classes, ema_lambda=args.ema_m, lam_lo=args.lam_lo, lam_hi=args.lam_hi, tau_base=args.tau_base, tau_base_w=args.tau_base_w, tau_floor=args.tau_min, use_floor=True)
    os.makedirs(args.save_dir, exist_ok=True)

    best_src_val_auc = -float('inf')
    best_round = None
    best_src_val_metrics = None
    tgt_test_metrics_at_best_src_val = None
    last_round_src_val_metrics = None
    last_round_tgt_test_metrics = None
    best_ckpt_path = os.path.join(args.save_dir, 'best_by_src_val.pth')
    early_stop_patience = max(0, int(getattr(args, 'early_stop_patience', 0)))
    early_stop_min_rounds = max(0, int(getattr(args, 'early_stop_min_rounds', 0)))
    no_improve_rounds = 0
    completed_rounds = 0
    early_stopped = False
    use_fixed_tau = (args.fixed_tau is not None)
    fixed_tau_value = float(args.fixed_tau) if use_fixed_tau else None
    if use_fixed_tau and not (0.0 <= fixed_tau_value <= 1.0):
        raise ValueError(f'--fixed_tau must be in [0, 1], got {fixed_tau_value}')
    print(
        f"[ThrMode] {'fixed' if use_fixed_tau else 'dynamic'}"
        + (f" (tau={fixed_tau_value:.6f})" if use_fixed_tau else '')
    )
    if early_stop_patience > 0:
        print(
            f"[EarlyStop] monitor=src_val_auc patience={early_stop_patience} "
            f"min_rounds={early_stop_min_rounds}"
        )

    def evaluate_and_record(rd):
        nonlocal best_src_val_auc, best_round
        nonlocal best_src_val_metrics, tgt_test_metrics_at_best_src_val
        nonlocal last_round_src_val_metrics, last_round_tgt_test_metrics
        nonlocal no_improve_rounds

        src_val_metrics = _evaluate_labeled_loader(model, src_val_loader, device=device, thr=0.5)
        tgt_test_metrics = _evaluate_labeled_loader(model, tgt_test_loader, device=device, thr=0.5)
        last_round_src_val_metrics = src_val_metrics
        last_round_tgt_test_metrics = tgt_test_metrics

        print(f"[Round {rd:02d}] src_val: {_fmt_metrics(src_val_metrics)}")
        print(f"[Round {rd:02d}] tgt_test: {_fmt_metrics(tgt_test_metrics)}")

        src_auc = src_val_metrics['auc']
        improved = bool(np.isfinite(src_auc) and src_auc > best_src_val_auc)
        if improved:
            no_improve_rounds = 0
            best_src_val_auc = float(src_auc)
            best_round = int(rd)
            best_src_val_metrics = dict(src_val_metrics)
            tgt_test_metrics_at_best_src_val = dict(tgt_test_metrics)
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"[Best@src_val] round={rd:02d} "
                f"src_val_auc={src_val_metrics['auc']:.4f} "
                f"tgt_test_auc={tgt_test_metrics['auc']:.4f} "
                f"saved={best_ckpt_path}"
            )
        else:
            no_improve_rounds += 1
            print(
                f"[EarlyStop] no_improve_rounds={no_improve_rounds} "
                f"best_src_val_auc={best_src_val_auc:.4f}"
            )
        return improved

    def should_early_stop(rd):
        if (
            early_stop_patience > 0
            and rd >= early_stop_min_rounds
            and no_improve_rounds >= early_stop_patience
        ):
            print(
                f"[Early Stop] no src_val_auc improvement for {no_improve_rounds} rounds "
                f"(patience={early_stop_patience}). Stop at round={rd}."
            )
            return True
        return False

    for rd in range(1, args.rounds + 1):
        print(f"\n===== Round {rd}/{args.rounds} | Stage A: pseudo + threshold + prototype update =====")
        model.eval()
        id2y = {}
        round_conf_sum = 0.0
        round_conf_count = 0
        round_keep_sum = 0
        round_total_sum = 0
        round_conf_min = float('inf')
        round_conf_max = float('-inf')
        round_conf_samples = []
        # Round-wise prototype update: collect all selected target samples across
        # the whole scan, then update prototypes ONCE at round end. Assignments in
        # round r therefore all use the fixed M^{(r-1)} (paper Algorithm 1 / Sec 3.6).
        sel_feats, sel_q, sel_y = [], [], []
        if use_fixed_tau:
            tau_cls_start = [fixed_tau_value] * int(args.num_classes)
        else:
            tau_cls_start = threshor.current_tau_map().tolist()
        ptilde_start = threshor.p_tilde.tolist()
        tau_t_value = float(np.mean(tau_cls_start))
        print(
            "[Thr][start] tau_global=%.6f | tau_cls=[%s] | p_tilde=[%s]" % (
                tau_t_value,
                ', '.join(f"{v:.6f}" for v in tau_cls_start),
                ', '.join(f"{v:.6f}" for v in ptilde_start),
            )
        )

        with torch.no_grad():
            pbar = tqdm(
                DataLoader(
                    tgt_base_ds, batch_size=args.bs_tgt, shuffle=False, num_workers=args.workers,
                    pin_memory=True, worker_init_fn=make_worker_init_fn(args.seed + 33),
                    persistent_workers=(args.workers > 0)
                ),
                desc='A-stage target scan', ncols=100
            )
            for xb_t, cid_t in pbar:
                xb_t = xb_t.to(device, non_blocking=True)
                logits_t, feat_t = model.forward_with_feat(xb_t) if hasattr(model, 'forward_with_feat') else (model(xb_t), None)
                if feat_t is None:
                    feat_t = logits_t
                if args.use_nmf_pseudo:
                    q_proto, p_cls = proto.nmf_assign(feat_t, beta_loss=args.beta_loss, iters=args.nmf_assign_iters)
                else:
                    q_proto, p_cls = proto.soft_assign(feat_t)

                q_batch = p_cls
                if use_fixed_tau:
                    tau_map = torch.full(
                        (int(args.num_classes),),
                        fixed_tau_value,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    tau_map = threshor.update_and_get(q_batch).to(device)
                y_hat = q_batch.argmax(dim=1)
                conf_t = q_batch.max(dim=1).values
                keep = conf_t > tau_map[y_hat]
                sel = int(keep.sum().item())
                tot = keep.numel()
                conf_cpu = conf_t.detach().float().cpu()
                round_conf_sum += float(conf_cpu.sum().item())
                round_conf_count += int(conf_cpu.numel())
                round_keep_sum += sel
                round_total_sum += tot
                if conf_cpu.numel() > 0:
                    cmin = float(conf_cpu.min().item())
                    cmax = float(conf_cpu.max().item())
                    if cmin < round_conf_min:
                        round_conf_min = cmin
                    if cmax > round_conf_max:
                        round_conf_max = cmax
                    round_conf_samples.append(conf_cpu)
                pbar.set_postfix_str(f"keep {sel}/{tot}")

                if keep.any():
                    # Collect selected samples; the prototype update is deferred to
                    # the end of the round (round-wise, not batch-wise).
                    sel_feats.append(feat_t[keep].detach().cpu())
                    sel_q.append(q_proto[keep].detach().cpu())
                    sel_y.append(y_hat[keep].detach().cpu())
                    kept_ids = [str(c) for m, c in zip(keep.tolist(), cid_t) if m]
                    kept_y = y_hat[keep].tolist()
                    for c, y in zip(kept_ids, kept_y):
                        id2y[c] = int(y)

        # ---- Round-wise prototype update: M^{(r-1)} -> M^{(r)} (once per round) ----
        if sel_feats:
            feats_all = torch.cat(sel_feats, dim=0).to(device)
            q_all = torch.cat(sel_q, dim=0).to(device)
            y_all = torch.cat(sel_y, dim=0).to(device)
            if args.mask_cross_class_update:
                proto.momentum_update_masked(feats_all, q_all, y_all)
            else:
                proto.momentum_update(feats_all, q_all)

        if use_fixed_tau:
            tau_cls_end = [fixed_tau_value] * int(args.num_classes)
        else:
            tau_cls_end = threshor.current_tau_map().tolist()
        ptilde_end = threshor.p_tilde.tolist()
        print(
            "[Thr][end]   tau_global=%.6f | tau_cls=[%s] | p_tilde=[%s]" % (
                float(np.mean(tau_cls_end)),
                ', '.join(f"{v:.6f}" for v in tau_cls_end),
                ', '.join(f"{v:.6f}" for v in ptilde_end),
            )
        )
        if round_conf_count > 0:
            conf_all = torch.cat(round_conf_samples, dim=0) if round_conf_samples else None
            conf_mean = round_conf_sum / max(1, round_conf_count)
            keep_ratio = round_keep_sum / max(1, round_total_sum)
            if conf_all is not None and conf_all.numel() > 0:
                q50 = float(torch.quantile(conf_all, 0.50).item())
                q90 = float(torch.quantile(conf_all, 0.90).item())
                q99 = float(torch.quantile(conf_all, 0.99).item())
            else:
                q50 = float('nan')
                q90 = float('nan')
                q99 = float('nan')
            print(
                "[Conf][round %d] min=%.6f p50=%.6f p90=%.6f p99=%.6f max=%.6f mean=%.6f | keep=%d/%d (%.2f%%)"
                % (
                    rd,
                    round_conf_min,
                    q50,
                    q90,
                    q99,
                    round_conf_max,
                    conf_mean,
                    round_keep_sum,
                    round_total_sum,
                    keep_ratio * 100.0,
                )
            )
        print(f"[A-stage] selected target pseudo samples: {len(id2y)}")

        print(f"===== Round {rd}/{args.rounds} | Stage B: contrastive =====")
        model.train()
        print('id2y size =', len(id2y), 'examples:', list(id2y)[:5])
        tgt_pseudo_ds = NPYPseudoDataset(tgt_base_ds, id2y)
        if len(tgt_pseudo_ds) == 0:
            print('[B-stage] pseudo subset is empty, skip this round contrastive training.')
            evaluate_and_record(rd)
            completed_rounds = rd
            if should_early_stop(rd):
                early_stopped = True
                break
            continue

        gen_t_seed = args.seed + 303 + rd
        dl_tgt_pseudo = DataLoader(
            tgt_pseudo_ds, batch_size=args.bs_tgt, shuffle=True, num_workers=args.workers,
            pin_memory=True, drop_last=True, worker_init_fn=make_worker_init_fn(args.seed + 44 + rd),
            generator=make_generator(gen_t_seed), persistent_workers=(args.workers > 0)
        )

        def inf_cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        it_src = inf_cycle(dl_src)
        for epoch in range(max(1, args.epochs_contrast)):
            pbar_b = tqdm(dl_tgt_pseudo, desc=f'B-stage contrastive (ep {epoch+1}/{args.epochs_contrast})', ncols=100)
            for xb_t, yb_t_hat in pbar_b:
                xb_s, yb_s = next(it_src)
                xb_s = xb_s.to(device, non_blocking=True)
                yb_s = yb_s.to(device, non_blocking=True)
                xb_t = xb_t.to(device, non_blocking=True)
                yb_t_hat = yb_t_hat.to(device, non_blocking=True)

                optim.zero_grad()
                logits_s, feat_s = model.forward_with_feat(xb_s) if hasattr(model, 'forward_with_feat') else (model(xb_s), None)
                if feat_s is None:
                    feat_s = logits_s
                loss_src = nn.CrossEntropyLoss()(logits_s, yb_s) * float(args.lam_src_ce)

                logits_t, feat_t = model.forward_with_feat(xb_t) if hasattr(model, 'forward_with_feat') else (model(xb_t), None)
                if feat_t is None:
                    feat_t = logits_t

                z_s = projector(feat_s)
                z_t = projector(feat_t)
                loss_con = symmetric_infoNCE(z_s, yb_s, z_t, yb_t_hat, tau=args.tau_con)
                loss = loss_src + args.lam_con * loss_con
                loss.backward()
                optim.step()
                sched.step()
                pbar_b.set_postfix(loss=f"{float(loss.item()):.4f}", n_sel=len(tgt_pseudo_ds))

        if args.nmf_refresh_each_round:
            try:
                X_pool = _collect_target_feature_pool(model, tgt_base_ds, args, device, rd)
                if X_pool is not None and X_pool.shape[0] >= max(200, proto.mu.shape[0] * 2):
                    print(
                        f'[NMF] post-B refresh uses current-network target features: '
                        f'round={rd} pool={X_pool.shape[0]}'
                    )
                    proto.nmf_refresh(
                        X_pool, max_iter=args.nmf_max_iter, beta_loss=args.beta_loss,
                        alphaH=args.alphaH, l1_ratio=args.l1_ratio, random_state=42
                    )
            except Exception as e:
                print(f'[NMF] refresh skipped due to error: {e}')

        evaluate_and_record(rd)
        completed_rounds = rd
        if should_early_stop(rd):
            early_stopped = True
            break

    summary = {
        "selection_rule": "best_by_src_val_auc",
        "early_stopped": early_stopped,
        "completed_rounds": completed_rounds,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_rounds": early_stop_min_rounds,
        "best_round": best_round,
        "best_src_val_metrics": best_src_val_metrics if best_src_val_metrics is not None else _nan_metrics(),
        "tgt_test_metrics_at_best_src_val": (
            tgt_test_metrics_at_best_src_val if tgt_test_metrics_at_best_src_val is not None else _nan_metrics()
        ),
        "last_round_src_val_metrics": last_round_src_val_metrics if last_round_src_val_metrics is not None else _nan_metrics(),
        "last_round_tgt_test_metrics": last_round_tgt_test_metrics if last_round_tgt_test_metrics is not None else _nan_metrics(),
    }
    summary_path = os.path.join(args.save_dir, 'metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    legacy_summary_path = os.path.join(args.save_dir, 'best_by_src_val.json')
    with open(legacy_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n===== Final Summary =====')
    print('[Selection] best model selected by src_val_auc')
    print(
        f"[Summary] early_stopped={summary['early_stopped']} "
        f"completed_rounds={summary['completed_rounds']}"
    )
    print(f"[Summary] best_round={summary['best_round']}")
    print(f"[Summary] best_src_val_metrics={summary['best_src_val_metrics']}")
    print(f"[Summary] tgt_test_metrics_at_best_src_val={summary['tgt_test_metrics_at_best_src_val']}")
    print(f"[Summary] last_round_src_val_metrics={summary['last_round_src_val_metrics']}")
    print(f"[Summary] last_round_tgt_test_metrics={summary['last_round_tgt_test_metrics']}")
    print(f"[Summary] saved: {summary_path}")
