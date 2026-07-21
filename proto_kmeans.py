#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proto_kmeans.py
类内多原型生成（KMeans，可按类自适应 K），用于后续 NMF/在线软分配与对比学习。
输入：
  --src_feats   源域特征 .npy 或 .pt  (N,D)；与 --src_csv 行对齐
  --src_csv     源域标注CSV，含 [case_id,label]
  --weights     (可选) 样本权重 .npy/.pt/.csv；长度 N
  --K           (可选) 全局固定 K_per_class；若不设则自适应搜索
  --Kmax        自适应搜索时每类的最大K（含）
  --out_centers 输出 centers.npy（按类拼接：先类0的 K0 个，再类1 的 K1 个…）
  --out_meta    输出 meta.json（记录每类K_c、索引范围、silhouette等）

注意：不依赖 sklearn；内置最小KMeans（L2）。
"""
import os, json, argparse
import numpy as np
import torch

def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt",".pth"):
        return torch.load(path, map_location="cpu").float().numpy()
    elif ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        col = [c for c in df.columns if c.lower().startswith("weight") or c.lower() in ("w","weights")]
        arr = df[col[0]].values if col else df.iloc[:, -1].values
        return arr.astype(np.float32)
    else:
        raise ValueError(f"unsupported ext: {ext}")

def kmeans_l2(X, K, iters=50, seed=0, sample_weights=None):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, size=K, replace=False)
    C = X[idx].copy()                        # (K,D)
    w = None if sample_weights is None else sample_weights.reshape(-1,1).astype(np.float32)

    for _ in range(iters):
        # assignment
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        xc = X @ C.T                          # (N,K)
        xx = (X*X).sum(axis=1, keepdims=True) # (N,1)
        cc = (C*C).sum(axis=1, keepdims=True).T # (1,K)
        dist2 = xx + cc - 2*xc
        a = dist2.argmin(axis=1)              # (N,)

        # update
        C_new = np.zeros_like(C)
        for k in range(K):
            m = (a == k)
            if not m.any():
                # re-seed空簇
                C_new[k] = X[rng.integers(0, N)]
            else:
                if w is None:
                    C_new[k] = X[m].mean(axis=0)
                else:
                    ww = w[m]
                    C_new[k] = (ww * X[m]).sum(axis=0) / (ww.sum() + 1e-8)
        if np.allclose(C_new, C, atol=1e-5):
            break
        C = C_new
    return C, a

def silhouette_score_l2(X, labels, C):
    # 近似 silhouette：样本到本簇中心/最近异簇中心的距离差
    N = X.shape[0]
    K = C.shape[0]
    # 预计算
    xx = (X*X).sum(axis=1, keepdims=True)             # (N,1)
    cc = (C*C).sum(axis=1, keepdims=True).T           # (1,K)
    d2 = xx + cc - 2*(X @ C.T)                        # (N,K)
    d = np.sqrt(np.maximum(d2, 0.0) + 1e-12)          # (N,K)
    own = d[np.arange(N), labels]                     # (N,)

    # 最近异簇距离
    mask_same = (labels[:, None] == np.arange(K)[None, :])  # (N,K) bool
    d_mask = d + mask_same * 1e9
    near_other = d_mask.min(axis=1)                   # (N,)

    denom = np.maximum(near_other, own) + 1e-12
    s = (near_other - own) / denom
    return float(np.clip(s.mean(), -1.0, 1.0))


def search_best_K_per_class(Xc, Kmax=6, seed=0, sample_weights=None):
    # 从 {1..Kmax} 里选 silhouette 最优的 K
    best = (1, -1e9, None)  # (K, score, centers)
    for K in range(1, Kmax+1):
        C, a = kmeans_l2(Xc, K, iters=80, seed=seed, sample_weights=sample_weights)
        sc = silhouette_score_l2(Xc, a, C) if K>1 and Xc.shape[0] >= 4*K else 0.0
        if sc > best[1]:
            best = (K, sc, C)
    return best  # K, score, centers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_feats", required=True)
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--id_col", default="case_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--weights", default=None, help="可选样本权重（域相似度、置信度等）")
    ap.add_argument("--K", type=int, default=None, help="固定每类 K；不设则自适应")
    ap.add_argument("--Kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_centers", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    import pandas as pd
    X = load_array(args.src_feats).astype(np.float32)  # (N,D)
    df = pd.read_csv(args.src_csv)
    y = df[args.label_col].astype(int).values
    assert len(y) == X.shape[0], "src_feats 与 src_csv 行数不一致"
    W = load_array(args.weights) if args.weights else None
    if W is not None: assert len(W) == X.shape[0], "weights 长度需等于 N"

    classes = sorted(np.unique(y).tolist())
    centers_all = []
    meta = {"per_class": [], "dim": int(X.shape[1])}
    base = 0
    for c in classes:
        m = (y == c)
        Xc = X[m]
        Wc = (W[m] if W is not None else None)
        if args.K is None:
            Kc, sil, Cc = search_best_K_per_class(Xc, Kmax=args.Kmax, seed=args.seed, sample_weights=Wc)
        else:
            Cc, a = kmeans_l2(Xc, args.K, iters=80, seed=args.seed, sample_weights=Wc)
            sil = silhouette_score_l2(Xc, a, Cc) if args.K>1 else 0.0
            Kc = args.K
        centers_all.append(Cc)
        meta["per_class"].append({
            "class": int(c),
            "K": int(Kc),
            "silhouette": float(sil),
            "start": int(base),
            "end": int(base + Kc)  # [start, end)
        })
        base += Kc

    C = np.vstack(centers_all).astype(np.float32)
    os.makedirs(os.path.dirname(args.out_centers), exist_ok=True)
    np.save(args.out_centers, C)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] centers={C.shape}  saved -> {args.out_centers}")
    print(f"[OK] meta saved -> {args.out_meta}")

if __name__ == "__main__":
    main()
