# nmf_lib_assign.py
# 用 sklearn 的 NMF 在目标域上做伪标签：
#   mode=fixed  : 固定基底(用你的 centers 当 H，不更新 H)
#   mode=adapt  : 自适应基底(用 centers 初始化 H，并允许更新 H)
#
# 输出：case_id, y_hat, conf, 以及二分类时的 p1_nmf

import argparse, os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition._nmf import non_negative_factorization

def minmax_fit_transform_together(X, C, eps=1e-12):
    Z = np.vstack([X, C]).astype(np.float32)
    mn = Z.min(axis=0, keepdims=True)
    mx = Z.max(axis=0, keepdims=True)
    rng = np.maximum(mx - mn, eps)
    return (X - mn)/rng, (C - mn)/rng

def row_normalize(P, eps=1e-12):
    s = P.sum(axis=1, keepdims=True) + eps
    return P / s

def align_components_by_cos(H_new, H_init):
    # 用余弦相似度把学到的 H_new(K,D) 与初始 H_init(K,D) 对齐（避免组件顺序打乱）
    def l2n(A): 
        n = np.sqrt((A*A).sum(axis=1, keepdims=True)) + 1e-12
        return A/n
    A = l2n(H_new); B = l2n(H_init)
    S = A @ B.T  # (K,K)
    # 贪心匹配（K小于10时足够用；需要更稳可用匈牙利算法）
    K = H_new.shape[0]
    used_rows, used_cols = set(), set()
    pairs = []
    for _ in range(K):
        i,j = np.unravel_index(np.argmax(S), S.shape)
        while i in used_rows or j in used_cols:
            S[i,j] = -1
            i,j = np.unravel_index(np.argmax(S), S.shape)
        pairs.append((i,j))
        used_rows.add(i); used_cols.add(j)
        S[i,:] = -1; S[:,j] = -1
    # 重排 H_new 和 W（注意：X≈W@H，H 行的重排 ≡ W 列的同样重排）
    perm = [i for i,_ in sorted(pairs, key=lambda x:x[1])]
    return perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="目标域特征 .npy (N,D)")
    ap.add_argument("--centers", required=True, help="簇心/原型 .npy (K,D)")
    ap.add_argument("--ids-ref", required=True, help="包含 case_id 的CSV（顺序与特征一致）")
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--mode", choices=["fixed","adapt"], default="adapt",
                    help="fixed: 固定基底H不更新；adapt: 允许H更新（推荐）")
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--beta-loss", default="frobenius", choices=["frobenius","kullback-leibler","itakura-saito"])
    ap.add_argument("--alphaW", type=float, default=0.0, help="W正则强度")
    ap.add_argument("--alphaH", type=float, default=0.0, help="H正则强度（adapt时可>0防漂移过头）")
    ap.add_argument("--l1-ratio", type=float, default=0.0, help="0=L2, 1=L1, 介于其间是弹性网")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    X = np.load(args.feat).astype(np.float32)      # (N,D)
    C = np.load(args.centers).astype(np.float32)   # (K,D)
    N, D = X.shape
    K = C.shape[0]
    assert C.shape[1] == D, "centers 维度与特征不一致"

    # 读 case_id
    df_ids = pd.read_csv(args.ids_ref)
    if "case_id" not in df_ids.columns:
        raise ValueError("ids-ref 缺少 case_id 列")
    case_ids = df_ids["case_id"].astype(str).values
    assert len(case_ids) == N, "ids-ref 与特征行数不一致"

    # 保证非负：X, C 一起做逐维 min-max 到 [0,1]
    Xs, Cs = minmax_fit_transform_together(X, C)

    # sklearn 记号：X ≈ W @ H，H 形状应为 (K, D)，正好放入我们的 centers
    # 初始化
    H_init = np.clip(Cs, 0, None)     # (K,D) 基底
    W_init = np.maximum(Xs @ H_init.T, 1e-6)  # 一个合理的非负初始系数 (N,K)

    update_H = (args.mode == "adapt")  # adapt: 允许更新 H；fixed: 不更新 H

    W, H, n_iter = non_negative_factorization(
        Xs, W=W_init, H=H_init, init='custom',
        update_H=update_H, solver='mu', beta_loss=args.beta_loss,
        max_iter=args.max_iter, random_state=args.random_state, tol=1e-6,
        alpha_W=args.alphaW, alpha_H=args.alphaH, l1_ratio=args.l1_ratio
    )
    # 说明：这里 W=(N,K) 是样本的非负“软分配”，H=(K,D) 是（可能已自适应过的）基底/簇心

    # 若允许更新 H，可能出现组件顺序轻微交换——对齐回初始顺序
    if update_H:
        perm = align_components_by_cos(H, H_init)
        H = H[perm, :]
        W = W[:, perm]

    # 归一化得到“概率”
    P = row_normalize(W)
    y_hat = P.argmax(axis=1).astype(int)
    conf = P.max(axis=1)

    out = pd.DataFrame({"case_id": case_ids, "y_hat": y_hat, "conf": conf})
    if K == 2:
        out["p1_nmf"] = P[:, 1]
        out["h0"] = W[:, 0]; out["h1"] = W[:, 1]
    else:
        for k in range(K):
            out[f"p{k}_nmf"] = P[:, k]
            out[f"h{k}"] = W[:, k]

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    out.to_csv(args.outcsv, index=False)
    print(f"[OK] saved -> {args.outcsv} | N={N}, K={K}, iters={n_iter}, mode={args.mode}")

if __name__ == "__main__":
    main()
