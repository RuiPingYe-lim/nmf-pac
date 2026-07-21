
#PrototypeBank（KMeans/NMF、动量更新、刷新）
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition._nmf import non_negative_factorization
from .contrastive import l2n
# 复用你现有的工具
from proto_kmeans import kmeans_l2, search_best_K_per_class
from nmf_lib_assign import minmax_fit_transform_together, align_components_by_cos, row_normalize


def _rescale_rows_to_norm(H: np.ndarray, target_norms: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    H = H.astype(np.float32, copy=False)
    target_norms = np.asarray(target_norms, dtype=np.float32).reshape(-1, 1)
    row_norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(row_norms, eps) * target_norms


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, K: int = None, Kmax: int = 4,
                 proto_m: float = 0.9, temp_proto: float = 0.07, device: str = 'cuda'):
        super().__init__()
        self.C = num_classes; self.D = feat_dim
        self.device = torch.device(device if (torch.cuda.is_available() and 'cuda' in device) else 'cpu')
        self.proto_m = float(proto_m); self.temp_proto = float(temp_proto)
        self.per_class_K = [K if K is not None else None] * self.C
        self.offsets = None; self.mu = None
        self.register_buffer('_dummy', torch.zeros(1))

    @torch.no_grad()
    def from_source_init(self, model, dl_src, K: int = None, Kmax: int = 4, searchK: bool = True,
                         init_mode: str = 'kmeans', nmf_beta: str = 'frobenius', nmf_max_iter: int = 150,
                         nmf_alphaH: float = 0.0, nmf_l1_ratio: float = 0.0, nmf_random_state: int = 42):
        model.eval()
        from collections import defaultdict
        feats_by_cls = defaultdict(list)
        for xb, yb in tqdm(dl_src, desc='[Init] extract src feats', ncols=100, leave=False):
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.no_grad():
                logits, feat = (model.forward_with_feat(xb) if hasattr(model, 'forward_with_feat') else (model(xb), None))
                if feat is None: feat = logits
            f = feat.detach().float().cpu().numpy(); y = yb.detach().cpu().numpy()
            for cls in range(self.C):
                m = (y == cls)
                if m.any(): feats_by_cls[cls].append(f[m])
        centers = []; self.per_class_K = []; base = 0; self.offsets = []
        for c in range(self.C):
            Xc = np.vstack(feats_by_cls[c]) if len(feats_by_cls[c]) else np.zeros((0, self.D), dtype=np.float32)
            if Xc.shape[0] == 0:
                Cc = np.zeros((1, self.D), dtype=np.float32); Cc[0,0] = 1.0; Kc = 1
            else:
                if init_mode.lower() == 'kmeans':
                    if searchK and K is None:
                        Kc, sil, Cc = search_best_K_per_class(Xc, Kmax=Kmax, seed=42, sample_weights=None)
                    else:
                        Kc = K if K is not None else min(2, max(1, Xc.shape[0]//10))
                        Cc, _ = kmeans_l2(Xc, K=Kc, iters=80, seed=42, sample_weights=None)
                elif init_mode.lower() == 'nmf':
                    if searchK and K is None:
                        Kc = _search_best_K_per_class_nmf(Xc, Kmax=Kmax, beta_loss=nmf_beta,
                                                          max_iter=max(80, nmf_max_iter//2), random_state=nmf_random_state)
                    else:
                        Kc = K if K is not None else 1
                    xmin = Xc.min(axis=0, keepdims=True); xmax = Xc.max(axis=0, keepdims=True)
                    scale = np.maximum(xmax - xmin, 1e-6); Xs = (Xc - xmin) / scale
                    W, H, _ = non_negative_factorization(
                        Xs, n_components=Kc, init='nndsvd', solver='mu', beta_loss=nmf_beta,
                        max_iter=nmf_max_iter, tol=1e-6, random_state=nmf_random_state,
                        alpha_H=nmf_alphaH, l1_ratio=nmf_l1_ratio, alpha_W=0.0
                    )
                    H_orig = H * scale + xmin
                    rho_bar = float(np.linalg.norm(Xc, axis=1).mean())
                    Cc = _rescale_rows_to_norm(
                        H_orig,
                        np.full((Kc,), rho_bar, dtype=np.float32),
                    ).astype(np.float32)
                elif init_mode.lower() == 'svd':
                    # SVD (PCA-style) prototype: top-Kc right singular directions of the
                    # class feature matrix, sign-aligned to the class mean and rescaled to
                    # the per-class mean feature norm -- the linear-algebra analogue of the
                    # NMF prototype (same rho_bar rescaling, no non-negativity constraint).
                    Kc = K if K is not None else 1
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                    comps = Vt[:Kc].astype(np.float32).copy()
                    mu_c = Xc.mean(axis=0)
                    for k in range(comps.shape[0]):
                        if float(comps[k] @ mu_c) < 0:
                            comps[k] = -comps[k]
                    rho_bar = float(np.linalg.norm(Xc, axis=1).mean())
                    Cc = _rescale_rows_to_norm(
                        comps,
                        np.full((Kc,), rho_bar, dtype=np.float32),
                    ).astype(np.float32)
                else:
                    raise ValueError(f'Unknown init_mode={init_mode}')
            centers.append(Cc); self.per_class_K.append(int(Kc))
            self.offsets.append((int(base), int(base + Kc))); base += Kc
        C_all = np.vstack(centers).astype(np.float32)
        self.mu = torch.from_numpy(C_all).to(self.device)
        print(f'[Init:{init_mode}] prototypes: SumK={self.mu.shape[0]} | per-class={self.per_class_K}')

    @torch.no_grad()
    def _proto_slices(self):
        return [slice(s,e) for (s,e) in self.offsets]

    @torch.no_grad()
    def soft_assign(self, f: torch.Tensor):
        assert self.mu is not None, 'PrototypeBank 未初始化'
        f = l2n(f, dim=1); mu = l2n(self.mu, dim=1)
        sim = (f @ mu.t()) / self.temp_proto
        q = torch.softmax(sim, dim=1)        # [N,SumK]
        N = f.size(0)
        p_cls = torch.zeros(N, self.C, device=f.device, dtype=q.dtype)
        for c, (s,e) in enumerate(self.offsets): p_cls[:, c] = q[:, s:e].sum(dim=1)
        return q, p_cls

    @torch.no_grad()
    def nmf_assign(self, f: torch.Tensor, beta_loss='frobenius', iters: int = 60):
        assert self.mu is not None, 'PrototypeBank 未初始化'
        X = f.detach().float().cpu().numpy(); H = self.mu.detach().float().cpu().numpy()
        Xs, Cs = minmax_fit_transform_together(X, H)
        W_init = np.maximum(Xs @ Cs.T, 1e-6)
        W, H_fix, _ = non_negative_factorization(
            Xs, W=W_init, H=Cs, init='custom', update_H=False, solver='mu',
            beta_loss=beta_loss, max_iter=iters, tol=1e-6, random_state=0
        )
        W = row_normalize(W)
        q = torch.from_numpy(W).to(f.device)
        N = q.size(0)
        p_cls = torch.zeros(N, self.C, device=f.device, dtype=q.dtype)
        for c, (s, e) in enumerate(self.offsets): p_cls[:, c] = q[:, s:e].sum(dim=1)
        return q, p_cls

    @torch.no_grad()
    def momentum_update(self, f_sel: torch.Tensor, q_sel: torch.Tensor):
        # Responsibility-weighted EMA update. Prototypes that receive no
        # responsibility mass in this update (e.g. empty classes) are left
        # UNCHANGED, matching the paper: s_c = 0  =>  mu_c^{(r)} = mu_c^{(r-1)}.
        if f_sel.numel() == 0:
            return
        f_sel = f_sel.detach()
        q_sel = q_sel.detach()
        w = q_sel.sum(dim=0)                      # [SumK] responsibility mass per prototype
        valid = w > 1e-6
        if not valid.any():
            return
        bar = torch.zeros_like(self.mu)
        bar[valid] = (q_sel[:, valid].t() @ f_sel) / w[valid].unsqueeze(1)
        self.mu[valid] = self.proto_m * self.mu[valid] + (1 - self.proto_m) * bar[valid]

    @torch.no_grad()
    def momentum_update_masked(self, f_sel: torch.Tensor, q_sel: torch.Tensor, y_hat_sel: torch.Tensor):
        # Restrict each sample's responsibility to the prototype slice of its
        # pseudo-label class while PRESERVING the raw responsibility values.
        # (Do NOT renormalize each row to 1, otherwise the weighting degenerates
        # to a plain mean and the responsibility-weighted update is lost.)
        if f_sel.numel() == 0:
            return
        q_masked = torch.zeros_like(q_sel)
        for c, (s, e) in enumerate(self.offsets):
            m = y_hat_sel.eq(c)
            if m.any():
                q_masked[m, s:e] = q_sel[m, s:e]
        self.momentum_update(f_sel, q_masked)

    @torch.no_grad()
    def nmf_refresh(self, feats_pool: np.ndarray, max_iter: int = 300, beta_loss: str = 'frobenius',
                    alphaH: float = 0.0, l1_ratio: float = 0.0, random_state: int = 42):
        if feats_pool.shape[0] < self.mu.shape[0] * 2: return
        X = feats_pool.astype(np.float32)
        H_init = self.mu.detach().float().cpu().numpy()
        Z = np.vstack([X, H_init]).astype(np.float32)
        mn = Z.min(axis=0, keepdims=True)
        mx = Z.max(axis=0, keepdims=True)
        rng = np.maximum(mx - mn, 1e-12)
        Xs, Cs = (X - mn) / rng, (H_init - mn) / rng
        W_init = np.maximum(Xs @ Cs.T, 1e-6)
        W, H, n_iter = non_negative_factorization(
            Xs, W=W_init, H=Cs, init='custom', update_H=True, solver='mu',
            beta_loss=beta_loss, max_iter=max_iter, random_state=random_state, tol=1e-6,
            alpha_H=alphaH, l1_ratio=l1_ratio, alpha_W=0.0
        )
        perm = align_components_by_cos(H, Cs); H = H[perm, :]
        H_orig = H * rng + mn

        P = row_normalize(W_init)
        p_cls = np.zeros((P.shape[0], self.C), dtype=np.float32)
        for c, (s, e) in enumerate(self.offsets):
            p_cls[:, c] = P[:, s:e].sum(axis=1)
        y_hat = p_cls.argmax(axis=1)
        global_rho = float(np.linalg.norm(X, axis=1).mean())
        target_norms = []
        for c, (s, e) in enumerate(self.offsets):
            Xc = X[y_hat == c]
            rho_c = float(np.linalg.norm(Xc, axis=1).mean()) if Xc.shape[0] > 0 else global_rho
            target_norms.extend([rho_c] * (e - s))
        H_orig = _rescale_rows_to_norm(H_orig, np.asarray(target_norms, dtype=np.float32))

        self.mu = torch.from_numpy(H_orig.astype(np.float32)).to(self.device)
        print(f'[NMF] refreshed H with {feats_pool.shape[0]} feats | iters={n_iter}')

# 辅助：按NMF重构误差“肘部法”自动选 K_c
def _search_best_K_per_class_nmf(Xc: np.ndarray, Kmax: int, beta_loss: str='frobenius',
                                 max_iter: int=100, random_state: int=42, rel_improve_thr: float=0.05) -> int:
    if Xc.shape[0] < 2: return 1
    xmin = Xc.min(axis=0, keepdims=True); xmax = Xc.max(axis=0, keepdims=True)
    scale = np.maximum(xmax - xmin, 1e-6); Xs = (Xc - xmin) / scale
    prev_err = None; best_k = 1
    for k in range(1, max(2, Kmax)+1):
        W, H, _ = non_negative_factorization(
            Xs, n_components=k, init='nndsvd', solver='mu', beta_loss=beta_loss,
            max_iter=max_iter, tol=1e-6, random_state=random_state
        )
        recon = W @ H; err = np.linalg.norm(Xs - recon, ord='fro')
        if prev_err is not None:
            rel_improve = (prev_err - err) / max(prev_err, 1e-8)
            if rel_improve < rel_improve_thr: break
        prev_err = err; best_k = k
    return best_k
