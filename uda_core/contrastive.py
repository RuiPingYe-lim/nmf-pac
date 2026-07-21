# l2归一化/InfoNCE/熵最小化
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def l2n(x: torch.Tensor, dim=1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def pairwise_logits(z_q, z_k, tau: float):
    return (z_q @ z_k.t()) / tau

def entropy_minimization(probs: torch.Tensor):
    return -(probs * torch.log(torch.clamp(probs, 1e-8, 1.0))).sum(dim=1).mean()

def symmetric_infoNCE(z_s, y_s, z_t, y_t, tau=0.07):
    z_s = l2n(z_s, dim=1); z_t = l2n(z_t, dim=1)
    logits_st = pairwise_logits(z_s, z_t, tau=tau)        # [Ns,Nt]
    y_s_ = y_s.view(-1, 1); y_t_ = y_t.view(1, -1)
    pos_mask_st = (y_s_ == y_t_).float()
    loss_st = torch.tensor(0.0, device=z_s.device)
    valid_rows = (pos_mask_st.sum(dim=1) > 0)
    if valid_rows.any():
        logp_st = F.log_softmax(logits_st[valid_rows], dim=1)
        loss_st = -(logp_st * pos_mask_st[valid_rows]).sum() / pos_mask_st[valid_rows].sum().clamp_min(1.0)
    logits_ts = pairwise_logits(z_t, z_s, tau=tau)        # [Nt,Ns]
    pos_mask_ts = pos_mask_st.t()
    loss_ts = torch.tensor(0.0, device=z_s.device)
    valid_rows2 = (pos_mask_ts.sum(dim=1) > 0)
    if valid_rows2.any():
        logp_ts = F.log_softmax(logits_ts[valid_rows2], dim=1)
        loss_ts = -(logp_ts * pos_mask_ts[valid_rows2]).sum() / pos_mask_ts[valid_rows2].sum().clamp_min(1.0)
    return 0.5 * (loss_st + loss_ts)