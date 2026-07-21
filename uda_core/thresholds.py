import numpy as np
import torch

class ClasswiseEMAThreshold:
    def __init__(self, num_classes: int, ema_lambda: float = 0.95, tau_floor: float = 0.0, use_floor: bool = False, lam_lo: float = 0.70, lam_hi: float = 0.95, tau_base: float = 0.75, tau_base_w: float = 0.0):
        self.C = int(num_classes)
        self.lam = float(ema_lambda)
        self.t = 0
        self.tau_t = np.ones(self.C) / self.C  # 按类别初始化阈值
        self.p_tilde = np.ones(self.C, dtype=np.float32) / self.C
        self.tau_floor = float(tau_floor)
        self.use_floor = bool(use_floor)
        self.lam_lo = float(lam_lo)
        self.lam_hi = float(lam_hi)
        self.tau_base = float(tau_base)
        self.tau_base_w = float(tau_base_w)
    #等权方差
    # def update_and_get(self, q_batch: torch.Tensor) -> torch.Tensor:
    #     assert q_batch.ndim == 2 and q_batch.size(1) == self.C  # 确保 q_batch 形状正确
    #     qb = q_batch.detach().float().cpu().numpy()  # 获取当前批次的预测置信度
    #     if self.t % 50 == 0:  # 每 50 个批次打印一次，避免日志过多
    #         with np.printoptions(precision=4, suppress=True):
    #             print(f"[dbg] qb shape={qb.shape}\n{qb[:31]}")
    #     # 计算每个样本的最大置信度
    #     mean_max = np.mean(np.max(qb, axis=1)) if qb.shape[0] > 0 else (1.0 / self.C)
    #     print("Mean Max: ", mean_max,self.C)  # 打印 mean_max
        
        
    #     class_difficulty = np.var(qb, axis=0)  # 计算置信度的方差，作为难度的度量
    #     print("Class Difficulty: ", class_difficulty,self.C)  # 打印每个类别的难度
        
    #     # 动态调整每个类别的更新速率（动态增长因子）
    #     dynamic_lambda = np.clip(1.0 - class_difficulty, 0.7, 0.95)  # 难度大的类别增长因子较小
    #     print("Dynamic Lambda: ", dynamic_lambda,self.C)  # 打印 dynamic_lambda
        
    #     # 更新每个类别的阈值
    #     if self.t == 0:
    #         self.tau_t = np.ones(self.C, dtype=np.float32) / self.C  # 初始时均匀分布
    #     else:
    #         for class_idx in range(self.C):
    #             # 打印当前类别的阈值更新信息
    #             print(f"Updating class {class_idx} - Current tau: {self.tau_t[class_idx]:.4f} | Dynamic Lambda: {dynamic_lambda[class_idx]:.4f}")
    #             self.tau_t[class_idx] = dynamic_lambda[class_idx] * self.tau_t[class_idx] + (1.0 - dynamic_lambda[class_idx]) * mean_max
    #             # 打印更新后的阈值
    #             print(f"New tau for class {class_idx}: {self.tau_t[class_idx]:.4f}")
    #     self.t += 1  # 更新批次计数器

    #     # 应用阈值下限
    #     if self.use_floor:
    #         self.tau_t = np.maximum(self.tau_t, self.tau_floor)
        
    #     return torch.from_numpy(self.tau_t.astype(np.float32))  # 返回阈值映射
    def update_and_get(self, q_batch: torch.Tensor) -> torch.Tensor:
        assert q_batch.ndim == 2 and q_batch.size(1) == self.C
        qb = q_batch.detach().float().cpu().numpy()    # [N, C]
        eps = 1e-8

        # ----- 按类的软加权均值/方差（条件于该类）-----
        w = qb                                        # 责任度
        w_sum = w.sum(axis=0) + eps                   # [C]
        mu_k = (w * qb).sum(axis=0) / w_sum           # 每类“期望置信度” E[p_k | k]，形状 [C]
        var_k = ((w * (qb - mu_k)**2).sum(axis=0) / w_sum).astype(np.float32)  # 加权方差 [C]

        # ----- 方差 -> 动态 EMA 系数 lambda_k -----
        lam_lo, lam_hi = self.lam_lo, self.lam_hi                   # 可按需调
        var_norm = np.clip(var_k / 0.25, 0.0, 1.0)    # Bernoulli 上界 0.25 归一化
        dynamic_lambda = lam_hi - (lam_hi - lam_lo) * var_norm  # 难→var大→λ小

        # ----- 初始化 or EMA 更新 -----
        if self.t == 0:
            self.tau_t = np.ones(self.C, dtype=np.float32) / self.C  # e.g., C=2 -> 0.5
        else:
            target_k = self.tau_base_w * self.tau_base + (1.0 - self.tau_base_w) * mu_k
            self.tau_t = dynamic_lambda * self.tau_t + (1.0 - dynamic_lambda) * target_k

        self.t += 1

        # （可选）同步更新类先验 p_tilde（如果你后续用得到）
        # self.p_tilde = 0.9 * self.p_tilde + 0.1 * (w_sum / w.shape[0])

        if self.use_floor:
            self.tau_t = np.maximum(self.tau_t, self.tau_floor)

        return torch.from_numpy(self.tau_t.astype(np.float32))


    def current_tau_map(self) -> torch.Tensor:
        if self.use_floor:
            self.tau_t = np.maximum(self.tau_t, self.tau_floor)
        return torch.from_numpy(self.tau_t.astype(np.float32))