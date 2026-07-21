# 伪标签子集封装
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

def _norm_id(x):
    return str(x).strip()

class NPYPseudoDataset(Dataset):
    """用 {case_id -> y_hat} 字典，匹配到 base_ds 的索引。base_ds 为 NPYInferDataset。"""
    def __init__(self, base_ds, id2y: dict):
        self.base = base_ds
        id2y = { _norm_id(k): int(v) for k, v in id2y.items() }
        base_ids = None
        for attr in ('case_ids','ids','id_list','keys'):
            if hasattr(base_ds, attr):
                val = getattr(base_ds, attr)
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    base_ids = [ _norm_id(c) for c in val ]; break
        keep_idx, keep_y = [], []
        if base_ids is not None:
            id2idx = { cid: i for i, cid in enumerate(base_ids) }
            for cid, y in id2y.items():
                i = id2idx.get(cid, None)
                if i is not None: keep_idx.append(i); keep_y.append(y)
        else:
            for i in range(len(base_ds)):
                _, cid = base_ds[i]; cid = _norm_id(cid)
                if cid in id2y: keep_idx.append(i); keep_y.append(id2y[cid])
        self.keep_idx = keep_idx; self.keep_y = keep_y
        print(f'[PseudoDS] matched {len(self.keep_idx)} / {len(id2y)} selected ids (base={len(base_ds)})')
    def __len__(self):
        return len(self.keep_idx)
    def __getitem__(self, i):
        idx = self.keep_idx[i]
        xb, _cid = self.base[idx]
        return xb, self.keep_y[i]