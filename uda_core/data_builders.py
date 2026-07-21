# 源/目标 DataLoader 构建
# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data import DataLoader
from .repro import make_worker_init_fn, make_generator
from data import NPYSliceDataset, NPYInferDataset

def build_src_loader(npy_root, csv, plane, resize, id_col, label_col, single_file_case, id_zero_pad,
                     batch_size, workers, seed, volume_proj='mean'):
    ds = NPYSliceDataset(
        npy_root=npy_root, csv_file=csv, plane=plane,
        id_col=id_col, label_col=label_col, resize=resize,
        single_file_case=single_file_case, id_zero_pad=id_zero_pad,
        augment=True, volume_proj=volume_proj
    )
    worker_init = make_worker_init_fn(seed + 11)
    gen = make_generator(seed + 101)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, generator=gen, persistent_workers=(workers > 0)
    )

def build_src_proto_loader(npy_root, csv, plane, resize, id_col, label_col, single_file_case, id_zero_pad,
                           batch_size, workers, seed, volume_proj='mean'):
    # Deterministic loader for source NMF-prototype initialization:
    # no augmentation, no shuffling, no dropping of the last batch, so the
    # prototypes are built from the full, fixed source training set.
    ds = NPYSliceDataset(
        npy_root=npy_root, csv_file=csv, plane=plane,
        id_col=id_col, label_col=label_col, resize=resize,
        single_file_case=single_file_case, id_zero_pad=id_zero_pad,
        augment=False, volume_proj=volume_proj
    )
    worker_init = make_worker_init_fn(seed + 11)
    gen = make_generator(seed + 101)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False,
        worker_init_fn=worker_init, generator=gen, persistent_workers=(workers > 0)
    )

def build_tgt_loader(npy_root, csv_ids, plane, resize, id_col, single_file_case, id_zero_pad,
                     batch_size, workers, seed, volume_proj='mean'):
    df = pd.read_csv(csv_ids, dtype={id_col: str})
    case_ids = df[id_col].astype(str).str.strip().tolist()
    ds = NPYInferDataset(
        npy_root=npy_root, case_ids=case_ids, plane=plane,
        resize=resize, single_file_case=single_file_case, id_zero_pad=id_zero_pad,
        volume_proj=volume_proj
    )
    worker_init = make_worker_init_fn(seed + 22)
    gen = make_generator(seed + 202)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init, generator=gen, persistent_workers=(workers > 0)
    )

# 目标域评估 loader（有标签，augment=False, shuffle=False）
def build_tgt_test_loader(npy_root, csv, plane, resize, id_col, label_col, single_file_case, id_zero_pad,
                          batch_size, workers, seed, volume_proj='mean'):
    ds = NPYSliceDataset(
        npy_root=npy_root, csv_file=csv, plane=plane,
        id_col=id_col, label_col=label_col, resize=resize,
        single_file_case=single_file_case, id_zero_pad=id_zero_pad,
        augment=False, volume_proj=volume_proj
    )
    worker_init = make_worker_init_fn(seed + 33)
    gen = make_generator(seed + 303)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False,
        worker_init_fn=worker_init, generator=gen, persistent_workers=(workers > 0)
    )
