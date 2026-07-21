# -*- coding: utf-8 -*-
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_read_csv(path: str, id_col_default: str = "case_id") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={id_col_default: str})
    if id_col_default not in df.columns:
        df = df.rename(columns={df.columns[0]: id_col_default})
    df[id_col_default] = df[id_col_default].astype(str)
    return df


def build_transform(resize: int, augment: bool = False) -> T.Compose:
    aug = [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10)] if augment else []
    base = [
        T.ToTensor(),
        T.Resize((resize, resize), antialias=True),
        T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return T.Compose(aug + base)


def list_stem_lengths(dirpath: str) -> List[int]:
    stems = []
    for f in glob.glob(os.path.join(dirpath, "*.npy")):
        stem = os.path.splitext(os.path.basename(f))[0]
        stems.append(len(stem.split("-")[0]))
    return sorted(list(set(stems)))


def find_files_for_case(
    npy_root: str,
    plane: str,
    case_id_raw: str,
    single_file_case: bool,
    zero_pad: Optional[int],
) -> List[str]:
    base = os.path.join(npy_root, plane) if plane else npy_root
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Directory not found: {base}")

    case_id_raw = str(case_id_raw)
    pads = [4, 6, 8, 10]
    if zero_pad and int(zero_pad) not in pads:
        pads = [int(zero_pad)] + pads

    candidates = [case_id_raw]
    if case_id_raw.isdigit():
        candidates += [case_id_raw.zfill(z) for z in pads]
    candidates = list(dict.fromkeys(candidates))

    stem_lengths = list_stem_lengths(base)

    if single_file_case:
        for cid in candidates:
            f = os.path.join(base, f"{cid}.npy")
            if os.path.isfile(f):
                return [f]
        if case_id_raw.isdigit():
            for z in stem_lengths:
                cid = case_id_raw.zfill(z)
                f = os.path.join(base, f"{cid}.npy")
                if os.path.isfile(f):
                    return [f]
    else:
        for cid in candidates:
            g = sorted(glob.glob(os.path.join(base, f"{cid}-*.npy")))
            if g:
                return g
        if case_id_raw.isdigit():
            for z in stem_lengths:
                cid = case_id_raw.zfill(z)
                g = sorted(glob.glob(os.path.join(base, f"{cid}-*.npy")))
                if g:
                    return g
    return []


def intensity_norm01(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.astype(np.float32)


def volume_to_2d(arr: np.ndarray, proj: str = "mean") -> np.ndarray:
    arr = np.asarray(arr)
    proj = str(proj).lower()
    if proj not in {"mean", "center", "max", "median"}:
        raise ValueError(f"Unsupported volume projection: {proj}")

    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        if proj == "center":
            img = arr[arr.shape[0] // 2]
        else:
            s, h, w = arr.shape[0], arr.shape[-2], arr.shape[-1]
            looks_like_shw = s >= 4 and h >= 32 and w >= 32
            looks_like_hwc = arr.shape[-1] in (1, 3) and arr.shape[0] == h and arr.shape[1] == w
            if looks_like_shw and not looks_like_hwc:
                if proj == "max":
                    img = arr.max(axis=0)
                elif proj == "median":
                    img = np.median(arr, axis=0)
                else:
                    img = arr.mean(axis=0)
            else:
                c = arr.shape[-1]
                img = arr[..., 0] if c == 1 else arr.mean(axis=-1)
    else:
        img = arr[arr.shape[0] // 2]

    return intensity_norm01(img.astype(np.float32))


class NPYSliceDataset(Dataset):
    """Labeled dataset for train/val/test. Optionally returns case_id."""

    def __init__(
        self,
        npy_root,
        csv_file,
        plane="sagittal",
        id_col="case_id",
        label_col="label",
        resize=224,
        single_file_case=False,
        id_zero_pad=None,
        augment=False,
        return_case_id=False,
        volume_proj="mean",
    ):
        super().__init__()
        self.npy_root = npy_root
        self.plane = plane
        self.single_file_case = single_file_case
        self.id_zero_pad = id_zero_pad
        self.return_case_id = bool(return_case_id)
        self.volume_proj = str(volume_proj).lower()

        df = try_read_csv(csv_file, id_col_default=id_col)
        if label_col not in df.columns:
            raise ValueError(f"{csv_file} missing label column '{label_col}'")

        self.samples = []  # (path, y, case_id)
        for _, row in df.iterrows():
            cid = str(row[id_col])
            y = int(row[label_col])
            files = find_files_for_case(npy_root, plane, cid, single_file_case, id_zero_pad)
            if not files:
                mode = "single-file" if single_file_case else "multi-slice"
                raise FileNotFoundError(f"[src] missing {mode} sample: {cid}")
            if single_file_case:
                self.samples.append((files[0], y, cid))
            else:
                self.samples += [(f, y, cid) for f in files]

        self.tf = build_transform(int(resize), augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, cid = self.samples[idx]
        arr = np.load(path, allow_pickle=False)
        img2d = volume_to_2d(arr, proj=self.volume_proj)
        x = self.tf(Image.fromarray(img2d))
        y_t = torch.tensor(y, dtype=torch.long)
        if self.return_case_id:
            return x, y_t, cid
        return x, y_t


class NPYInferDataset(Dataset):
    """Target-domain inference dataset: returns (x, case_id)."""

    def __init__(
        self,
        npy_root,
        case_ids,
        plane="sagittal",
        resize=224,
        single_file_case=True,
        id_zero_pad=None,
        volume_proj="mean",
    ):
        super().__init__()
        self.items = []
        self.volume_proj = str(volume_proj).lower()
        self.tf = T.Compose(
            [
                T.ToTensor(),
                T.Resize((resize, resize), antialias=True),
                T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        for cid in case_ids:
            files = find_files_for_case(npy_root, plane, cid, single_file_case, id_zero_pad)
            if not files:
                mode = "single-file" if single_file_case else "multi-slice"
                raise FileNotFoundError(f"[infer] missing {mode} sample: {cid}")
            if single_file_case:
                self.items.append((files[0], str(cid)))
            else:
                self.items += [(f, str(cid)) for f in files]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cid = self.items[idx]
        arr = np.load(path)
        img2d = volume_to_2d(arr, proj=self.volume_proj)
        return self.tf(Image.fromarray(img2d)), cid
