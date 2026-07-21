# -*- coding: utf-8 -*-
 # 可复现设置与清单
import os, json, random
import numpy as np
import torch
import pandas as pd
import sklearn


def set_global_seed(seed: int = 42, deterministic: bool = True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except TypeError:
            pass
        torch.backends.cudnn.benchmark = True


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)
    return _init_fn


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def repro_manifest(save_dir: str, seed: int, deterministic: bool):
    meta = {
        'seed': int(seed),
        'deterministic': bool(deterministic),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', ''),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
        'torch_version': torch.__version__,
        'cuda_version': getattr(torch.version, 'cuda', None),
        'cudnn_version': torch.backends.cudnn.version(),
        'torch_cudnn_deterministic': torch.backends.cudnn.deterministic,
        'torch_cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'run_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('[Repro] run_manifest.json saved:', meta)
