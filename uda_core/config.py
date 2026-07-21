# -*- coding: utf-8 -*-
import argparse

def build_parser():
    ap = argparse.ArgumentParser(
        'End-to-End UDA (Prototypes + NMF Pseudo + Classwise EMA Threshold + Symmetric InfoNCE)'
    )
    # ===== data =====
    ap.add_argument('--npy_src', required=True)
    ap.add_argument('--csv_src', required=True)
    ap.add_argument('--csv_src_val', default='')
    ap.add_argument('--npy_tgt', required=True)
    ap.add_argument('--csv_tgt', required=True)
    ap.add_argument('--npy_tgt_test', default='')
    ap.add_argument('--csv_tgt_test', default='')
    ap.add_argument('--plane', default='sagittal', choices=['sagittal', 'coronal', 'axial'])
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument(
        '--volume_proj',
        default='mean',
        choices=['mean', 'center', 'max', 'median'],
        help='How to convert each MRI volume/sequence into a 2D case-level image.',
    )

    ap.add_argument('--id_col_src', default='case_id')
    ap.add_argument('--label_col_src', default='label')
    ap.add_argument('--single_file_case_src', action='store_true')
    ap.add_argument('--id_zero_pad_src', type=int, default=None)

    ap.add_argument('--id_col_tgt', default='case_id')
    ap.add_argument('--single_file_case_tgt', action='store_true')
    ap.add_argument('--id_zero_pad_tgt', type=int, default=None)
    ap.add_argument('--id_col_tgt_test', default='case_id')
    ap.add_argument('--label_col_tgt_test', default='label')
    ap.add_argument('--single_file_case_tgt_test', action='store_true')
    ap.add_argument('--id_zero_pad_tgt_test', type=int, default=None)

    # ===== model =====
    ap.add_argument('--backbone', default='custom_resnet50_space')
    ap.add_argument('--pretrained', default='imagenet')
    ap.add_argument('--init_from', default='', help='可选初始化权重 .pth 路径')
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--proj_dim', type=int, default=256)

    # ===== train (common) =====
    ap.add_argument('--epochs', type=int, default=20)   # 兼容旧字段
    ap.add_argument('--bs_src', type=int, default=1)
    ap.add_argument('--bs_tgt', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true', default=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_dir', default='./checkpoints_online')
    ap.add_argument('--workers', type=int, default=8)

    # ===== prototypes / NMF =====
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--Kmax', type=int, default=4)
    ap.add_argument('--proto_m', type=float, default=0.95)
    ap.add_argument('--tau_proto', type=float, default=0.10)
    ap.add_argument('--refresh_every', type=int, default=0)
    ap.add_argument('--nmf_pool_size', type=int, default=4000)
    ap.add_argument('--nmf_max_iter', type=int, default=300)
    ap.add_argument('--beta_loss', default='frobenius',
                    choices=['frobenius', 'kullback-leibler', 'itakura-saito'])
    ap.add_argument('--alphaH', type=float, default=0.0)
    ap.add_argument('--l1_ratio', type=float, default=0.0)

    # ===== prototype init mode =====
    ap.add_argument('--proto_init', default='kmeans', choices=['kmeans', 'nmf', 'svd'])
    ap.add_argument('--nmf_init_beta', default='frobenius',
                    choices=['frobenius', 'kullback-leibler', 'itakura-saito'])
    ap.add_argument('--nmf_init_max_iter', type=int, default=150)
    ap.add_argument('--nmf_init_alphaH', type=float, default=0.0)
    ap.add_argument('--nmf_init_l1_ratio', type=float, default=0.0)

    # ===== FreeMatch-style EMA =====
    ap.add_argument('--tau_base', type=float, default=0.75)
    ap.add_argument('--tau_min', type=float, default=0.55)
    ap.add_argument('--cover_max', type=float, default=0.6)
    ap.add_argument('--lam_lo', type=float, default=0.70)
    ap.add_argument('--lam_hi', type=float, default=0.95)
    ap.add_argument('--tau_base_w', type=float, default=0.0)
    ap.add_argument('--ema_m', type=float, default=0.95)
    ap.add_argument('--fixed_tau', type=float, default=None,
                    help='If set, disable dynamic threshold updates and use a fixed classwise threshold.')

    # ===== simplified round-based =====
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument(
        '--early_stop_patience',
        type=int,
        default=0,
        help='Stop after this many rounds without src_val_auc improvement. 0 disables early stopping.',
    )
    ap.add_argument(
        '--early_stop_min_rounds',
        type=int,
        default=0,
        help='Do not early-stop before this many rounds have completed.',
    )
    ap.add_argument('--epochs_contrast', type=int, default=1)
    ap.add_argument('--use_nmf_pseudo', action='store_true', default=False)
    ap.add_argument('--mask_cross_class_update', action='store_true', default=True)
    ap.add_argument('--nmf_assign_iters', type=int, default=60)
    ap.add_argument('--lam_src_ce', type=float, default=0.2)
    ap.add_argument('--lam_con', type=float, default=1.0)
    ap.add_argument('--tau_con', type=float, default=0.07)
    # Post-Stage-B re-running of NMF on target features. OFF by default to match
    # the paper (Sec. 3.6: "This process does not re-run NMF on them"). Pass the
    # flag only to ablate this component.
    ap.add_argument('--nmf_refresh_each_round', action='store_true', default=False)

    return ap
