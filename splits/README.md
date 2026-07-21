# Data splits

Fixed train/validation/test split assignments used for all experiments in the
paper. These files contain **only case identifiers and labels** (and, for the
breast set, relative image paths and class names) — **no raw medical images are
redistributed**. Download the images from the original providers (see the main
`README.md`) and map them with these files to reproduce the exact splits.

Splits are fixed before training. Target-domain labels are used **only** for
final evaluation — never for training, pseudo-labeling, thresholding, or
checkpoint selection.

## Layout and format

| Modality | Files | Columns |
|---|---|---|
| Knee MRI (`knee/KneeMRI`, `knee/MRNet-v1.0`) | `train_0.csv`, `valid_0.csv`, `test_0.csv` | `case_id,label` |
| Breast US (`breast/BUSI`, `breast/BrEaST`) | `train_0.csv`, `valid_0.csv`, `test_0.csv` | `case_id,image_path,label,class_name,domain` |
| Knee X-ray (`xray/`) | `split_kneeKL224.csv`, `split_MedicalExpert-II.csv` (single file, `split` column) | `split,label,filename` |

`label`: 0 = negative/normal, 1 = positive (ACL injury for knee MRI, KL≥2
osteoarthritis for knee X-ray, malignant for breast US).

## Per-class counts

| Dataset | Train (c0/c1) | Val (c0/c1) | Test (c0/c1) | Total |
|---|---|---|---|---|
| MRNet-v1.0 | 1010 (856/154) | 120 (66/54) | 120 (66/54) | 1250 |
| KneeMRI | 640 (483/157) | 93 (69/24) | 184 (138/46) | 917 |
| kneeKL224 | 4732 (2286/2446) | 673 (328/345) | 1360 (639/721) | 6765 |
| MedicalExpert-II | 712 (312/400) | 101 (44/57) | 205 (90/115) | 1018 |
| BUSI | 452 (305/147) | 65 (44/21) | 130 (88/42) | 647 |
| BrEaST | 175 (107/68) | 26 (16/10) | 51 (31/20) | 252 |
