#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect multi-seed UDA baseline results.

The script scans experiment folders for best_metrics.json, metrics_summary.json,
or best_by_src_val.json, then writes:
  - per-seed records
  - method-level mean/std summary
  - a Markdown table for reports

Examples:
  python scripts/collect_multiseed_results.py \
    --root outputs/baselines_mrnet_to_knee_center \
    --direction MRNet_to_KneeMRI \
    --out_dir outputs/summary/mrnet_to_knee

  python scripts/collect_multiseed_results.py \
    --root knee2mrnet/outputs/baselines_knee_to_mrnet_multiseed_native_hparams \
    --direction KneeMRI_to_MRNet \
    --out_dir outputs/summary/knee_to_mrnet
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


METRIC_FILES = ("best_metrics.json", "metrics_summary.json", "best_by_src_val.json", "metrics.json")
REPORT_METRICS = [
    "tgt_auc",
    "tgt_acc",
    "tgt_prec",
    "tgt_rec",
    "tgt_spec",
    "tgt_f1",
    "src_val_auc",
    "best_epoch",
]


def load_json(path: str) -> Optional[Dict]:
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            continue
    return None


def safe_float(value):
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean_std(values: Iterable[float]):
    vals = finite_values(values)
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def fmt_mean_std(mean: float, std: float, digits: int) -> str:
    if not math.isfinite(mean):
        return ""
    if not math.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def nested_get(data: Dict, path: List[str]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_number(data: Dict, paths: List[List[str]]) -> float:
    for path in paths:
        val = nested_get(data, path)
        out = safe_float(val)
        if math.isfinite(out):
            return out
    return float("nan")


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for y, p in zip(y_true, y_pred):
        if y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
        elif y == 1 and p == 1:
            tp += 1
    return tn, fp, fn, tp


def div0(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def prf_from_counts(tn: int, fp: int, fn: int, tp: int, average: str) -> Tuple[float, float, float]:
    prec_pos = div0(tp, tp + fp)
    rec_pos = div0(tp, tp + fn)
    f1_pos = div0(2.0 * prec_pos * rec_pos, prec_pos + rec_pos)

    prec_neg = div0(tn, tn + fn)
    rec_neg = div0(tn, tn + fp)
    f1_neg = div0(2.0 * prec_neg * rec_neg, prec_neg + rec_neg)

    if average == "binary":
        return prec_pos, rec_pos, f1_pos
    if average == "macro":
        return (
            0.5 * (prec_neg + prec_pos),
            0.5 * (rec_neg + rec_pos),
            0.5 * (f1_neg + f1_pos),
        )
    if average == "weighted":
        n0 = tn + fp
        n1 = fn + tp
        total = n0 + n1
        return (
            div0(n0 * prec_neg + n1 * prec_pos, total),
            div0(n0 * rec_neg + n1 * rec_pos, total),
            div0(n0 * f1_neg + n1 * f1_pos, total),
        )
    raise ValueError(f"Unsupported average: {average}")


def load_prediction_labels(path: str) -> Optional[Tuple[List[int], List[int]]]:
    if not os.path.isfile(path):
        return None
    y_true: List[int] = []
    y_pred: List[int] = []
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "label" not in reader.fieldnames or "pred" not in reader.fieldnames:
                return None
            for row in reader:
                try:
                    y_true.append(int(float(row["label"])))
                    y_pred.append(int(float(row["pred"])))
                except Exception:
                    continue
    except Exception:
        return None
    if not y_true or len(y_true) != len(y_pred):
        return None
    return y_true, y_pred


def recompute_classification_metrics(exp_dir: str, split: str, average: str) -> Dict[str, float]:
    names = [f"pred_{split}_best.csv", f"pred_{split}.csv"]
    for name in names:
        loaded = load_prediction_labels(os.path.join(exp_dir, name))
        if loaded is None:
            continue
        y_true, y_pred = loaded
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        prec, rec, f1 = prf_from_counts(tn, fp, fn, tp, average=average)
        acc = div0(tn + tp, tn + fp + fn + tp)
        spec = div0(tn, tn + fp)
        return {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "spec": spec,
            "f1": f1,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    return {}


def extract_metrics(data: Dict) -> Dict[str, float]:
    return {
        "src_val_auc": first_number(
            data,
            [
                ["src_val_auc"],
                ["best_src_val_auc"],
                ["best_src_val_metrics", "auc"],
                ["src_val", "auc"],
                ["source_val", "auc"],
            ],
        ),
        "src_val_acc": first_number(
            data,
            [
                ["src_val_acc"],
                ["best_src_val_acc"],
                ["best_src_val_metrics", "acc"],
                ["src_val", "acc"],
            ],
        ),
        "src_val_f1": first_number(
            data,
            [
                ["src_val_f1"],
                ["best_src_val_f1"],
                ["best_src_val_metrics", "f1"],
                ["src_val", "f1"],
            ],
        ),
        "tgt_auc": first_number(
            data,
            [
                ["tgt_test_auc_at_best"],
                ["tgt_test_auc_at_best_src_val"],
                ["target_auc_at_best"],
                ["tgt_test_metrics_at_best_src_val", "auc"],
                ["target_test", "auc"],
                ["tgt_test", "auc"],
            ],
        ),
        "tgt_acc": first_number(
            data,
            [
                ["tgt_test_acc_at_best"],
                ["tgt_test_acc_at_best_src_val"],
                ["tgt_test_metrics_at_best_src_val", "acc"],
                ["target_test", "acc"],
                ["tgt_test", "acc"],
            ],
        ),
        "tgt_prec": first_number(
            data,
            [
                ["tgt_test_prec_at_best"],
                ["tgt_test_precision_at_best"],
                ["tgt_test_metrics_at_best_src_val", "prec"],
                ["tgt_test_metrics_at_best_src_val", "precision"],
                ["target_test", "prec"],
                ["tgt_test", "prec"],
            ],
        ),
        "tgt_rec": first_number(
            data,
            [
                ["tgt_test_rec_at_best"],
                ["tgt_test_recall_at_best"],
                ["tgt_test_sens_at_best"],
                ["tgt_test_metrics_at_best_src_val", "rec"],
                ["tgt_test_metrics_at_best_src_val", "recall"],
                ["target_test", "rec"],
                ["tgt_test", "rec"],
            ],
        ),
        "tgt_spec": first_number(
            data,
            [
                ["tgt_test_spec_at_best"],
                ["tgt_test_spec_at_best_src_val"],
                ["tgt_test_metrics_at_best_src_val", "spec"],
                ["target_test", "spec"],
                ["tgt_test", "spec"],
            ],
        ),
        "tgt_f1": first_number(
            data,
            [
                ["tgt_test_f1_at_best"],
                ["tgt_test_f1_weighted_at_best_src_val"],
                ["tgt_test_metrics_at_best_src_val", "f1"],
                ["tgt_test_metrics_at_best_src_val", "f1_weighted"],
                ["target_test", "f1"],
                ["tgt_test", "f1"],
            ],
        ),
        "best_epoch": first_number(data, [["best_epoch"], ["best_round"], ["epoch"]]),
    }


def iter_metric_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        names = set(filenames)
        for metric_name in METRIC_FILES:
            if metric_name in names:
                yield os.path.join(dirpath, metric_name)
                break


def infer_seed(exp_dir: str, args_data: Optional[Dict]) -> str:
    if args_data and "seed" in args_data:
        return str(args_data["seed"])
    for part in reversed(os.path.normpath(exp_dir).split(os.sep)):
        m = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", part, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        if part.isdigit():
            return part
    return ""


def infer_method(exp_dir: str, root: str, args_data: Optional[Dict]) -> str:
    if args_data:
        for key in ("baseline", "method_name"):
            if args_data.get(key):
                return str(args_data[key])

    rel = os.path.relpath(exp_dir, root)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    clean_parts = [p for p in parts if not re.match(r"seed[_-]?\d+$", p, flags=re.IGNORECASE)]

    if clean_parts:
        first = clean_parts[0]
        if first.lower().startswith("baseline_"):
            return first[len("baseline_") :]
        return re.sub(r"[_-]?center$", "", first, flags=re.IGNORECASE)

    base = os.path.basename(os.path.normpath(exp_dir))
    return re.sub(r"^baseline[_-]", "", base, flags=re.IGNORECASE)


def collect_records(roots: List[str], direction: str, average: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for root in roots:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            print(f"[WARN] missing root: {root_abs}")
            continue

        for metric_path in iter_metric_files(root_abs):
            exp_dir = os.path.dirname(metric_path)
            data = load_json(metric_path)
            if not data:
                print(f"[WARN] failed to parse: {metric_path}")
                continue

            args_path = os.path.join(exp_dir, "args.json")
            args_data = load_json(args_path) if os.path.isfile(args_path) else None
            metrics = extract_metrics(data)
            if average != "from_json":
                tgt_recomputed = recompute_classification_metrics(exp_dir, "tgt_test", average=average)
                if tgt_recomputed:
                    metrics.update(
                        {
                            "tgt_acc": tgt_recomputed["acc"],
                            "tgt_prec": tgt_recomputed["prec"],
                            "tgt_rec": tgt_recomputed["rec"],
                            "tgt_spec": tgt_recomputed["spec"],
                            "tgt_f1": tgt_recomputed["f1"],
                        }
                    )
                src_recomputed = recompute_classification_metrics(exp_dir, "src_val", average=average)
                if src_recomputed:
                    metrics.update(
                        {
                            "src_val_acc": src_recomputed["acc"],
                            "src_val_f1": src_recomputed["f1"],
                        }
                    )
            rec: Dict[str, object] = {
                "direction": direction,
                "method": infer_method(exp_dir, root_abs, args_data),
                "seed": infer_seed(exp_dir, args_data),
                "exp_dir": exp_dir,
                "metric_file": metric_path,
                "selection_metric": data.get("selection_metric", "src_val_auc"),
                "average": average,
            }
            rec.update(metrics)
            records.append(rec)
    return records


def write_csv(records: List[Dict[str, object]], path: str, fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def summarize(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["method"])].append(rec)

    summary: List[Dict[str, object]] = []
    for method, rows in grouped.items():
        out: Dict[str, object] = {
            "method": method,
            "n": len(rows),
            "seeds": ",".join(sorted({str(r.get("seed", "")) for r in rows if str(r.get("seed", ""))})),
        }
        for metric in REPORT_METRICS:
            vals = [safe_float(r.get(metric)) for r in rows]
            m, s = mean_std(vals)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        summary.append(out)

    summary.sort(
        key=lambda r: (
            safe_float(r.get("tgt_auc_mean")),
            safe_float(r.get("tgt_f1_mean")),
        ),
        reverse=True,
    )
    return summary


def write_markdown(summary: List[Dict[str, object]], path: str, digits: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = [
        "Method",
        "N",
        "Target AUC",
        "ACC",
        "Precision",
        "Recall/Sens",
        "Specificity",
        "F1",
        "Src Val AUC",
        "Best Epoch",
    ]
    metric_cols = [
        "tgt_auc",
        "tgt_acc",
        "tgt_prec",
        "tgt_rec",
        "tgt_spec",
        "tgt_f1",
        "src_val_auc",
        "best_epoch",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for rec in summary:
        row = [str(rec["method"]), str(rec["n"])]
        for col in metric_cols:
            row.append(fmt_mean_std(safe_float(rec.get(f"{col}_mean")), safe_float(rec.get(f"{col}_std")), digits))
        lines.append("| " + " | ".join(row) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser("Collect multi-seed baseline results")
    ap.add_argument("--root", action="append", required=True, help="Result root to scan. Can be repeated.")
    ap.add_argument("--direction", default="", help="Direction label, e.g. MRNet_to_KneeMRI")
    ap.add_argument("--out_dir", required=True, help="Directory for per_seed.csv, summary.csv, summary.md")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument(
        "--average",
        choices=["from_json", "binary", "macro", "weighted"],
        default="from_json",
        help=(
            "PRF averaging. from_json preserves saved metrics. binary/macro/weighted "
            "recompute ACC/Precision/Recall/F1/Specificity from pred_*_best.csv."
        ),
    )
    args = ap.parse_args()

    records = collect_records(args.root, args.direction, average=args.average)
    os.makedirs(args.out_dir, exist_ok=True)

    per_seed_csv = os.path.join(args.out_dir, "per_seed.csv")
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")

    per_seed_fields = [
        "direction",
        "method",
        "seed",
        "average",
        "selection_metric",
        "src_val_auc",
        "src_val_acc",
        "src_val_f1",
        "tgt_auc",
        "tgt_acc",
        "tgt_prec",
        "tgt_rec",
        "tgt_spec",
        "tgt_f1",
        "best_epoch",
        "exp_dir",
        "metric_file",
    ]
    write_csv(records, per_seed_csv, per_seed_fields)

    summary = summarize(records)
    summary_fields = ["method", "n", "seeds"]
    for metric in REPORT_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(summary, summary_csv, summary_fields)
    write_markdown(summary, summary_md, args.digits)

    print(f"[Done] records={len(records)} methods={len(summary)}")
    print(f"[Done] per-seed csv: {per_seed_csv}")
    print(f"[Done] summary csv: {summary_csv}")
    print(f"[Done] markdown: {summary_md}")
    for rec in summary:
        print(
            f"[Summary] {rec['method']}: "
            f"n={rec['n']} "
            f"tgt_auc={fmt_mean_std(safe_float(rec.get('tgt_auc_mean')), safe_float(rec.get('tgt_auc_std')), args.digits)} "
            f"tgt_f1={fmt_mean_std(safe_float(rec.get('tgt_f1_mean')), safe_float(rec.get('tgt_f1_std')), args.digits)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
