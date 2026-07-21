#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _get_metric(d: Dict, path: List[str]) -> float:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    return _safe_float(cur)


def _parse_param_from_dirname(dirname: str, exp_prefix: str) -> Optional[float]:
    if not dirname.startswith(exp_prefix):
        return None
    raw = dirname[len(exp_prefix):]
    try:
        return float(raw)
    except Exception:
        return None


def _load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None


def collect_records(root_dir: str, exp_prefix: str):
    records = []
    skipped = []
    for name in sorted(os.listdir(root_dir)):
        exp_dir = os.path.join(root_dir, name)
        if not os.path.isdir(exp_dir) or not name.startswith(exp_prefix):
            continue

        param_value = _parse_param_from_dirname(name, exp_prefix)
        if param_value is None:
            skipped.append((name, "cannot parse param value from dirname"))
            continue

        summary_json = os.path.join(exp_dir, "metrics_summary.json")
        if not os.path.isfile(summary_json):
            legacy_json = os.path.join(exp_dir, "best_by_src_val.json")
            if os.path.isfile(legacy_json):
                summary_json = legacy_json
            else:
                skipped.append((name, "missing metrics_summary.json"))
                continue

        data = _load_json(summary_json)
        if data is None:
            skipped.append((name, f"failed to load json: {summary_json}"))
            continue

        best_round = data.get("best_round", None)
        record = {
            "exp_dir": exp_dir,
            "param_value": float(param_value),
            "best_round": best_round if best_round is not None else "",
            "src_val_auc": _get_metric(data, ["best_src_val_metrics", "auc"]),
            "src_val_acc": _get_metric(data, ["best_src_val_metrics", "acc"]),
            "tgt_test_auc": _get_metric(data, ["tgt_test_metrics_at_best_src_val", "auc"]),
            "tgt_test_acc": _get_metric(data, ["tgt_test_metrics_at_best_src_val", "acc"]),
            "tgt_test_f1": _get_metric(data, ["tgt_test_metrics_at_best_src_val", "f1"]),
            "tgt_test_spec": _get_metric(data, ["tgt_test_metrics_at_best_src_val", "spec"]),
        }
        records.append(record)
    records.sort(key=lambda x: x["param_value"])
    return records, skipped


def save_csv(records: List[Dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = [
        "param_value",
        "best_round",
        "src_val_auc",
        "src_val_acc",
        "tgt_test_auc",
        "tgt_test_acc",
        "tgt_test_f1",
        "tgt_test_spec",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def save_plot(records: List[Dict], param_name: str, out_png: str):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    x = [r["param_value"] for r in records]
    y_tgt = [r["tgt_test_auc"] for r in records]

    plt.figure(figsize=(7.2, 4.8))

    # 用“虚线 + 散点”代替粗实线，视觉上会柔和很多
    plt.plot(x, y_tgt, linestyle="--", linewidth=1.2, alpha=0.7)
    plt.scatter(x, y_tgt, marker="s", s=42)

    # 标注最优点
    # 用竖直虚线标出最优参数位置
    best_idx = max(range(len(y_tgt)), key=lambda i: y_tgt[i])
    best_x = x[best_idx]
    best_y = y_tgt[best_idx]

    plt.vlines(best_x, ymin=0.5, ymax=best_y,
           linestyle="--", linewidth=1.2, color="orange", alpha=0.6)
    

    plt.title(r"(a)The AUC versus parameter $\lambda$ on MRNet to KneeMRI")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("AUC")

    # AUC 随机基线
    plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)

    # 纵轴从 0.5 起，更合理，也不会显得波动过分夸张
    plt.ylim(0.5, max(y_tgt) + 0.03)

    plt.grid(True, alpha=0.3, linestyle="--")

    # 只画一条线时，图例可以不要
    # plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Collect sensitivity results and plot AUC curve")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--exp_prefix", required=True)
    ap.add_argument("--param_name", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    records, skipped = collect_records(args.root_dir, args.exp_prefix)
    for name, reason in skipped:
        print(f"[Skip] {name}: {reason}")

    if len(records) == 0:
        raise SystemExit("No valid experiment records found.")

    save_csv(records, args.out_csv)
    save_plot(records, args.param_name, args.out_png)
    print(f"[Done] records={len(records)}")
    print(f"[Done] csv={args.out_csv}")
    print(f"[Done] png={args.out_png}")


if __name__ == "__main__":
    main()
