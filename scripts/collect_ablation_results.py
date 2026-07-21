#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, Optional


EXPECTED_SETTINGS = [
    "dynamic_tau",
    "fixed_tau_070",
    "fixed_tau_075",
    "fixed_tau_080",
    "fixed_tau_085",
    "fixed_tau_090",
]

BEST_LINE_RE = re.compile(
    r"\[Best@src_val\]\s+round=(\d+)\s+src_val_auc=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+tgt_test_auc=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def _safe_float(v):
    try:
        x = float(v)
        if math.isnan(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _extract_from_summary(exp_dir: str) -> Optional[Dict]:
    for name in ("metrics_summary.json", "best_by_src_val.json"):
        data = _load_json(os.path.join(exp_dir, name))
        if data is None:
            continue
        best_round = data.get("best_round", "")
        src_auc = _safe_float((data.get("best_src_val_metrics") or {}).get("auc"))
        tgt_auc = _safe_float((data.get("tgt_test_metrics_at_best_src_val") or {}).get("auc"))
        return {
            "best_round": best_round,
            "best_src_val_auc": src_auc,
            "tgt_test_auc_at_best_src_val": tgt_auc,
        }
    return None


def _extract_from_log(exp_dir: str) -> Optional[Dict]:
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.isfile(log_path):
        return None

    best = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BEST_LINE_RE.search(line)
            if not m:
                continue
            best = {
                "best_round": int(m.group(1)),
                "best_src_val_auc": _safe_float(m.group(2)),
                "tgt_test_auc_at_best_src_val": _safe_float(m.group(3)),
            }
    return best


def collect_one(root_dir: str, setting: str) -> Dict:
    exp_dir = os.path.join(root_dir, setting)
    rec = {
        "setting": setting,
        "best_round": "",
        "best_src_val_auc": float("nan"),
        "tgt_test_auc_at_best_src_val": float("nan"),
        "save_dir": exp_dir,
    }

    if not os.path.isdir(exp_dir):
        rec["error"] = "missing_dir"
        return rec

    parsed = _extract_from_summary(exp_dir)
    if parsed is None:
        parsed = _extract_from_log(exp_dir)

    if parsed is None:
        rec["error"] = "missing_summary_and_best_log"
        return rec

    rec.update(parsed)
    rec["error"] = ""
    return rec


def save_csv(records, out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = [
        "setting",
        "best_round",
        "best_src_val_auc",
        "tgt_test_auc_at_best_src_val",
        "save_dir",
        "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser("Collect ablation results by best source-val round")
    ap.add_argument("--root_dir", required=True, help="Root dir containing ablation subdirs")
    ap.add_argument("--out_csv", default="", help="Output csv path")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_csv = args.out_csv or os.path.join(root_dir, "ablation_results.csv")

    records = [collect_one(root_dir, s) for s in EXPECTED_SETTINGS]
    save_csv(records, out_csv)

    print(f"[Done] saved csv: {out_csv}")
    for r in records:
        print(
            f"[Result] {r['setting']}: "
            f"best_round={r['best_round']} "
            f"best_src_val_auc={r['best_src_val_auc']} "
            f"tgt_test_auc_at_best_src_val={r['tgt_test_auc_at_best_src_val']} "
            f"error={r.get('error', '')}"
        )


if __name__ == "__main__":
    main()
