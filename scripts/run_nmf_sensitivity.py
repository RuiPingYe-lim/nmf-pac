#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple


PRESETS: Dict[str, Dict[str, List[str]]] = {
    "quick": {
        "K": ["1", "2", "3"],
        "beta_loss": ["frobenius", "kullback-leibler"],
        "nmf_assign_iters": ["40", "60", "100"],
        "alphaH": ["0", "1e-4", "1e-3"],
    },
    # Exact grid used for Figure 7 in the paper (four axes: K, loss type, T, m).
    "paper": {
        "K": ["1", "2", "3", "4"],
        "beta_loss": ["frobenius", "kullback-leibler"],
        "nmf_assign_iters": ["20", "40", "60", "100", "150"],
        "proto_m": ["0.90", "0.95", "0.97", "0.99"],
    },
}

METRIC_PATTERNS = {
    "val_src_auc": re.compile(r"\[Val\]\s+src\s+AUC=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "best_src_val_auc": re.compile(r"best_src_val_auc[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
    "tgt_test_auc": re.compile(r"tgt(?:_test)?_auc(?:_at_best_src_val)?[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
}


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def split_values(raw: str) -> List[str]:
    values = [x.strip() for x in raw.split(",")]
    return [x for x in values if x]


def parse_sweep(items: Sequence[str]) -> Dict[str, List[str]]:
    sweep: Dict[str, List[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --sweep item {item!r}; expected name=v1,v2,...")
        name, raw_values = item.split("=", 1)
        name = name.strip().lstrip("-").replace("-", "_")
        values = split_values(raw_values)
        if not name or not values:
            raise ValueError(f"Bad --sweep item {item!r}; empty name or values")
        sweep[name] = values
    return sweep


def strip_save_dir(args: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--save_dir":
            i += 2
            continue
        if token.startswith("--save_dir="):
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def load_json(path: str):
    if not os.path.isfile(path):
        return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def as_float(value):
    try:
        x = float(value)
        return x if not math.isnan(x) else ""
    except Exception:
        return ""


def extract_metrics_from_json(save_dir: str) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for name in ("metrics_summary.json", "best_by_src_val.json"):
        data = load_json(os.path.join(save_dir, name))
        if not isinstance(data, dict):
            continue
        metrics["best_round"] = data.get("best_round", "")
        src = data.get("best_src_val_metrics") or {}
        tgt = data.get("tgt_test_metrics_at_best_src_val") or {}
        metrics["src_val_auc"] = as_float(src.get("auc"))
        metrics["src_val_acc"] = as_float(src.get("acc"))
        metrics["tgt_test_auc"] = as_float(tgt.get("auc"))
        metrics["tgt_test_acc"] = as_float(tgt.get("acc"))
        metrics["tgt_test_f1"] = as_float(tgt.get("f1"))
        metrics["tgt_test_spec"] = as_float(tgt.get("spec"))
        return metrics
    return metrics


def extract_metrics_from_log(log_path: str) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    if not os.path.isfile(log_path):
        return metrics
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for key, pattern in METRIC_PATTERNS.items():
                    m = pattern.search(line)
                    if m:
                        metrics[key] = as_float(m.group(1))
    except Exception:
        return metrics
    return metrics


def experiment_done(save_dir: str) -> bool:
    if os.path.isfile(os.path.join(save_dir, "DONE")):
        return True
    if os.path.isfile(os.path.join(save_dir, "metrics_summary.json")):
        return True
    if os.path.isfile(os.path.join(save_dir, "best_by_src_val.json")):
        return True
    return False


def iter_experiments(sweep: Dict[str, List[str]]) -> Iterable[Tuple[str, str]]:
    for param, values in sweep.items():
        for value in values:
            yield param, value


def write_manifest(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_command(cmd: Sequence[str], log_path: str, dry_run: bool) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    printable = " ".join(subprocess.list2cmdline([x]) for x in cmd)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[Command] {printable}\n")
        log.flush()
        if dry_run:
            print(printable)
            log.write("[DryRun] skipped execution\n")
            return 0

        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def save_summary_csv(records: List[Dict[str, object]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = [
        "param",
        "value",
        "status",
        "exit_code",
        "best_round",
        "src_val_auc",
        "src_val_acc",
        "tgt_test_auc",
        "tgt_test_acc",
        "tgt_test_f1",
        "tgt_test_spec",
        "val_src_auc",
        "save_dir",
        "log_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(
        "Run NMF parameter sensitivity experiments. Put train_online_end2end.py args after --."
    )
    ap.add_argument("--entry", default="train_online_end2end.py", help="Training entry script")
    ap.add_argument("--root_dir", required=True, help="Output root for all sensitivity runs")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="paper")
    ap.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Override/add a sweep, e.g. --sweep K=1,2,3 --sweep alphaH=0,1e-4,1e-3",
    )
    ap.add_argument("--only", default="", help="Comma-separated parameter names to run from the selected sweep")
    ap.add_argument("--skip_existing", action="store_true", default=True, help="Skip runs with DONE or summary json")
    ap.add_argument("--rerun", action="store_true", help="Rerun even if outputs already exist")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--summary_csv", default="", help="Default: <root_dir>/sensitivity_summary.csv")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed to the training entry after --")
    args = ap.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    train_args = strip_save_dir(train_args)

    sweep = {k: list(v) for k, v in PRESETS[args.preset].items()}
    sweep.update(parse_sweep(args.sweep))
    if args.only:
        wanted = {x.strip().lstrip("-").replace("-", "_") for x in args.only.split(",") if x.strip()}
        sweep = {k: v for k, v in sweep.items() if k in wanted}
        if not sweep:
            raise SystemExit(f"No sweep parameters matched --only={args.only!r}")

    root_dir = os.path.abspath(args.root_dir)
    entry = os.path.abspath(args.entry)
    summary_csv = args.summary_csv or os.path.join(root_dir, "sensitivity_summary.csv")
    os.makedirs(root_dir, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entry": entry,
        "root_dir": root_dir,
        "preset": args.preset,
        "sweep": sweep,
        "base_train_args": train_args,
    }
    write_manifest(os.path.join(root_dir, "sensitivity_manifest.json"), manifest)

    records: List[Dict[str, object]] = []
    total = sum(len(v) for v in sweep.values())
    print(f"[Plan] experiments={total} root={root_dir}")

    for idx, (param, value) in enumerate(iter_experiments(sweep), start=1):
        exp_name = f"{param}_{safe_name(value)}"
        save_dir = os.path.join(root_dir, param, exp_name)
        log_path = os.path.join(save_dir, "train.log")
        rec: Dict[str, object] = {
            "param": param,
            "value": value,
            "save_dir": save_dir,
            "log_path": log_path,
            "exit_code": "",
            "status": "",
        }

        if args.skip_existing and not args.rerun and experiment_done(save_dir):
            print(f"[Skip {idx}/{total}] {exp_name}")
            rec["status"] = "skipped_existing"
            rec.update(extract_metrics_from_json(save_dir))
            rec.update(extract_metrics_from_log(log_path))
            records.append(rec)
            save_summary_csv(records, summary_csv)
            continue

        os.makedirs(save_dir, exist_ok=True)
        write_manifest(
            os.path.join(save_dir, "sensitivity_case.json"),
            {"param": param, "value": value, "save_dir": save_dir},
        )

        cmd = [
            sys.executable,
            entry,
            *train_args,
            f"--{param}",
            str(value),
        ]
        if param == "K":
            cmd.extend(["--Kmax", str(value)])
        cmd.extend(["--save_dir", save_dir])

        print(f"[Run {idx}/{total}] {param}={value}")
        exit_code = run_command(cmd, log_path=log_path, dry_run=args.dry_run)
        rec["exit_code"] = exit_code
        rec["status"] = "ok" if exit_code == 0 else "failed"

        if exit_code == 0 and not args.dry_run:
            with open(os.path.join(save_dir, "DONE"), "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat(timespec="seconds") + "\n")

        rec.update(extract_metrics_from_json(save_dir))
        rec.update(extract_metrics_from_log(log_path))
        records.append(rec)
        save_summary_csv(records, summary_csv)

        if exit_code != 0 and not args.continue_on_error:
            print(f"[Stop] failed at {param}={value}, exit_code={exit_code}")
            return int(exit_code)

    print(f"[Done] summary={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
