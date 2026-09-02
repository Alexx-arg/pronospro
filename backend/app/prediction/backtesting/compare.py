"""Comparison of runs — Sprint 5.10.

Pure aggregation over persisted summaries. Never trains, never reads raw
train/val/test data — only ``summary.json`` + ``config.json`` via
``RunsStore``. CSV only (no pyarrow/parquet).

CSV columns (D1):
  model_name, model_version, dataset_version, iterator_params,
  n_folds, mean_accuracy, std_accuracy, mean_log_loss, std_log_loss,
  mean_ece, mean_mce, total_n_predictions, weighted_log_loss, weighted_ece
Ponderadas (D2): weighted = Σ n_f·metric / Σ n_f

Folds vacíos (D4): n_folds=0 → métricas agregadas NaN, total 0, sin
contaminar ponderadas.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.prediction.storage import layout as layout_mod

COLUMNS: list[str] = [
    "model_name",
    "model_version",
    "dataset_version",
    "iterator_params",
    "n_folds",
    "mean_accuracy",
    "std_accuracy",
    "mean_log_loss",
    "std_log_loss",
    "mean_ece",
    "mean_mce",
    "total_n_predictions",
    "weighted_log_loss",
    "weighted_ece",
]


def _load_runs(
    base_path: Path, dataset_version: str, iterator_params: dict[str, Any] | None
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Load (run_id, config, summary) for runs matching filter.

    If iterator_params is None, only dataset_version is checked.
    Raises ValueError if none found.
    """
    runs_root = base_path / "data" / "models" / "runs"
    if not runs_root.is_dir():
        raise ValueError(f"runs root not found: {runs_root}")
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_id.startswith("comparison_"):
            continue
        config_path = layout_mod.run_config_path(base_path, run_id=run_id)
        summary_path = layout_mod.run_summary_path(base_path, run_id=run_id)
        if not config_path.is_file() or not summary_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config.get("dataset_version") != dataset_version:
            continue
        if iterator_params is not None:
            # Compare via JSON dump sorted for determinism
            cfg_params = config.get("iterator_params")
            if json.dumps(cfg_params, sort_keys=True) != json.dumps(iterator_params, sort_keys=True):
                continue
        out.append((run_id, config, summary))
    if not out:
        raise ValueError(
            f"no runs found for dataset_version={dataset_version!r} iterator_params={iterator_params!r}"
        )
    return out


def _validate_same_dataset_and_params(
    configs: list[dict[str, Any]],
) -> None:
    """Ensure all configs share dataset_version and iterator_params, else fail."""
    if not configs:
        return
    first_ds = configs[0].get("dataset_version")
    first_params = json.dumps(configs[0].get("iterator_params"), sort_keys=True)
    for cfg in configs[1:]:
        if cfg.get("dataset_version") != first_ds:
            raise ValueError(
                f"dataset_version mismatch: {first_ds!r} vs {cfg.get('dataset_version')!r}"
            )
        if json.dumps(cfg.get("iterator_params"), sort_keys=True) != first_params:
            raise ValueError(
                f"iterator_params mismatch: {first_params!r} vs {json.dumps(cfg.get('iterator_params'), sort_keys=True)!r}"
            )


def compare_runs(
    base_path: Path | str = Path("data"),
    *,
    dataset_version: str | None = None,
    iterator_params: dict[str, Any] | None = None,
    run_ids: list[str] | None = None,
    date_tag: str | None = None,
) -> Path:
    """Aggregate runs and write comparison CSV.

    Provide either (dataset_version + iterator_params) as filter, or explicit run_ids.
    If run_ids given, dataset_version/iterator_params are inferred and validated.

    Returns path to written CSV.
    """
    base_path = Path(base_path)
    # Resolve runs to compare
    runs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if run_ids is not None:
        for rid in run_ids:
            cfg_path = layout_mod.run_config_path(base_path, run_id=rid)
            sum_path = layout_mod.run_summary_path(base_path, run_id=rid)
            if not cfg_path.is_file() or not sum_path.is_file():
                raise ValueError(f"run not found: {rid}")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            summ = json.loads(sum_path.read_text(encoding="utf-8"))
            runs.append((rid, cfg, summ))
        # Validate same dataset/params
        _validate_same_dataset_and_params([c for _, c, _ in runs])
        # Infer dataset_version if not provided
        if dataset_version is None:
            dataset_version = runs[0][1].get("dataset_version", "unknown")
    else:
        if dataset_version is None:
            raise ValueError("must provide dataset_version or run_ids")
        runs = _load_runs(base_path, dataset_version, iterator_params)

    # Validate again (covers run_ids path)
    _validate_same_dataset_and_params([c for _, c, _ in runs])

    # Determine iterator_params for CSV (first run)
    first_params = runs[0][1].get("iterator_params", iterator_params or {})
    params_str = json.dumps(first_params, sort_keys=True)
    ds_version = runs[0][1].get("dataset_version", dataset_version or "")

    # Build rows
    rows: list[dict[str, Any]] = []
    for _run_id, config, summary in sorted(runs, key=lambda x: x[0]):
        folds = summary.get("folds", [])
        n_folds = int(summary.get("n_folds", len(folds)))
        # Fallback if summary n_folds mismatched
        if n_folds != len(folds):
            n_folds = len(folds)
        model_name = str(config.get("model_name", summary.get("model_name", "")))
        model_version = str(config.get("model_version", summary.get("model_version", "")))

        if n_folds == 0:
            row = {
                "model_name": model_name,
                "model_version": model_version,
                "dataset_version": ds_version,
                "iterator_params": params_str,
                "n_folds": 0,
                "mean_accuracy": float("nan"),
                "std_accuracy": float("nan"),
                "mean_log_loss": float("nan"),
                "std_log_loss": float("nan"),
                "mean_ece": float("nan"),
                "mean_mce": float("nan"),
                "total_n_predictions": 0,
                "weighted_log_loss": float("nan"),
                "weighted_ece": float("nan"),
            }
            rows.append(row)
            continue

        # Extract per-fold metrics
        accs: list[float] = []
        lls: list[float] = []
        eces: list[float] = []
        mces: list[float] = []
        ns: list[int] = []
        for f in folds:
            accs.append(float(f.get("accuracy", float("nan"))))
            lls.append(float(f.get("log_loss", float("nan"))))
            eces.append(float(f.get("ece", float("nan"))))
            mces.append(float(f.get("mce", float("nan"))))
            ns.append(int(f.get("n_predictions", 0)))

        total_n = int(sum(ns))
        # Simple means
        mean_acc = float(np.nanmean(accs)) if accs else float("nan")
        std_acc = float(np.nanstd(accs, ddof=0)) if len(accs) > 1 else 0.0 if len(accs) == 1 else float("nan")
        mean_ll = float(np.nanmean(lls)) if lls else float("nan")
        std_ll = float(np.nanstd(lls, ddof=0)) if len(lls) > 1 else 0.0 if len(lls) == 1 else float("nan")
        mean_ece = float(np.nanmean(eces)) if eces else float("nan")
        mean_mce = float(np.nanmean(mces)) if mces else float("nan")
        # Weighted
        if total_n > 0:
            w_ll = float(np.nansum([n * ll for n, ll in zip(ns, lls, strict=False)]) / total_n)
            w_ece = float(np.nansum([n * e for n, e in zip(ns, eces, strict=False)]) / total_n)
        else:
            w_ll = float("nan")
            w_ece = float("nan")

        row = {
            "model_name": model_name,
            "model_version": model_version,
            "dataset_version": ds_version,
            "iterator_params": params_str,
            "n_folds": int(n_folds),
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "mean_log_loss": float(mean_ll),
            "std_log_loss": float(std_ll),
            "mean_ece": float(mean_ece),
            "mean_mce": float(mean_mce),
            "total_n_predictions": int(total_n),
            "weighted_log_loss": float(w_ll),
            "weighted_ece": float(w_ece),
        }
        rows.append(row)

    # Sort rows deterministically by model_name
    rows.sort(key=lambda r: (r["model_name"], r["model_version"]))

    # Write CSV
    if date_tag is None:
        date_tag = datetime.now().strftime("%Y%m%d")
    out_path = layout_mod.comparison_csv_path(base_path, date_tag=date_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            # Convert NaN to string "nan" for CSV (csv writer will handle)
            writer.writerow(r)
    return out_path


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Compare runs (Sprint 5.10)")
    parser.add_argument("--base-path", type=str, default="data", help="Base path containing data/models/runs")
    parser.add_argument("--dataset-version", type=str, required=False, default=None)
    parser.add_argument("--iterator-params", type=str, required=False, default=None, help="JSON string for iterator_params filter")
    parser.add_argument("--run-ids", type=str, nargs="*", default=None, help="Explicit run_ids to compare")
    parser.add_argument("--date-tag", type=str, required=False, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    iterator_params = None
    if args.iterator_params:
        iterator_params = json.loads(args.iterator_params)
    out = compare_runs(
        base_path=Path(args.base_path),
        dataset_version=args.dataset_version,
        iterator_params=iterator_params,
        run_ids=args.run_ids,
        date_tag=args.date_tag,
    )
    print(f"comparison written to {out}")


if __name__ == "__main__":
    main()


__all__ = ["COLUMNS", "compare_runs"]
