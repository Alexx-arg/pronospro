"""RunsStore — per-run config + per-fold metrics (Sprint 5.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.prediction.storage import layout as layout_mod


class RunsStore:
    """Persist run config, fold metrics, summary."""

    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = Path(base_path)

    def save_config(self, run_id: str, config: dict[str, Any], *, overwrite: bool = False) -> Path:
        path = layout_mod.run_config_path(self.base_path, run_id=run_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"run config already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def save_fold_metrics(
        self,
        run_id: str,
        fold_id: str,
        metrics: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        path = layout_mod.fold_metrics_path(self.base_path, run_id=run_id, fold_id=fold_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"fold metrics already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def save_fold_predictions(
        self,
        run_id: str,
        fold_id: str,
        rows: list[dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> Path:
        import csv

        path = layout_mod.fold_predictions_path(self.base_path, run_id=run_id, fold_id=fold_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"fold predictions already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def save_fold_calibrator(
        self,
        run_id: str,
        fold_id: str,
        calibrator_dict: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        path = layout_mod.fold_dir(self.base_path, run_id=run_id, fold_id=fold_id) / "calibrator.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"fold calibrator already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(calibrator_dict, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def save_summary(
        self,
        run_id: str,
        summary: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        path = layout_mod.run_summary_path(self.base_path, run_id=run_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"run summary already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_summary(self, run_id: str) -> dict[str, Any]:
        path = layout_mod.run_summary_path(self.base_path, run_id=run_id)
        if not path.is_file():
            raise FileNotFoundError(f"run summary not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


__all__ = ["RunsStore"]
