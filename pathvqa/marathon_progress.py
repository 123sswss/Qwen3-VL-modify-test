"""Lightweight scheduling and progress logging for marathon validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


MARATHON_PROGRESS_COLUMNS = (
    "timestamp",
    "epoch",
    "global_step",
    "overall",
    "yes_no",
    "free_form",
    "latest_train_loss",
    "elapsed_hours",
    "evaluation_minutes",
    "checkpoint",
    "status",
    "error",
)


def scheduled_validation_epochs(
    total_epochs: int,
    start_epoch: int,
) -> tuple[int, ...]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if start_epoch < 1 or start_epoch > total_epochs:
        raise ValueError("start_epoch must be within the training schedule")
    return tuple(range(start_epoch, total_epochs + 1))


class MarathonProgressLog:
    """Append one flushed, machine-readable summary row per validation epoch."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recorded_epochs: set[int] = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row.get("epoch"):
                        self._recorded_epochs.add(int(row["epoch"]))
        else:
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=MARATHON_PROGRESS_COLUMNS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                handle.flush()

    def append(self, row: Mapping[str, Any]) -> None:
        epoch = int(row["epoch"])
        if epoch in self._recorded_epochs:
            raise RuntimeError(f"Marathon epoch {epoch} is already recorded")
        normalized = {
            key: row.get(key, "")
            for key in MARATHON_PROGRESS_COLUMNS
        }
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=MARATHON_PROGRESS_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writerow(normalized)
            handle.flush()
        self._recorded_epochs.add(epoch)
