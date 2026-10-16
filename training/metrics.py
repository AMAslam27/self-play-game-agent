"""Append raw training measurements and recover CSVs to a checkpoint boundary."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

SCHEMAS = {
    "decisions": [
        "session",
        "episode",
        "decision",
        "learning_updates",
        "epsilon",
        "learning_rate",
    ],
    "updates": [
        "session",
        "episode",
        "decision",
        "update",
        "loss",
        "lr_used",
        "lr_next",
    ],
    "episodes": [
        "session",
        "episode",
        "seat",
        "outcome",
        "return",
        "decisions",
        "updates",
        "total_decisions",
        "total_updates",
        "duration_seconds",
        "training_seconds",
    ],
    "evaluation": [
        "session",
        "episode",
        "decision",
        "update",
        "opponent",
        "seat",
        "games",
        "wins",
        "draws",
        "losses",
        "average_return",
        "duration_seconds",
        "status",
        "db_run_id",
    ],
}

OPPONENT_FIELDS = ["session", "episode", "category", "opponent_id"]


def _active_schemas(*, include_opponents: bool) -> dict[str, list[str]]:
    schemas = dict(SCHEMAS)
    if include_opponents:
        schemas["opponents"] = OPPONENT_FIELDS
    return schemas


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace metadata atomically; leave the previous file intact on failure."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_metrics(run_dir: Path, table: str) -> list[dict[str, str]]:
    path = run_dir / f"{table}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def restore_metric_offsets(
    run_dir: Path,
    offsets: Mapping[str, int],
    *,
    include_opponents: bool = False,
) -> None:
    """Archive trailing history before truncating to the saved checkpoint.

    This prevents duplicate counters when resuming an older checkpoint or
    recovering from a crash after metrics were flushed but before saving.
    """
    schemas = _active_schemas(include_opponents=include_opponents)
    if set(offsets) != set(schemas):
        raise ValueError(
            "Checkpoint metric offsets do not match the active CSV schemas"
        )
    changed: list[tuple[Path, int]] = []
    for table, fields in schemas.items():
        offset = offsets[table]
        path = run_dir / f"{table}.csv"
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or not path.is_file()
            or path.stat().st_size < offset
        ):
            raise ValueError(f"Metrics cannot be restored to checkpoint: {table}")
        with path.open(encoding="utf-8", newline="") as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError(f"Unexpected CSV schema: {table}")
        if path.stat().st_size > offset:
            changed.append((path, offset))
    if not changed:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run_dir / "recoveries" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path, _ in changed:
        shutil.copy2(path, archive / path.name)
    for path, offset in changed:
        with path.open("r+b") as stream:
            stream.truncate(offset)


class MetricsRecorder:
    """Raw CSV writer; the coordinator commits records after complete episodes."""

    def __init__(self, run_dir: Path, *, include_opponents: bool = False):
        self.run_dir = run_dir
        self._schemas = _active_schemas(include_opponents=include_opponents)
        self._files: dict[str, TextIO] = {}
        self._writers: dict[str, csv.DictWriter[str]] = {}
        try:
            # Validate all existing headers before opening any file for writing.
            for table, fields in self._schemas.items():
                path = run_dir / f"{table}.csv"
                existing = path.exists() and path.stat().st_size > 0
                if existing:
                    with path.open(encoding="utf-8", newline="") as check:
                        if next(csv.reader(check), None) != fields:
                            raise ValueError(f"Unexpected CSV schema: {table}")
            for table, fields in self._schemas.items():
                path = run_dir / f"{table}.csv"
                existing = path.exists() and path.stat().st_size > 0
                stream = path.open("a", encoding="utf-8", newline="")
                self._files[table] = stream
                writer = csv.DictWriter(stream, fieldnames=fields)
                self._writers[table] = writer
                if not existing:
                    writer.writeheader()
        except BaseException:
            self.close()
            raise

    def write(self, table: str, row: Mapping[str, Any]) -> None:
        if table not in self._writers:
            raise ValueError(f"Metrics table is not enabled: {table}")
        self._writers[table].writerow(row)

    def flush(self) -> None:
        for stream in self._files.values():
            stream.flush()

    def offsets(self) -> dict[str, int]:
        self.flush()
        return {
            table: (self.run_dir / f"{table}.csv").stat().st_size
            for table in self._schemas
        }

    def close(self) -> None:
        for stream in self._files.values():
            stream.close()

    def __enter__(self) -> MetricsRecorder:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
