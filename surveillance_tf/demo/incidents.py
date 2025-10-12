"""SQLite persistence for incidents."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    severity TEXT NOT NULL,
    class_name TEXT NOT NULL,
    mean_conf REAL NOT NULL,
    ci_low REAL NOT NULL,
    ci_high REAL NOT NULL,
    video_path TEXT NOT NULL,
    clip_path TEXT,
    report_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class IncidentRecord:
    start_ts: float
    end_ts: float
    severity: str
    class_name: str
    mean_conf: float
    ci_low: float
    ci_high: float
    video_path: str
    clip_path: Optional[str] = None
    report_path: Optional[str] = None
    id: Optional[int] = None


class IncidentDAO:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SCHEMA)
        self.conn.commit()
        LOGGER.info("Incident DB initialised at %s", self.db_path)

    def log(self, record: IncidentRecord) -> int:
        query = (
            "INSERT INTO incidents (start_ts, end_ts, severity, class_name, mean_conf, ci_low, ci_high, video_path, clip_path, report_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        cur = self.conn.execute(
            query,
            (
                record.start_ts,
                record.end_ts,
                record.severity,
                record.class_name,
                record.mean_conf,
                record.ci_low,
                record.ci_high,
                record.video_path,
                record.clip_path,
                record.report_path,
            ),
        )
        self.conn.commit()
        inserted_id = cur.lastrowid
        LOGGER.debug("Inserted incident id=%d", inserted_id)
        return inserted_id

    def update_paths(self, incident_id: int, clip_path: Optional[str], report_path: Optional[str]) -> None:
        self.conn.execute(
            "UPDATE incidents SET clip_path=?, report_path=? WHERE id=?",
            (clip_path, report_path, incident_id),
        )
        self.conn.commit()

    def recent(self, limit: int = 20) -> List[IncidentRecord]:
        cur = self.conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        records: List[IncidentRecord] = []
        for row in rows:
            records.append(
                IncidentRecord(
                    id=row["id"],
                    start_ts=row["start_ts"],
                    end_ts=row["end_ts"],
                    severity=row["severity"],
                    class_name=row["class_name"],
                    mean_conf=row["mean_conf"],
                    ci_low=row["ci_low"],
                    ci_high=row["ci_high"],
                    video_path=row["video_path"],
                    clip_path=row["clip_path"],
                    report_path=row["report_path"],
                )
            )
        return records

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "IncidentDAO":  # pragma: no cover - context manager convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()


__all__ = ["IncidentDAO", "IncidentRecord"]
