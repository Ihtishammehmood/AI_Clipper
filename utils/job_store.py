"""SQLite-backed job index used by the dashboard."""
from __future__ import annotations

import sqlite3
from contextlib import closing

from config import Settings
from core.models import JobState
from utils.file_manager import JobPaths


def _connect(st: Settings) -> sqlite3.Connection:
    st.output_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(st.output_dir / "dashboard.sqlite3", timeout=30)
    db.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            stage TEXT NOT NULL,
            clip_count INTEGER NOT NULL,
            approved_count INTEGER NOT NULL,
            state_json TEXT NOT NULL
        )"""
    )
    return db


def save_job(st: Settings, state: JobState) -> None:
    with closing(_connect(st)) as db:
        with db:
            db.execute(
                """INSERT INTO jobs (job_id, created_at, stage, clip_count, approved_count, state_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                   created_at=excluded.created_at, stage=excluded.stage,
                   clip_count=excluded.clip_count, approved_count=excluded.approved_count,
                   state_json=excluded.state_json""",
                (
                    state.job_id,
                    state.created_at,
                    state.stage,
                    len(state.clips),
                    sum(clip.status == "approved" for clip in state.clips),
                    state.model_dump_json(),
                ),
            )


def list_jobs(st: Settings) -> list[JobState]:
    with closing(_connect(st)) as db:
        rows = db.execute("SELECT state_json FROM jobs ORDER BY created_at DESC").fetchall()
    return [JobState.model_validate_json(row[0]) for row in rows]


def dashboard_stats(st: Settings) -> dict[str, int]:
    import_legacy_jobs(st)
    with closing(_connect(st)) as db:
        row = db.execute(
            """SELECT COUNT(*), COALESCE(SUM(clip_count), 0),
                      COALESCE(SUM(approved_count), 0)
               FROM jobs"""
        ).fetchone()
    return {"projects": row[0], "clips": row[1], "approved": row[2]}


def import_legacy_jobs(st: Settings) -> None:
    """Import existing state.json projects once when the new index is empty."""
    with closing(_connect(st)) as db:
        if db.execute("SELECT 1 FROM jobs LIMIT 1").fetchone():
            return
    for folder in st.output_dir.iterdir():
        if folder.is_dir():
            state_path = JobPaths(st, folder.name).state
            if state_path.is_file():
                try:
                    save_job(st, JobState.model_validate_json(state_path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
