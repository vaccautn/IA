"""Local SQLite job persistence and transactional, oldest-first claiming."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


class JobNotFound(LookupError):
    """A requested local job does not exist."""


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "jobs.sqlite3"
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, error TEXT)"
            )

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def list(self):
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100"
                )
            ]

    def get(self, job_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return dict(row)

    def add(self, job_id, name):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, 'pending', ?, NULL)",
                (job_id, name, datetime.now(timezone.utc).isoformat()),
            )

    def update(self, job_id, status, error=None):
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, job_id)
            )

    def recover(self, connection=None):
        if connection is None:
            with self.connect() as conn:
                self.recover(conn)
            return
        connection.execute(
            "UPDATE jobs SET status='failed', error='El servicio se interrumpió. Volvé a cargar el video para reintentar.' WHERE status='processing'"
        )

    def claim(self):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE jobs SET status='processing' WHERE id=?", (row["id"],)
                )
                return row["id"]
