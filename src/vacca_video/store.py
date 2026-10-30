"""Local SQLite job persistence and transactional, oldest-first claiming."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading


class JobNotFound(LookupError):
    """A requested local job does not exist."""


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory / "jobs.sqlite3"
        self._job_locks_guard = threading.Lock()
        self._job_locks = {}
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, error TEXT)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "idempotency_key" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN idempotency_key TEXT")
            if "source_sha256" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN source_sha256 TEXT")
            if "source_size" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN source_size INTEGER")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_idempotency_key "
                "ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
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

    def get_by_idempotency_key(self, idempotency_key):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
        return dict(row) if row is not None else None

    @contextmanager
    def lock_job(self, job_id):
        """Serialize terminal cleanup and retry while sharing one process-local Store."""
        with self._job_locks_guard:
            entry = self._job_locks.get(job_id)
            if entry is None:
                entry = (threading.RLock(), 0)
            lock, users = entry
            self._job_locks[job_id] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._job_locks_guard:
                lock, users = self._job_locks[job_id]
                if users == 1:
                    del self._job_locks[job_id]
                else:
                    self._job_locks[job_id] = (lock, users - 1)

    def add(self, job_id, name, idempotency_key=None, source_sha256=None, source_size=None):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, name, status, created_at, error, idempotency_key, source_sha256, source_size) "
                "VALUES (?, ?, 'pending', ?, NULL, ?, ?, ?)",
                (
                    job_id,
                    name,
                    datetime.now(timezone.utc).isoformat(),
                    idempotency_key,
                    source_sha256,
                    source_size,
                ),
            )

    def update(self, job_id, status, error=None):
        with self.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, job_id)
            )

    def retry_failed(self, job_id):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE jobs SET status='pending', error=NULL WHERE id=? AND status='failed'",
                (job_id,),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return dict(row)

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
