import json
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

from models import IdeaInput, ValidationResult


class Storage:
    def __init__(self, db_path: str = "idea_engine.db"):
        self.db_path = db_path
        # For :memory: keep a single persistent connection; file DBs use per-call connections.
        self._memory_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close(self, conn: sqlite3.Connection):
        # The shared in-memory connection must stay open for the lifetime of
        # the Storage; only per-call file connections are closed.
        if self._memory_conn is None:
            conn.close()

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    submitter   TEXT,
                    track       TEXT,
                    stage       TEXT,
                    score_a     INTEGER,
                    score_b     INTEGER,
                    decision_a  TEXT,
                    decision_b  TEXT,
                    triage_score INTEGER,
                    quick_win   INTEGER DEFAULT 0,
                    strategic_bet INTEGER DEFAULT 0,
                    created_at  TEXT,
                    full_json   TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    def save(self, result: ValidationResult) -> str:
        if not result.id:
            result.id = str(uuid.uuid4())[:8]

        score_a = result.scores.composite_score_a if result.scores else None
        score_b = result.scores.composite_score_b if result.scores else None
        decision_a = result.scores.decision_a if result.scores else None
        decision_b = result.scores.decision_b if result.scores else None
        triage_score = result.triage.total_score if result.triage else None
        quick_win = int(result.scores.quick_win_flag) if result.scores else 0
        strategic_bet = int(result.scores.strategic_bet_flag) if result.scores else 0

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO validations
                (id, title, submitter, track, stage, score_a, score_b,
                 decision_a, decision_b, triage_score, quick_win, strategic_bet,
                 created_at, full_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.id,
                    result.idea.title,
                    result.idea.submitter,
                    result.idea.track.value,
                    result.stage_reached,
                    score_a,
                    score_b,
                    decision_a,
                    decision_b,
                    triage_score,
                    quick_win,
                    strategic_bet,
                    result.idea.created_at,
                    result.model_dump_json(),
                ),
            )
            conn.commit()
        finally:
            self._close(conn)
        return result.id

    def load(self, idea_id: str) -> Optional[ValidationResult]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT full_json FROM validations WHERE id = ?", (idea_id,)
            ).fetchone()
        finally:
            self._close(conn)
        if row:
            return ValidationResult.model_validate_json(row["full_json"])
        return None

    def list_all(self) -> List[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, title, submitter, track, stage,
                       score_a, score_b, decision_a, decision_b,
                       triage_score, quick_win, strategic_bet, created_at
                FROM validations ORDER BY created_at DESC
                """
            ).fetchall()
        finally:
            self._close(conn)
        return [dict(r) for r in rows]

    def delete(self, idea_id: str):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM validations WHERE id = ?", (idea_id,))
            conn.commit()
        finally:
            self._close(conn)
