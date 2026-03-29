"""
Experiment Tracker

Persistent SQLite-based logging for query cycles, latency metrics,
and human feedback. Stores data in data/experiments.db.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "experiments.db"


class ExperimentTracker:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    question TEXT,
                    answer TEXT,
                    confidence_score REAL,
                    grounding_status TEXT,
                    flagged INTEGER,
                    embed_latency_ms REAL,
                    retrieve_latency_ms REAL,
                    generate_latency_ms REAL,
                    critic_latency_ms REAL,
                    total_latency_ms REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    rating TEXT,
                    comment TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def log(self, record: dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO query_log (
                    timestamp, question, answer, confidence_score, grounding_status, flagged,
                    embed_latency_ms, retrieve_latency_ms, generate_latency_ms,
                    critic_latency_ms, total_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    record.get("question"),
                    record.get("answer"),
                    record.get("confidence_score"),
                    record.get("grounding_status"),
                    1 if record.get("flagged") else 0,
                    record.get("embed_latency_ms"),
                    record.get("retrieve_latency_ms"),
                    record.get("generate_latency_ms"),
                    record.get("critic_latency_ms"),
                    record.get("total_latency_ms"),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent(self, n: int) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(confidence_score) as avg_confidence,
                    AVG(CASE WHEN flagged=1 THEN 1.0 ELSE 0.0 END) as flag_rate,
                    AVG(embed_latency_ms) as avg_embed,
                    AVG(retrieve_latency_ms) as avg_retrieve,
                    AVG(generate_latency_ms) as avg_generate,
                    AVG(critic_latency_ms) as avg_critic,
                    AVG(total_latency_ms) as avg_total
                FROM query_log
            """).fetchone()
            return {
                "total_queries": row[0],
                "avg_confidence_score": round(row[1], 4) if row[1] is not None else 0.0,
                "flag_rate": round(row[2], 4) if row[2] is not None else 0.0,
                "avg_embed_latency_ms": round(row[3], 2) if row[3] is not None else 0.0,
                "avg_retrieve_latency_ms": round(row[4], 2) if row[4] is not None else 0.0,
                "avg_generate_latency_ms": round(row[5], 2) if row[5] is not None else 0.0,
                "avg_critic_latency_ms": round(row[6], 2) if row[6] is not None else 0.0,
                "avg_total_latency_ms": round(row[7], 2) if row[7] is not None else 0.0,
            }

    def log_feedback(self, query_id: int, rating: str, comment: str = "") -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO feedback (query_id, rating, comment, timestamp) VALUES (?, ?, ?, ?)",
                (query_id, rating, comment, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cursor.lastrowid

    def get_feedback_summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN rating='up' THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN rating='down' THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """).fetchone()
            total = row[0]
            thumbs_up = row[1] or 0
            thumbs_down = row[2] or 0

            up_agree = conn.execute("""
                SELECT COUNT(*) FROM feedback f
                JOIN query_log q ON f.query_id = q.id
                WHERE f.rating='up' AND q.flagged=0
            """).fetchone()[0]

            down_disagree = conn.execute("""
                SELECT COUNT(*) FROM feedback f
                JOIN query_log q ON f.query_id = q.id
                WHERE f.rating='down' AND q.flagged=0
            """).fetchone()[0]

            agreement_rate = round(up_agree / thumbs_up, 4) if thumbs_up > 0 else 0.0
            disagreement_rate = round(down_disagree / thumbs_down, 4) if thumbs_down > 0 else 0.0

            return {
                "total_feedback": total,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "agreement_rate": agreement_rate,
                "disagreement_rate": disagreement_rate,
            }

    def get_metrics(self) -> dict:
        summary = self.get_summary()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT total_latency_ms FROM query_log WHERE total_latency_ms IS NOT NULL"
                " ORDER BY total_latency_ms"
            ).fetchall()

        if rows:
            values = [r[0] for r in rows]
            p95_idx = max(0, int(len(values) * 0.95) - 1)
            p95 = round(values[p95_idx], 2)
        else:
            p95 = 0.0

        return {
            "total_queries": summary["total_queries"],
            "avg_total_latency_ms": summary["avg_total_latency_ms"],
            "avg_embed_latency_ms": summary["avg_embed_latency_ms"],
            "avg_retrieve_latency_ms": summary["avg_retrieve_latency_ms"],
            "avg_generate_latency_ms": summary["avg_generate_latency_ms"],
            "avg_critic_latency_ms": summary["avg_critic_latency_ms"],
            "flag_rate": summary["flag_rate"],
            "avg_confidence_score": summary["avg_confidence_score"],
            "p95_total_latency_ms": p95,
        }
