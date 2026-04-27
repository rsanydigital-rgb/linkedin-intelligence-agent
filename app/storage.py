"""
storage.py
----------
PostgreSQL persistence layer for trend history and API usage tracking.

Phase 4 feature. All operations degrade gracefully when DATABASE_URL is
not set — the pipeline continues with local file-based trend history.

Tables created on first use (no migration tool required for MVP):
  - trend_runs       : one row per analysis run
  - api_usage_log    : one row per external API call
  - topic_history    : summary view for the dashboard

Setup:
  pip install psycopg2-binary
  DATABASE_URL=postgresql://user:password@localhost:5432/linkedin_agent
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATABASE_URL = os.getenv("DATABASE_URL", "")

# Lazy singleton connection
_conn = None


def _get_conn():
    """Return a psycopg2 connection, or None if not configured."""
    global _conn

    if _conn is not None:
        try:
            # Cheap liveness check
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    if not _DATABASE_URL:
        logger.debug("DATABASE_URL not set — PostgreSQL persistence disabled")
        return None

    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore

        conn = psycopg2.connect(_DATABASE_URL)
        conn.autocommit = True
        _conn = conn
        _ensure_schema(conn)
        logger.info("PostgreSQL connected — persistence enabled")
        return _conn
    except Exception as exc:
        logger.warning("PostgreSQL unavailable — persistence disabled: %s", exc)
        return None


def _ensure_schema(conn) -> None:
    """Create tables if they do not exist (idempotent)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS trend_runs (
        id            SERIAL PRIMARY KEY,
        topic         TEXT        NOT NULL,
        captured_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        elapsed_secs  FLOAT,
        sources_count INTEGER,
        trends_json   JSONB,
        insights_json JSONB
    );

    CREATE INDEX IF NOT EXISTS trend_runs_topic_idx ON trend_runs (topic);
    CREATE INDEX IF NOT EXISTS trend_runs_captured_idx ON trend_runs (captured_at DESC);

    CREATE TABLE IF NOT EXISTS api_usage_log (
        id            SERIAL PRIMARY KEY,
        logged_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        source        TEXT        NOT NULL,
        query_hash    TEXT,
        status_code   INTEGER,
        response_ms   FLOAT,
        record_count  INTEGER,
        error_message TEXT
    );

    CREATE TABLE IF NOT EXISTS topic_history (
        topic         TEXT        PRIMARY KEY,
        last_run_at   TIMESTAMPTZ,
        total_runs    INTEGER     DEFAULT 0,
        avg_elapsed   FLOAT
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


# ---------------------------------------------------------------------------
# Public write functions
# ---------------------------------------------------------------------------

def save_run(
    topic: str,
    elapsed_secs: float,
    sources_count: int,
    trends: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> Optional[int]:
    """Persist a completed analysis run. Returns the new row id, or None."""
    conn = _get_conn()
    if conn is None:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trend_runs (topic, captured_at, elapsed_secs, sources_count, trends_json, insights_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    topic,
                    datetime.now(timezone.utc),
                    elapsed_secs,
                    sources_count,
                    json.dumps(trends),
                    json.dumps(insights),
                ),
            )
            row_id = cur.fetchone()[0]

            # Upsert topic summary
            cur.execute(
                """
                INSERT INTO topic_history (topic, last_run_at, total_runs, avg_elapsed)
                VALUES (%s, NOW(), 1, %s)
                ON CONFLICT (topic) DO UPDATE
                  SET last_run_at = EXCLUDED.last_run_at,
                      total_runs  = topic_history.total_runs + 1,
                      avg_elapsed = (topic_history.avg_elapsed * topic_history.total_runs + EXCLUDED.avg_elapsed)
                                    / (topic_history.total_runs + 1)
                """,
                (topic, elapsed_secs),
            )

        logger.info("Saved run id=%d for topic '%s'", row_id, topic)
        return row_id
    except Exception as exc:
        logger.error("Failed to save run for topic '%s': %s", topic, exc)
        return None


def log_api_call(
    source: str,
    query_hash: str,
    status_code: int,
    response_ms: float,
    record_count: int,
    error_message: str = "",
) -> None:
    """Log a single external API call for cost and reliability tracking."""
    conn = _get_conn()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_usage_log (source, query_hash, status_code, response_ms, record_count, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (source, query_hash, status_code, response_ms, record_count, error_message),
            )
    except Exception as exc:
        logger.warning("Failed to log API call for source '%s': %s", source, exc)


# ---------------------------------------------------------------------------
# Public read functions
# ---------------------------------------------------------------------------

def get_all_recent_runs(limit: int = 25) -> List[Dict[str, Any]]:
    """Return the most recent *limit* analysis runs across ALL topics (for history page)."""
    conn = _get_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (topic)
                    topic, captured_at, elapsed_secs, sources_count
                FROM trend_runs
                ORDER BY topic, captured_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "topic": r[0],
                    "captured_at": r[1].isoformat() if r[1] else None,
                    "elapsed_secs": r[2],
                    "sources_count": r[3],
                }
                for r in sorted(rows, key=lambda x: x[1] or "", reverse=True)
            ]
    except Exception as exc:
        logger.error("Failed to fetch all recent runs: %s", exc)
        return []


def get_recent_runs(topic: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent *limit* runs for a topic."""
    conn = _get_conn()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, captured_at, elapsed_secs, sources_count, trends_json
                FROM trend_runs
                WHERE topic = %s
                ORDER BY captured_at DESC
                LIMIT %s
                """,
                (topic, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "captured_at": r[1].isoformat(),
                    "elapsed_secs": r[2],
                    "sources_count": r[3],
                    "trends": r[4],
                }
                for r in rows
            ]
    except Exception as exc:
        logger.error("Failed to fetch runs for topic '%s': %s", topic, exc)
        return []


def get_api_usage_summary(days: int = 7) -> List[Dict[str, Any]]:
    """Return per-source API usage totals for the last *days* days."""
    conn = _get_conn()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source,
                       COUNT(*)                                AS total_calls,
                       ROUND(AVG(response_ms)::numeric, 1)    AS avg_ms,
                       SUM(record_count)                       AS total_records,
                       COUNT(*) FILTER (WHERE status_code >= 400) AS error_count
                FROM api_usage_log
                WHERE logged_at >= NOW() - INTERVAL '%s days'
                GROUP BY source
                ORDER BY total_calls DESC
                """,
                (days,),
            )
            rows = cur.fetchall()
            return [
                {
                    "source": r[0],
                    "total_calls": r[1],
                    "avg_response_ms": float(r[2] or 0),
                    "total_records": r[3],
                    "error_count": r[4],
                }
                for r in rows
            ]
    except Exception as exc:
        logger.error("Failed to fetch API usage summary: %s", exc)
        return []


def storage_status() -> Dict[str, Any]:
    """Health check for the /metrics endpoint."""
    conn = _get_conn()
    if conn is None:
        return {"status": "disabled", "database_url": None}

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trend_runs")
            total_runs = cur.fetchone()[0]
        return {
            "status": "connected",
            "total_runs": total_runs,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}