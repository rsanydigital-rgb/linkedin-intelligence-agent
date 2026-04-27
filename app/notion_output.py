"""
notion_output.py
----------------
Notion page output for stakeholder-facing reports (Phase 5 feature).

Each analysis run is pushed as a structured Notion page containing:
  - Executive summary
  - Trend table
  - Key takeaways and opportunities
  - Metadata (sources, elapsed time, run date)

Setup:
  pip install notion-client
  NOTION_API_KEY=secret_...
  NOTION_DATABASE_ID=...   (the database where pages will be created)

IMPORTANT — database schema:
  The database only needs a single "Name" (title) property.
  All other content is written as page blocks, not database properties,
  so you do NOT need to add Topic / Date / Sources / Elapsed columns.
  The page will be created regardless of your database's property schema.

The integration is opt-in — if NOTION_API_KEY is not set, all calls
are silent no-ops and the pipeline is unaffected.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Read env vars lazily (via functions) so that:
#    1. Changes to .env + server restart are always picked up.
#    2. Uvicorn --reload scenarios work correctly.
#    3. The /metrics health-check reflects the live config state.
# ─────────────────────────────────────────────────────────────────────────────
def _key() -> str:
    return os.getenv("NOTION_API_KEY", "").strip()


def _db() -> str:
    return os.getenv("NOTION_DATABASE_ID", "").strip()


def is_available() -> bool:
    return bool(_key() and _db())


# ── Notion rich_text block limit ──────────────────────────────────────────────
_MAX_TEXT = 1990  # Notion API hard limit is 2000; leave a small buffer


def _truncate(text: str) -> str:
    """Truncate text to Notion's 2000-char rich_text limit."""
    if len(text) <= _MAX_TEXT:
        return text
    return text[:_MAX_TEXT - 3] + "..."


# ── Public API ────────────────────────────────────────────────────────────────

def push_report(
    topic: str,
    insights: Dict[str, Any],
    trends: List[Dict[str, Any]],
    sources_analysed: int,
    elapsed_seconds: float,
) -> Optional[str]:
    """
    Create a Notion page for this analysis run.

    Returns the URL of the created page, or None on failure / not configured.

    Robustness notes:
    - Only sets the page "Name" (title) property — avoids 400 errors from
      databases that don't have Topic/Date/Sources columns.
    - All rich content is written as child blocks, not properties.
    - Each block's text is truncated to Notion's 2000-char limit.
    - Detailed error messages are logged so you can diagnose failures.
    """
    notion_key = _key()
    notion_db = _db()

    if not notion_key or not notion_db:
        logger.debug("Notion not configured — skipping output")
        return None

    try:
        from notion_client import Client  # type: ignore
        from notion_client.errors import APIResponseError  # type: ignore
    except ImportError:
        logger.warning(
            "notion-client not installed. Run: pip install notion-client. "
            "Notion output skipped."
        )
        return None

    client = Client(auth=notion_key)

    # Validate the database is accessible before trying to create a page.
    # This surfaces auth / ID errors with a clear message.
    try:
        client.databases.retrieve(database_id=notion_db)
    except Exception as exc:
        logger.error(
            "Notion: cannot access database '%s'. "
            "Check NOTION_DATABASE_ID and that your integration has been shared "
            "with the database. Error: %s",
            notion_db, exc,
        )
        return None

    title = f"Intelligence Report: {topic}"

    try:
        page = client.pages.create(
            parent={"database_id": notion_db},
            # Only set the title property — every Notion database has this.
            # Avoids 400 errors caused by missing custom properties.
            properties={
                "Name": {
                    "title": [{"text": {"content": _truncate(title)}}]
                }
            },
            children=_build_blocks(topic, insights, trends, sources_analysed, elapsed_seconds),
        )
        url = page.get("url", "")
        logger.info("Notion page created: %s", url)
        return url

    except Exception as exc:
        # Log the full error so it's visible in server logs.
        error_detail = str(exc)
        if "401" in error_detail or "unauthorized" in error_detail.lower():
            logger.error(
                "Notion auth failed — NOTION_API_KEY may be expired or invalid. "
                "Error: %s", exc
            )
        elif "400" in error_detail:
            logger.error(
                "Notion returned 400 — likely a property type mismatch. "
                "Your database only needs a 'Name' (title) column; all other "
                "content is written as blocks. Error: %s", exc
            )
        elif "404" in error_detail:
            logger.error(
                "Notion database not found — check NOTION_DATABASE_ID. "
                "Also ensure your integration is shared with the database "
                "(in Notion: open database → ••• → Add connections → select your integration). "
                "Error: %s", exc
            )
        else:
            logger.error("Notion page creation failed: %s", exc)
        return None


# ── Block builders ─────────────────────────────────────────────────────────────

def _build_blocks(
    topic: str,
    insights: Dict[str, Any],
    trends: List[Dict[str, Any]],
    sources: int,
    elapsed: float,
) -> List[Dict]:
    blocks: List[Dict] = []

    # ── Header ────────────────────────────────────────────────────────────────
    blocks.append(_heading(f"🔍 Intelligence Report: {topic}", level=1))
    blocks.append(_callout(
        f"📊  Sources analysed: {sources}  |  "
        f"Elapsed: {round(elapsed, 1)}s  |  "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    ))
    blocks.append(_divider())

    # ── Executive Summary ─────────────────────────────────────────────────────
    summary = str(insights.get("summary", "No summary available."))
    if summary:
        blocks.append(_heading("Executive Summary", level=2))
        # Split long summaries across multiple paragraph blocks
        for chunk in _split_text(summary):
            blocks.append(_paragraph(chunk))

    # ── Top Trends ────────────────────────────────────────────────────────────
    if trends:
        blocks.append(_divider())
        blocks.append(_heading("📈 Top Trends", level=2))
        for t in trends[:8]:
            label = t.get("label", "Unnamed Trend")
            score = t.get("score", 0)
            kws = t.get("keywords", [])[:5]
            kw_str = ", ".join(kws) if kws else "—"
            # Use plain text — Notion rich_text does NOT render **markdown**
            blocks.append(_bullet(f"{label}  (score: {score:.2f})  —  {kw_str}"))

    # ── Key Takeaways ─────────────────────────────────────────────────────────
    takeaways = insights.get("key_takeaways", [])
    if takeaways:
        blocks.append(_divider())
        blocks.append(_heading("💡 Key Takeaways", level=2))
        for item in takeaways:
            for chunk in _split_text(str(item)):
                blocks.append(_bullet(chunk))

    # ── Opportunities ─────────────────────────────────────────────────────────
    opportunities = insights.get("opportunities", [])
    if opportunities:
        blocks.append(_divider())
        blocks.append(_heading("🚀 Opportunity Signals", level=2))
        for item in opportunities:
            for chunk in _split_text(str(item)):
                blocks.append(_bullet(chunk))

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = insights.get("confidence", 0)
    if confidence:
        blocks.append(_divider())
        conf_pct = f"{confidence:.0%}" if confidence <= 1 else f"{confidence:.0f}%"
        blocks.append(_paragraph(f"Analysis confidence: {conf_pct}"))

    return blocks


def _split_text(text: str) -> List[str]:
    """Split text into ≤1990-char chunks, breaking at sentence boundaries."""
    if len(text) <= _MAX_TEXT:
        return [text]
    chunks = []
    while text:
        if len(text) <= _MAX_TEXT:
            chunks.append(text)
            break
        # Try to break at last sentence boundary within limit
        cut = text.rfind(". ", 0, _MAX_TEXT)
        if cut == -1:
            cut = _MAX_TEXT
        else:
            cut += 1  # include the period
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


# ── Primitive block constructors ───────────────────────────────────────────────

def _heading(text: str, level: int = 2) -> Dict:
    kind = {1: "heading_1", 2: "heading_2", 3: "heading_3"}.get(level, "heading_2")
    return {
        "object": "block",
        "type": kind,
        kind: {"rich_text": [{"type": "text", "text": {"content": _truncate(text)}}]},
    }


def _paragraph(text: str) -> Dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": _truncate(text)}}]},
    }


def _bullet(text: str) -> Dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": _truncate(text)}}]
        },
    }


def _callout(text: str) -> Dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": _truncate(text)}}],
            "icon": {"emoji": "📊"},
            "color": "blue_background",
        },
    }


def _divider() -> Dict:
    return {"object": "block", "type": "divider", "divider": {}}


# ── Health check ──────────────────────────────────────────────────────────────

def notion_status() -> Dict[str, Any]:
    """
    Live health check — re-reads env vars each call so it reflects
    the current server state without requiring a restart.
    """
    notion_key = _key()
    notion_db = _db()

    if not notion_key:
        return {"status": "disabled", "reason": "NOTION_API_KEY not set"}
    if not notion_db:
        return {"status": "disabled", "reason": "NOTION_DATABASE_ID not set"}

    try:
        from notion_client import Client  # type: ignore
        client = Client(auth=notion_key)
        db = client.databases.retrieve(database_id=notion_db)
        title_parts = db.get("title", [])
        db_name = title_parts[0].get("plain_text", notion_db) if title_parts else notion_db
        return {
            "status": "connected",
            "database": db_name,
            "database_id": notion_db,
        }
    except Exception as exc:
        err = str(exc)
        if "401" in err or "unauthorized" in err.lower():
            return {
                "status": "error",
                "reason": "Invalid NOTION_API_KEY — check that the token starts with 'secret_'",
                "detail": err,
            }
        if "404" in err:
            return {
                "status": "error",
                "reason": (
                    "Database not found — verify NOTION_DATABASE_ID. "
                    "Also share your integration with the database in Notion: "
                    "open the database → ••• menu → Add connections → select your integration."
                ),
                "detail": err,
            }
        return {"status": "error", "reason": str(exc)}