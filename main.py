"""
main.py  —  LinkedIn Intelligence Agent v2.1
---------------------------------------------
Fixes vs v2.0:
  - Added GET /config_status  →  frontend config banner
  - /schedule now returns clear "not configured" vs "n8n 404" distinction
  - /notify/digest gracefully handles missing SendGrid key
  - /notify/slack gracefully handles missing webhook
  - /export/notion gracefully handles missing Notion keys
  - All Phase 5 endpoints return structured { success, detail } on failure
    instead of raising bare 502 so the frontend can show friendly messages
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as _http
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from app.query_expansion import expand_topic
from app.query_intelligence import analyze_query
from app.collector import collect_with_breakdown
from app.processor import process
from app.insights import generate_insights
from app.logging_setup import configure_logging, log_stage
from app.opportunities import score_opportunities
from app.trend_history import load_previous_trends, save_trends, compute_trend_evolution
from app.validation import validate_collected_posts
from app.sentiment import analyse as sentiment_analyse
# LangGraph orchestrator — replaces linear pipeline in /analyze endpoint
from app.langgraph_orchestrator import run_pipeline as _lg_run_pipeline  # noqa: F401 (imported for side-effect: graph compile)

from app.cache import get_cached_result, set_cached_result, make_cache_key, cache_stats
from app.storage import save_run, get_api_usage_summary, storage_status, get_all_recent_runs
from app.notifier import send_trend_alert, send_daily_digest, notifier_status
from app.notion_output import push_report, notion_status

configure_logging()
logger = logging.getLogger("main")

# In-memory store of the most recent analysis result per topic
_last_results: Dict[str, Any] = {}


app = FastAPI(
    title="LinkedIn Intelligence Agent",
    description="AI-powered topic intelligence pipeline",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# LinkedIn timeline helper
# ---------------------------------------------------------------------------

def _timeline_to_datetime(timeline: str) -> datetime:
    """Convert a timeline string to a UTC cutoff datetime."""
    now = datetime.now(timezone.utc)
    mapping = {
        "24h":  timedelta(hours=24),
        "7d":   timedelta(days=7),
        "30d":  timedelta(days=30),
        "90d":  timedelta(days=90),
        "180d": timedelta(days=180),
    }
    delta = mapping.get(timeline, timedelta(days=30))
    return now - delta


def _make_analysis_cache_key(
    topic: str,
    sources: str,
    linkedin_post_count: int,
    linkedin_timeline: str,
    linkedin_region: str,
) -> str:
    return make_cache_key(f"{topic}|{sources}|{linkedin_post_count}|{linkedin_timeline}|{linkedin_region}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    sources: str = Field(default="web_news")
    linkedin_post_count: int = Field(default=25, ge=5, le=200)
    linkedin_timeline: str = Field(default="30d")  # "24h","7d","30d","90d","180d"
    linkedin_region: str = Field(default="global")


class TrendItem(BaseModel):
    label: str
    keywords: List[str]
    score: float


class TrendEvolutionItem(BaseModel):
    trend: str
    previous_score: float
    current_score: float
    change: str
    status: str


class OpportunityItem(BaseModel):
    trend: str
    opportunity_score: float
    reason: str


class QueryIntelligence(BaseModel):
    original_query: str
    query_type: str
    reason: str
    effective_topic: str
    suggestions: List[str]


class InsightBlock(BaseModel):
    summary: str
    key_takeaways: List[str]
    opportunities: List[str]
    confidence: float


class QuerySourceItem(BaseModel):
    source: str
    url: str
    content_preview: str
    collected_at: str
    sentiment: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None  # likes/comments/shares/author for LinkedIn


class QueryResultGroup(BaseModel):
    query: str
    results: List[QuerySourceItem]


class AnalyzeResponse(BaseModel):
    topic: str
    query_intelligence: QueryIntelligence
    queries: List[str]
    sources_analysed: int
    query_results: List[QueryResultGroup]
    trends: List[TrendItem]
    trend_evolution: List[TrendEvolutionItem]
    opportunity_scores: List[OpportunityItem]
    top_keywords: List[str]
    insights: InsightBlock
    elapsed_seconds: float
    cached: bool = False
    notion_url: Optional[str] = None
    sentiment_summary: Optional[Dict[str, Any]] = None
    linkedin_post_count: int = 25
    linkedin_timeline: str = "30d"
    linkedin_region: str = "global"


class NotionExportRequest(BaseModel):
    topic: str


class DigestRequest(BaseModel):
    topic: str
    email: str = ""
    report_data: Dict[str, Any] = Field(default_factory=dict)


class SlackRequest(BaseModel):
    topic: str


class ScheduleRequest(BaseModel):
    topic: str
    cron_expression: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>")


@app.get("/{page_name}.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_page(page_name: str):
    """Serve any .html page from the frontend/ directory by name."""
    html_path = Path(__file__).parent / "frontend" / f"{page_name}.html"
    if html_path.exists() and html_path.stat().st_size > 0:
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content=f"<h1 style='font-family:monospace;color:#cdd6e8;background:#080b10;padding:40px'>Page not found: {page_name}.html</h1>",
        status_code=404,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/config_status")
async def config_status():
    """
    Returns which integrations are configured so the frontend
    can show a helpful warning banner for unconfigured services.
    """
    return {
        "serpapi":   bool(os.getenv("SERPAPI_KEY", "").strip()),
        "newsapi":   bool(os.getenv("NEWS_API_KEY", "").strip()),
        "apify":     bool(os.getenv("APIFY_API_TOKEN", "").strip()),
        "apify_engagement_note": "Engagement metrics (likes/comments/shares) require a paid Apify plan. Free tier returns post text only.",
        "slack":     bool(os.getenv("SLACK_WEBHOOK_URL", "").strip()),
        "sendgrid":  bool(os.getenv("SENDGRID_API_KEY", "").strip()),
        "notion":    bool(os.getenv("NOTION_API_KEY", "").strip() and os.getenv("NOTION_DATABASE_ID", "").strip()),
        "n8n":       bool(os.getenv("N8N_WEBHOOK_URL", "").strip()),
        "apify":     bool(os.getenv("APIFY_API_TOKEN", "").strip()),
        "redis":     bool(os.getenv("REDIS_URL", "").strip()),
        "postgres":  bool(os.getenv("DATABASE_URL", "").strip()),
        "llm_model": os.getenv("LLM_ORCHESTRATOR_MODEL", "not set"),
        "llm_base":  os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }


@app.get("/metrics")
async def metrics():
    return {
        "cache": cache_stats(),
        "storage": storage_status(),
        "notifier": notifier_status(),
        "notion": notion_status(),
        "api_usage_7d": get_api_usage_summary(days=7),
    }

@app.get("/history")
async def history(limit: int = 25):
    """Return the most recent analysis runs across all topics (for Reports page history table)."""
    rows = get_all_recent_runs(limit=limit)
    return rows



@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    refresh: bool = Query(default=False),
):
    t0 = time.perf_counter()
    topic = request.topic.strip()
    logger.info("=== Analysis: '%s' refresh=%s sources=%s ===", topic, refresh, request.sources)
    cache_key = _make_analysis_cache_key(
        topic,
        request.sources,
        request.linkedin_post_count,
        request.linkedin_timeline,
        request.linkedin_region,
    )

    # ── Cache lookup (skip if refresh=true) ──────────────────────────────
    if not refresh:
        cached = get_cached_result(cache_key)
        if cached:
            cached["cached"] = True
            _last_results[topic.lower()] = cached
            return AnalyzeResponse(**{k: v for k, v in cached.items() if k in AnalyzeResponse.model_fields})

    # ── LangGraph pipeline ────────────────────────────────────────────────
    from app.langgraph_orchestrator import run_pipeline
    linkedin_date_since = _timeline_to_datetime(request.linkedin_timeline) if "linkedin" in request.sources else None
    result = run_pipeline(
        topic=topic,
        sources=request.sources,
        linkedin_post_count=request.linkedin_post_count,
        linkedin_date_since=linkedin_date_since,
        linkedin_region=request.linkedin_region,
    )

    if result.get("error"):
        err = result["error"]
        if "No valid data" in err or "No posts met" in err:
            raise HTTPException(404, err)
        raise HTTPException(502, err)

    # Unpack pipeline outputs
    query_analysis         = result["query_analysis"]
    queries                = result["queries"]
    validated_query_groups = result["validated_query_groups"] or {}
    processed              = result["processed"]
    insights_dict          = result["insights"]
    trend_evolution        = result["trend_evolution"] or []
    opportunity_scores     = result["opportunity_scores"] or []
    elapsed                = result["elapsed_seconds"]

    log_stage(logger, layer="langgraph_pipeline", status="success",
              records=len(processed.documents), latency_ms=elapsed * 1000)

    # ── Non-fatal side effects ────────────────────────────────────────────
    notion_url = None
    try:
        notion_url = push_report(
            topic=topic, insights=insights_dict, trends=processed.trends,
            sources_analysed=len(processed.documents), elapsed_seconds=elapsed,
        )
    except Exception:
        logger.exception("Notion output failed (non-fatal)")

    try:
        send_trend_alert(topic=topic, trends=processed.trends)
    except Exception:
        logger.exception("Trend alert failed (non-fatal)")

    # ── Build response ────────────────────────────────────────────────────
    response_data = dict(
        topic=topic,
        query_intelligence=query_analysis,
        queries=queries,
        sources_analysed=len(processed.documents),
        query_results=[
            dict(query=q, results=[
                dict(source=i.source, url=i.url,
                     content_preview=i.content[:240] + ("..." if len(i.content) > 240 else ""),
                     collected_at=i.collected_at.isoformat(),
                     sentiment=sentiment_analyse(i.content),
                     metadata=i.metadata)
                for i in validated_query_groups.get(q, [])
            ])
            for q in queries
        ],
        trends=[dict(t) for t in processed.trends],
        trend_evolution=trend_evolution,
        opportunity_scores=opportunity_scores,
        top_keywords=processed.top_keywords,
        insights=insights_dict,
        elapsed_seconds=elapsed,
        cached=False,
        notion_url=notion_url,
        sentiment_summary=getattr(processed, "sentiment_summary", {}),
        linkedin_post_count=request.linkedin_post_count,
        linkedin_timeline=request.linkedin_timeline,
        linkedin_region=request.linkedin_region,
    )

    _last_results[topic.lower()] = response_data

    try:
        set_cached_result(cache_key, response_data)
    except Exception:
        logger.exception("Cache write failed (non-fatal)")

    try:
        save_run(topic=topic, elapsed_secs=elapsed, sources_count=len(processed.documents),
                 trends=processed.trends, insights=insights_dict)
    except Exception:
        logger.exception("PostgreSQL save failed (non-fatal)")

    logger.info("=== Done in %.2fs ===", elapsed)

    return AnalyzeResponse(
        topic=response_data["topic"],
        query_intelligence=QueryIntelligence(**response_data["query_intelligence"]),
        queries=response_data["queries"],
        sources_analysed=response_data["sources_analysed"],
        query_results=[QueryResultGroup(**g) for g in response_data["query_results"]],
        trends=[TrendItem(**t) for t in response_data["trends"]],
        trend_evolution=[TrendEvolutionItem(**i) for i in response_data["trend_evolution"]],
        opportunity_scores=[OpportunityItem(**i) for i in response_data["opportunity_scores"]],
        top_keywords=response_data["top_keywords"],
        insights=InsightBlock(**response_data["insights"]),
        elapsed_seconds=response_data["elapsed_seconds"],
        cached=False,
        notion_url=response_data.get("notion_url"),
        sentiment_summary=response_data.get("sentiment_summary"),
    )


# ---------------------------------------------------------------------------
# Phase 5 endpoints — all return { success, detail } on failure
# ---------------------------------------------------------------------------

@app.get("/notion/status")
async def notion_status_endpoint():
    """Live Notion connection health check (re-reads env vars each call)."""
    from app.notion_output import notion_status as _notion_status
    return _notion_status()


@app.post("/export/notion")
async def export_notion_endpoint(request: NotionExportRequest):
    notion_key = os.getenv("NOTION_API_KEY", "").strip()
    notion_db  = os.getenv("NOTION_DATABASE_ID", "").strip()

    if not notion_key:
        raise HTTPException(503, "NOTION_API_KEY not set — add it to your .env file")
    if not notion_db:
        raise HTTPException(503, "NOTION_DATABASE_ID not set — add it to your .env file")

    topic = request.topic.strip()
    result = _last_results.get(topic.lower()) or get_cached_result(make_cache_key(topic))
    if not result:
        raise HTTPException(404, "No analysis found for this topic. Run /analyze first.")

    try:
        url = push_report(
            topic=topic, insights=result.get("insights", {}),
            trends=result.get("trends", []),
            sources_analysed=result.get("sources_analysed", 0),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
        )
        if url is None:
            # push_report logs the specific error; surface it to the frontend
            status = notion_status()
            detail = status.get("reason") or status.get("detail") or "Unknown error — check server logs"
            raise HTTPException(502, f"Notion page creation failed: {detail}")
        return {"success": True, "notion_url": url}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Notion export failed")
        raise HTTPException(502, f"Notion export failed: {exc}")


@app.post("/notify/digest")
async def notify_digest_endpoint(request: DigestRequest):
    email = request.email.strip()
    if not email:
        raise HTTPException(400, "email field is required")

    sendgrid_key = os.getenv("SENDGRID_API_KEY", "").strip()
    if not sendgrid_key:
        raise HTTPException(503, "Email not configured — add SENDGRID_API_KEY to .env")

    # Determine sender — must be a SendGrid-verified email
    from_email = os.getenv("ALERT_EMAIL_TO", email)

    topic = request.topic.strip()
    stored = _last_results.get(topic.lower(), {})
    report = {**stored, **request.report_data} if request.report_data else stored
    if not report:
        report = {"topic": topic}

    insights = report.get("insights", {})
    trends   = report.get("trends", [])[:5]
    summary  = insights.get("summary", "No summary available.")

    trend_rows = "".join(
        f"<tr><td style='padding:4px 8px'>{t.get('label','')}</td>"
        f"<td style='padding:4px 8px;text-align:right'>{float(t.get('score',0)):.2f}</td></tr>"
        for t in trends
    )
    takeaway_li = "".join(f"<li>{t}</li>" for t in insights.get("key_takeaways", []))
    opp_li      = "".join(f"<li>{o}</li>" for o in insights.get("opportunities", []))

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px;margin:auto;color:#222">
      <h2 style="color:#0a66c2">Intelligence Digest: {topic}</h2>
      <p>{summary}</p>
      <h3>Top Trends</h3>
      <table style="border-collapse:collapse;width:100%">
        <thead><tr style="background:#f0f4ff">
          <th style="padding:4px 8px;text-align:left">Trend</th>
          <th style="padding:4px 8px;text-align:right">Score</th>
        </tr></thead>
        <tbody>{trend_rows}</tbody>
      </table>
      <h3>Key Takeaways</h3><ul>{takeaway_li}</ul>
      <h3>Opportunities</h3><ul>{opp_li}</ul>
      <hr/>
      <p style="font-size:11px;color:#888">
        Sources: {report.get('sources_analysed', 0)} | Time: {report.get('elapsed_seconds', 0)}s
      </p>
    </body></html>
    """

    try:
        resp = _http.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {sendgrid_key}", "Content-Type": "application/json"},
            json={
                "personalizations": [{"to": [{"email": email}]}],
                "from": {"email": from_email, "name": "Intelligence Agent"},
                "subject": f"Intelligence Report: {topic}",
                "content": [{"type": "text/html", "value": html_body}],
            },
            timeout=15,
        )
        if resp.status_code in (200, 202):
            logger.info("Email sent to %s", email)
            return {"success": True}
        # SendGrid returns errors in body
        error_body = resp.json() if resp.content else {}
        errors = error_body.get("errors", [{}])
        msg = errors[0].get("message", resp.text[:200]) if errors else resp.text[:200]
        raise HTTPException(502, f"SendGrid error: {msg}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Email digest failed")
        raise HTTPException(502, f"Email failed: {exc}")


@app.post("/notify/slack")
async def notify_slack_endpoint(request: SlackRequest):
    slack_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not slack_url:
        raise HTTPException(503, "Slack not configured — add SLACK_WEBHOOK_URL to .env")

    topic  = request.topic.strip()
    result = _last_results.get(topic.lower(), {})
    insights = result.get("insights", {})
    summary  = insights.get("summary", f"Analysis complete for: {topic}")
    trends   = result.get("trends", [])
    top_labels = " · ".join(t.get("label", "") for t in trends[:4]) or "N/A"

    try:
        resp = _http.post(
            slack_url,
            json={
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": f"📊 Intelligence Report: {topic}"}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*Top Trends:* {top_labels}"}},
                    {"type": "divider"},
                ],
                "text": f"Intelligence Report: {topic}",
            },
            timeout=10,
        )
        if resp.status_code == 200 and resp.text == "ok":
            return {"success": True}
        raise HTTPException(502, f"Slack returned: {resp.status_code} {resp.text[:100]}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Slack failed")
        raise HTTPException(502, f"Slack error: {exc}")




@app.get("/market-intel")
async def market_intel_endpoint(topic: str = Query(..., min_length=1)):
    from app.market_intel_final import interest_over_time, job_postings
    topic = topic.strip()
    if not topic:
        raise HTTPException(400, "topic is required")
    try:
        trends_data = interest_over_time(topic)
        jobs_data   = job_postings(topic)
        return {"topic": topic, "trends": trends_data, "jobs": jobs_data}
    except Exception as exc:
        logger.exception("market-intel failed")
        raise HTTPException(500, f"Market intel error: {exc}")


@app.post("/schedule")
async def schedule_report(request: ScheduleRequest):
    n8n_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not n8n_url:
        raise HTTPException(503, "n8n not configured — add N8N_WEBHOOK_URL to .env")

    payload = {
        "topic": request.topic,
        "cron_expression": request.cron_expression,
        "pipeline_url": os.getenv("APP_BASE_URL", "http://localhost:8000") + "/analyze",
    }

    try:
        resp = _http.post(n8n_url, json=payload, timeout=10)
        # n8n returns 200 with "Workflow was started" on success
        if resp.status_code == 200:
            return {"success": True, "message": f"Scheduled '{request.topic}' with cron: {request.cron_expression}"}
        if resp.status_code == 404:
            raise HTTPException(
                502,
                "n8n returned 404 — the webhook URL is wrong or the workflow is not activated. "
                "In n8n: open your workflow → click the Webhook node → switch from 'Test URL' to 'Production URL' → activate the workflow toggle (top-right)."
            )
        raise HTTPException(502, f"n8n returned {resp.status_code}: {resp.text[:200]}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("n8n scheduling failed")
        raise HTTPException(502, f"Scheduling error: {exc}")