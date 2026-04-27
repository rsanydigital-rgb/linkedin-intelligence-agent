"""
notifier.py
-----------
Alert and delivery dispatcher (Phase 5 feature).

Supports three channels, all opt-in via environment variables:
  - Slack webhook (SLACK_WEBHOOK_URL)
  - Email via SendGrid (SENDGRID_API_KEY + ALERT_EMAIL_TO)
  - Generic webhook (ALERT_WEBHOOK_URL) — for n8n, Zapier, etc.

All channels degrade gracefully: if a channel is not configured the call
is a silent no-op. Errors are logged but never propagate to the pipeline.

Usage:
  from app.notifier import send_trend_alert, send_daily_digest

  send_trend_alert(topic="Robotic Technologies", trends=processed.trends)
  send_daily_digest(report_data=response_dict)
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SLACK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
_SENDGRID_KEY = os.getenv("SENDGRID_API_KEY", "")
_ALERT_EMAIL = os.getenv("ALERT_EMAIL_TO", "")
_ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def send_trend_alert(topic: str, trends: List[Dict[str, Any]]) -> None:
    """
    Fire a concise alert when new trends are detected for *topic*.
    Called from main.py after each successful run.
    """
    if not trends:
        return

    top = trends[:3]
    lines = [f"• *{t['label']}* (score {t['score']:.2f})" for t in top]
    body = "\n".join(lines)

    message = f":bar_chart: *New trends detected for: {topic}*\n{body}"

    _send_slack(message)
    _send_webhook({"event": "trend_alert", "topic": topic, "trends": top})


def send_daily_digest(report_data: Dict[str, Any]) -> None:
    """
    Send a formatted daily digest email and/or Slack summary.
    Triggered externally by n8n on a cron schedule via POST /notify/digest.
    """
    topic = report_data.get("topic", "Unknown")
    insights = report_data.get("insights", {})
    summary = insights.get("summary", "No summary available.")
    elapsed = report_data.get("elapsed_seconds", 0)
    sources = report_data.get("sources_analysed", 0)

    slack_msg = (
        f":newspaper: *Daily Intelligence Digest — {topic}*\n"
        f"{summary}\n"
        f"_Sources: {sources} | Time: {elapsed}s_"
    )
    _send_slack(slack_msg)

    if _SENDGRID_KEY and _ALERT_EMAIL:
        html = _build_email_html(report_data)
        _send_sendgrid_email(
            subject=f"[LinkedIn Intelligence] Daily Digest: {topic}",
            html_content=html,
        )

    _send_webhook({"event": "daily_digest", "data": report_data})


def send_pipeline_error(topic: str, layer: str, error: str) -> None:
    """
    Alert on critical pipeline failures (all sources failed, LLM down, etc.).
    """
    message = (
        f":rotating_light: *Pipeline failure*\n"
        f"Topic: `{topic}` | Layer: `{layer}`\n"
        f"Error: {error}"
    )
    _send_slack(message)
    _send_webhook({"event": "pipeline_error", "topic": topic, "layer": layer, "error": error})


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------

def _send_slack(message: str) -> None:
    if not _SLACK_URL:
        return
    try:
        import requests
        resp = requests.post(
            _SLACK_URL,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Slack alert sent")
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)


def _send_sendgrid_email(subject: str, html_content: str) -> None:
    if not (_SENDGRID_KEY and _ALERT_EMAIL):
        return
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {_SENDGRID_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "personalizations": [{"to": [{"email": _ALERT_EMAIL}]}],
            "from": {"email": "agent@linkedin-intelligence.local"},
            "subject": subject,
            "content": [{"type": "text/html", "value": html_content}],
        }
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Email digest sent to %s", _ALERT_EMAIL)
    except Exception as exc:
        logger.warning("SendGrid email failed: %s", exc)


def _send_webhook(payload: Dict[str, Any]) -> None:
    if not _ALERT_WEBHOOK:
        return
    try:
        import requests
        resp = requests.post(
            _ALERT_WEBHOOK,
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Webhook dispatched to %s", _ALERT_WEBHOOK)
    except Exception as exc:
        logger.warning("Webhook dispatch failed: %s", exc)


# ---------------------------------------------------------------------------
# Email template
# ---------------------------------------------------------------------------

def _build_email_html(data: Dict[str, Any]) -> str:
    topic = data.get("topic", "")
    insights = data.get("insights", {})
    trends = data.get("trends", [])[:5]
    summary = insights.get("summary", "")
    takeaways = insights.get("key_takeaways", [])
    opportunities = insights.get("opportunities", [])

    trend_rows = "".join(
        f"<tr><td style='padding:4px 8px'>{t['label']}</td>"
        f"<td style='padding:4px 8px;text-align:right'>{t['score']:.2f}</td></tr>"
        for t in trends
    )
    takeaway_items = "".join(f"<li>{t}</li>" for t in takeaways)
    opp_items = "".join(f"<li>{o}</li>" for o in opportunities)

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px;margin:auto;color:#222">
      <h2 style="color:#0a66c2">LinkedIn Intelligence Digest: {topic}</h2>
      <p>{summary}</p>
      <h3>Top Trends</h3>
      <table style="border-collapse:collapse;width:100%">
        <thead><tr style="background:#f0f4ff">
          <th style="padding:4px 8px;text-align:left">Trend</th>
          <th style="padding:4px 8px;text-align:right">Score</th>
        </tr></thead>
        <tbody>{trend_rows}</tbody>
      </table>
      <h3>Key Takeaways</h3><ul>{takeaway_items}</ul>
      <h3>Opportunities</h3><ul>{opp_items}</ul>
      <hr/>
      <p style="font-size:11px;color:#888">
        Sources: {data.get('sources_analysed', 0)} |
        Time: {data.get('elapsed_seconds', 0)}s |
        LinkedIn Intelligence Agent
      </p>
    </body></html>
    """


def notifier_status() -> Dict[str, Any]:
    return {
        "slack": "configured" if _SLACK_URL else "disabled",
        "email": "configured" if (_SENDGRID_KEY and _ALERT_EMAIL) else "disabled",
        "webhook": "configured" if _ALERT_WEBHOOK else "disabled",
    }