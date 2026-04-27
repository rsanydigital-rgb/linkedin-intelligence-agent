"""
trend_history.py
----------------
Persistence and comparison utilities for trend evolution across runs.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any


HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "trend_history.json"


def load_previous_trends(topic: str) -> List[Dict[str, Any]]:
    history = _load_history()
    topic_key = _topic_key(topic)
    runs = history.get(topic_key, [])
    if not runs:
        return []
    return runs[-1].get("trends", [])


def save_trends(topic: str, trends: List[Dict[str, Any]]) -> None:
    history = _load_history()
    topic_key = _topic_key(topic)
    history.setdefault(topic_key, []).append(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "trends": trends,
        }
    )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def compute_trend_evolution(
    current_trends: List[Dict[str, Any]],
    previous_trends: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evolution = []
    for current in current_trends:
        previous = _best_previous_match(current, previous_trends)
        current_score = float(current.get("score", 0))
        previous_score = float(previous.get("score", 0)) if previous else 0.0

        if previous_score > 0:
            pct_change = ((current_score - previous_score) / previous_score) * 100.0
        elif current_score > 0:
            pct_change = 100.0
        else:
            pct_change = 0.0

        if previous is None:
            status = "new"
        elif pct_change > 10:
            status = "rising"
        elif pct_change < -10:
            status = "falling"
        else:
            status = "stable"

        evolution.append(
            {
                "trend": current["label"],
                "previous_score": round(previous_score, 2),
                "current_score": round(current_score, 2),
                "change": f"{pct_change:+.0f}%",
                "status": status,
            }
        )

    return evolution


def _best_previous_match(
    current: Dict[str, Any],
    previous_trends: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    current_keywords = set(current.get("keywords", []))
    best = None
    best_overlap = 0.0
    for previous in previous_trends:
        prev_keywords = set(previous.get("keywords", []))
        union = current_keywords | prev_keywords
        if not union:
            continue
        overlap = len(current_keywords & prev_keywords) / len(union)
        if overlap > best_overlap:
            best_overlap = overlap
            best = previous
    return best if best_overlap >= 0.2 else None


def _load_history() -> Dict[str, List[Dict[str, Any]]]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _topic_key(topic: str) -> str:
    return " ".join(topic.lower().split())
