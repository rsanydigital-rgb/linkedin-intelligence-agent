"""
opportunities.py
----------------
Deterministic opportunity scoring from trend data.
"""

from typing import Dict, List, Any


def score_opportunities(
    trends: List[Dict[str, Any]],
    trend_evolution: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    opportunities = []
    max_raw = 0.0

    evolution_lookup = {item["trend"]: item for item in trend_evolution}

    for trend in trends:
        evolution = evolution_lookup.get(trend["label"], {})
        growth_factor = _growth_factor(evolution.get("change", "+0%"), evolution.get("status", "stable"))
        uniqueness = _uniqueness_factor(trend, trends)
        raw_score = float(trend["score"]) * growth_factor * uniqueness
        max_raw = max(max_raw, raw_score)

        opportunities.append(
            {
                "trend": trend["label"],
                "_raw_score": raw_score,
                "growth_factor": growth_factor,
                "uniqueness": uniqueness,
                "reason": _reason(growth_factor, uniqueness),
            }
        )

    for item in opportunities:
        normalized = (item["_raw_score"] / max_raw * 10.0) if max_raw > 0 else 0.0
        item["opportunity_score"] = round(normalized, 2)
        del item["_raw_score"]
        del item["growth_factor"]
        del item["uniqueness"]

    opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return opportunities


def _growth_factor(change: str, status: str) -> float:
    try:
        pct = float(change.replace("%", ""))
    except ValueError:
        pct = 0.0

    if status == "new":
        return 1.35
    if status == "rising":
        return 1.0 + max(pct, 0.0) / 100.0
    if status == "falling":
        return max(0.5, 1.0 + pct / 100.0)
    return 1.0


def _uniqueness_factor(trend: Dict[str, Any], trends: List[Dict[str, Any]]) -> float:
    keywords = set(trend.get("keywords", []))
    if not keywords:
        return 1.0

    overlaps = []
    for other in trends:
        if other["label"] == trend["label"]:
            continue
        other_keywords = set(other.get("keywords", []))
        union = keywords | other_keywords
        overlap = len(keywords & other_keywords) / len(union) if union else 0.0
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return max(0.6, 1.4 - avg_overlap)


def _reason(growth_factor: float, uniqueness: float) -> str:
    if growth_factor >= 1.3 and uniqueness >= 1.1:
        return "high growth + low competition"
    if growth_factor >= 1.2:
        return "strong growth signal"
    if uniqueness >= 1.15:
        return "distinct cluster with limited overlap"
    return "solid trend strength with moderate competition"
