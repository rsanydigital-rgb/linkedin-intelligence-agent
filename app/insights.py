"""
insights.py
-----------
Generates strategic insights from structured JSON produced by processor.py.

If the configured LLM provider fails, the pipeline falls back to a rule-based
insight generator instead of failing the whole analysis request.
"""

import json
import logging
import os
import re
from typing import Any, Dict

from app.processor import ProcessedData

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))


def generate_insights(topic: str, data: ProcessedData) -> Dict[str, Any]:
    """Return insight dict from structured pipeline data."""
    structured_input = _build_structured_input(topic, data)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    orchestrator_model = os.getenv("LLM_ORCHESTRATOR_MODEL", "").strip()

    if openai_key and openai_key not in ("your_openai_key_here", "") and orchestrator_model:
        logger.info(
            "Generating insights via OpenAI-compatible API using %s (timeout=%ds)",
            orchestrator_model,
            _LLM_TIMEOUT,
        )
        try:
            return _insights_via_openai(structured_input, openai_key)
        except Exception as exc:
            logger.warning(
                "OpenAI-compatible insight generation failed; using rule-based fallback: %s",
                exc,
            )
            return _rule_based_insights(topic, data)

    if anthropic_key and anthropic_key != "your_anthropic_key_here":
        logger.info("Generating insights via Claude API (timeout=%ds)", _LLM_TIMEOUT)
        try:
            return _insights_via_claude(structured_input, anthropic_key)
        except Exception as exc:
            logger.warning(
                "Claude insight generation failed; using rule-based fallback: %s",
                exc,
            )
            return _rule_based_insights(topic, data)

    if openai_key and openai_key not in ("your_openai_key_here", ""):
        logger.info("Generating insights via OpenAI API (timeout=%ds)", _LLM_TIMEOUT)
        try:
            return _insights_via_openai(structured_input, openai_key)
        except Exception as exc:
            logger.warning(
                "OpenAI insight generation failed; using rule-based fallback: %s",
                exc,
            )
            return _rule_based_insights(topic, data)

    logger.warning("No LLM key configured; generating rule-based insights")
    return _rule_based_insights(topic, data)


def _build_structured_input(topic: str, data: ProcessedData) -> Dict[str, Any]:
    """Distil pipeline output into a compact JSON object for the LLM."""
    return {
        "topic": topic,
        "total_sources": len(data.documents),
        "top_keywords": data.top_keywords[:8],
        "trends": [
            {
                "label": trend["label"],
                "keywords": trend["keywords"][:4],
                "score": round(trend["score"], 3),
            }
            for trend in data.trends[:6]
        ],
        "top_docs": sorted(
            [
                {
                    "source": document["source"],
                    "engagement": document["engagement_score"],
                    "preview": document["content_preview"][:100],
                }
                for document in data.documents
            ],
            key=lambda item: item["engagement"],
            reverse=True,
        )[:3],
    }


_SYSTEM = (
    "You are an industry analyst. "
    "You receive ONLY structured JSON data about trend clusters, scores, keywords, engagement signals, "
    "and top documents. Based solely on this input, identify the strongest trends, explain why they are trending, "
    "highlight business opportunities, and cite the supporting signals from the data. "
    "Do NOT generalize. Base every claim on the input. Be specific and analytical. "
    "Reference concrete evidence such as cluster labels, keyword groupings, relative scores, repeated themes, "
    "and high-engagement document patterns whenever possible. "
    "Respond ONLY with a valid JSON object matching this schema exactly:\n"
    '{\n'
    '  "summary": "<2-3 sentence executive summary grounded in the data>",\n'
    '  "key_takeaways": ["<specific trend insight with supporting signal>", "<specific trend insight with supporting signal>", "<specific trend insight with supporting signal>", "<specific trend insight with supporting signal>"],\n'
    '  "opportunities": ["<specific business opportunity linked to the data>", "<specific business opportunity linked to the data>", "<specific business opportunity linked to the data>"],\n'
    '  "confidence": <float between 0 and 1>\n'
    '}\n'
    "No markdown, no preamble, no extra keys."
)


def _user_prompt(structured_input: Dict[str, Any]) -> str:
    return (
        "Analyse this intelligence data and return the JSON insight object. "
        "Focus on strongest trends, why they are rising, business opportunities, and supporting signals from the data:\n\n"
        f"{json.dumps(structured_input, indent=2)}"
    )


def _insights_via_claude(structured_input: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    import requests

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1024,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": _user_prompt(structured_input)}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    return _parse_insight_json(raw, structured_input["topic"])


def _insights_via_openai(structured_input: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    import requests

    model = os.getenv("LLM_ORCHESTRATOR_MODEL", DEFAULT_OPENAI_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_prompt(structured_input)},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    logger.debug("POSTing to %s/chat/completions with model=%s timeout=%d", base_url, model, _LLM_TIMEOUT)

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_insight_json(raw, structured_input["topic"])


def _rule_based_insights(topic: str, data: ProcessedData) -> Dict[str, Any]:
    top_trends = data.trends[:3]
    top_docs = sorted(data.documents, key=lambda doc: doc["engagement_score"], reverse=True)[:3]

    if top_trends:
        trend_summary = ", ".join(
            f"{trend['label']} ({trend['score']:.2f})" for trend in top_trends
        )
        summary = (
            f"Analysis of {len(data.documents)} filtered sources for '{topic}' shows the strongest trend clusters "
            f"around {trend_summary}. High-engagement documents reinforce these clusters through repeated keyword overlap "
            f"and concentrated scoring."
        )
    else:
        summary = (
            f"Analysis of {len(data.documents)} filtered sources for '{topic}' found limited cluster separation, "
            f"with only sparse supporting signals in the structured data."
        )

    takeaways = []
    for trend in top_trends[:4]:
        supporting_doc = next(
            (
                doc for doc in top_docs
                if set(trend["keywords"]).intersection(set(doc["keywords"]))
            ),
            None,
        )
        if supporting_doc:
            takeaways.append(
                f"'{trend['label']}' is a leading trend because its cluster score is {trend['score']:.2f} and it overlaps with high-engagement content scored at {supporting_doc['engagement_score']:.2f}, especially around {', '.join(supporting_doc['keywords'][:3])}."
            )
        else:
            takeaways.append(
                f"'{trend['label']}' stands out through a cluster score of {trend['score']:.2f} and repeated keywords such as {', '.join(trend['keywords'][:3])}."
            )

    while len(takeaways) < 4:
        takeaways.append("Structured signals are limited beyond the highest-scoring clusters.")

    opportunities = []
    for trend in top_trends[:3]:
        opportunities.append(
            f"A focused offering or content strategy around '{trend['label']}' is supported by its cluster score of {trend['score']:.2f} and recurring keywords like {', '.join(trend['keywords'][:3])}."
        )

    while len(opportunities) < 3:
        opportunities.append("Additional data collection is needed before identifying further opportunities.")

    confidence = _compute_confidence(data)

    return {
        "summary": summary,
        "key_takeaways": takeaways[:4],
        "opportunities": opportunities[:3],
        "confidence": confidence,
    }


def _parse_insight_json(raw: str, topic: str) -> Dict[str, Any]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(raw)
        required = {"summary", "key_takeaways", "opportunities"}
        if required.issubset(parsed.keys()):
            return {
                "summary": str(parsed["summary"]),
                "key_takeaways": list(parsed["key_takeaways"]),
                "opportunities": list(parsed["opportunities"]),
                "confidence": float(parsed.get("confidence", 0.0)),
            }
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse LLM insight JSON: %s", exc)

    return {
        "summary": f"Intelligence analysis complete for topic: {topic}.",
        "key_takeaways": ["Keyword trends extracted successfully.", "Multiple sources analysed."],
        "opportunities": ["Further domain-specific research recommended."],
        "confidence": 0.4,
    }


def _compute_confidence(data: ProcessedData) -> float:
    source_factor = min(len(data.documents) / 12.0, 1.0)

    keyword_sets = [set(doc.get("keywords", [])) for doc in data.documents if doc.get("keywords")]
    overlaps = []
    for index in range(len(keyword_sets)):
        for inner_index in range(index + 1, len(keyword_sets)):
            union = keyword_sets[index] | keyword_sets[inner_index]
            if union:
                overlaps.append(len(keyword_sets[index] & keyword_sets[inner_index]) / len(union))
    consistency_factor = sum(overlaps) / len(overlaps) if overlaps else 0.0

    trend_scores = [float(trend.get("score", 0.0)) for trend in data.trends]
    trend_factor = min((sum(trend_scores[:3]) / 9.0), 1.0) if trend_scores else 0.0

    confidence = 0.35 * source_factor + 0.3 * consistency_factor + 0.35 * trend_factor
    return round(min(max(confidence, 0.0), 1.0), 2)