"""
query_intelligence.py
---------------------
Heuristic classification of user queries before expansion.
"""

import re
from typing import Dict, List


GENERIC_TOPICS = {
    "ai": ["AI in healthcare", "AI startups", "enterprise AI automation"],
    "artificial intelligence": ["AI in healthcare", "AI startups", "enterprise AI automation"],
    "robotics": ["humanoid robotics", "warehouse robotics", "robotics computer vision"],
    "finance": ["fintech infrastructure", "retail investing platforms", "AI in finance"],
    "healthcare": ["AI diagnostics", "digital therapeutics", "healthcare operations automation"],
    "energy": ["grid storage", "solar software", "industrial electrification"],
}

VAGUE_TERMS = {"innovation", "future", "technology", "tech", "startup", "startups", "market"}


def analyze_query(topic: str) -> Dict[str, object]:
    cleaned = re.sub(r"\s+", " ", topic.strip())
    tokens = re.findall(r"[a-zA-Z0-9\-]+", cleaned.lower())
    token_count = len(tokens)

    query_type = "niche"
    reason = "Specific topic with meaningful modifiers."
    effective_topic = cleaned
    suggestions: List[str] = []

    if token_count <= 1 or cleaned.lower() in GENERIC_TOPICS:
        query_type = "broad"
        reason = "The query is very broad and could span multiple industries or use cases."
        suggestions = GENERIC_TOPICS.get(cleaned.lower(), _generic_suggestions(cleaned))
        effective_topic = suggestions[0] if suggestions else cleaned
    elif any(token in VAGUE_TERMS for token in tokens) and token_count <= 2:
        query_type = "vague"
        reason = "The query uses generic language without enough domain context."
        suggestions = _generic_suggestions(cleaned)
        effective_topic = suggestions[0] if suggestions else cleaned
    elif token_count >= 4:
        query_type = "niche"
        reason = "The query already contains enough context to support focused collection."

    return {
        "original_query": cleaned,
        "query_type": query_type,
        "reason": reason,
        "effective_topic": effective_topic,
        "suggestions": suggestions[:3],
    }


def _generic_suggestions(topic: str) -> List[str]:
    root = topic.strip()
    return [
        f"{root} in healthcare",
        f"{root} startups",
        f"{root} enterprise applications",
    ]
