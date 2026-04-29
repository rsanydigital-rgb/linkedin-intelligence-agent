"""
query_expansion.py
------------------
Expands a raw topic string into 5-8 targeted search queries using an LLM.
Falls back to deterministic rule-based expansion if no API key is available
or the configured LLM endpoint fails.
"""

import json
import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS", "8"))  # reduced from 30s — fail fast to rule-based fallback


def expand_topic(topic: str) -> List[str]:
    """Return 5-8 search queries derived from *topic*."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    orchestrator_model = os.getenv("LLM_ORCHESTRATOR_MODEL", "").strip()

    if openai_key and openai_key not in ("your_openai_key_here", "") and orchestrator_model:
        logger.info(
            "Expanding queries via OpenAI-compatible API using %s (timeout=%ds)",
            orchestrator_model,
            _LLM_TIMEOUT,
        )
        try:
            return _expand_via_openai(topic, openai_key)
        except Exception as exc:
            logger.warning(
                "OpenAI-compatible query expansion failed; using rule-based fallback: %s",
                exc,
            )
            return _expand_rule_based(topic)

    if anthropic_key and anthropic_key != "your_anthropic_key_here":
        logger.info("Expanding queries via Claude API (timeout=%ds)", _LLM_TIMEOUT)
        try:
            return _expand_via_claude(topic, anthropic_key)
        except Exception as exc:
            logger.warning(
                "Claude query expansion failed; using rule-based fallback: %s",
                exc,
            )
            return _expand_rule_based(topic)

    if openai_key and openai_key not in ("your_openai_key_here", ""):
        logger.info("Expanding queries via OpenAI API (timeout=%ds)", _LLM_TIMEOUT)
        try:
            return _expand_via_openai(topic, openai_key)
        except Exception as exc:
            logger.warning(
                "OpenAI query expansion failed; using rule-based fallback: %s",
                exc,
            )
            return _expand_rule_based(topic)

    logger.warning("No LLM key found; using rule-based query expansion")
    return _expand_rule_based(topic)


_SYSTEM_PROMPT = (
    "You are a search-query strategist. Given a topic, produce exactly 4 diverse, "
    "specific search queries that together give broad coverage of that topic: "
    "industry news, technical developments, market trends, and key players. "
    "Return ONLY a JSON array of strings, no extra text."
)


def _expand_via_claude(topic: str, api_key: str) -> List[str]:
    import requests

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 512,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Topic: {topic}"}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    return _parse_json_list(raw, topic)


def _expand_via_openai(topic: str, api_key: str) -> List[str]:
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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
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
    return _parse_json_list(raw, topic)


def _expand_rule_based(topic: str) -> List[str]:
    templates = [
        "{topic} latest trends 2026",
        "{topic} industry news and developments",
        "{topic} market growth and key players",
        "{topic} technology innovations",
    ]
    return [template.format(topic=topic) for template in templates]


def _parse_json_list(raw: str, topic: str) -> List[str]:
    """Best-effort extraction of a JSON string list from LLM output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(query, str) for query in queries):
            return queries[:8]
    except json.JSONDecodeError:
        pass

    found = re.findall(r'"([^"]{5,})"', raw)
    if found:
        return found[:8]

    logger.error("Could not parse LLM query list; using rule-based fallback")
    return _expand_rule_based(topic)