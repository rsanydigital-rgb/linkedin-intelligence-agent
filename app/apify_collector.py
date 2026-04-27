"""
apify_collector.py
------------------
LinkedIn data via Apify.

Supports:
  - user-selected post count
  - user-selected timeline cutoff
  - fallback between two actors
"""

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests as _http

logger = logging.getLogger(__name__)

_APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")
_ACTOR_PRIMARY = "harvestapi/linkedin-post-search"
_ACTOR_FALLBACK = "supreme_coder/linkedin-post"
_MAX_RESULTS = int(os.getenv("APIFY_MAX_RESULTS_PER_QUERY", "15"))
_APIFY_BASE = "https://api.apify.com/v2"


def is_available() -> bool:
    return bool(_APIFY_TOKEN and _APIFY_TOKEN.startswith("apify_api_"))


def collect_linkedin_posts(
    queries: List[str],
    max_total_results: Optional[int] = None,
    date_since: Optional[datetime] = None,
) -> List[Dict]:
    if not is_available():
        logger.debug("Apify token missing/invalid; LinkedIn skipped")
        return []
    if not queries:
        return []

    total_limit = max_total_results or _MAX_RESULTS
    per_query_limit = max(1, math.ceil(total_limit / len(queries)))
    fetch_limit = min(max(per_query_limit * 3, per_query_limit), 100)

    all_results: List[Dict] = []
    for query in queries:
        logger.info(
            "Apify LinkedIn: '%s' (show up to %d, fetch up to %d)",
            query,
            per_query_limit,
            fetch_limit,
        )
        items: List[Dict] = []
        for actor_id in (_ACTOR_PRIMARY, _ACTOR_FALLBACK):
            try:
                items = _run_actor(query, actor_id, fetch_limit)
                if items:
                    logger.info("Actor '%s': %d items for '%s'", actor_id, len(items), query)
                    break
            except Exception as exc:
                logger.warning("Actor '%s' failed: %s; trying next", actor_id, exc)

        parsed = _parse_items(items, query)
        if date_since is not None:
            parsed = [item for item in parsed if item["collected_at"] >= date_since]
        parsed = parsed[:per_query_limit]
        logger.info("Parsed %d LinkedIn posts for '%s'", len(parsed), query)
        all_results.extend(parsed)

    return all_results[:total_limit]


def _run_actor(query: str, actor_id: str, max_results: int) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {_APIFY_TOKEN}",
        "Content-Type": "application/json",
    }
    slug = actor_id.replace("/", "~")

    if "harvestapi" in actor_id:
        run_input = {
            "searchQueries": [query],
            "maxPosts": max_results,
            "proxy": {"useApifyProxy": True},
        }
    else:
        run_input = {
            "search": query,
            "maxPosts": max_results,
            "proxy": {"useApifyProxy": True},
        }

    resp = _http.post(
        f"{_APIFY_BASE}/acts/{slug}/runs",
        json=run_input,
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 404:
        raise RuntimeError(f"Actor '{actor_id}' not found")
    if resp.status_code == 402:
        raise RuntimeError(f"Payment required for '{actor_id}'")
    resp.raise_for_status()

    run_data = resp.json().get("data", {})
    run_id = run_data.get("id")
    dataset_id = run_data.get("defaultDatasetId")
    if not run_id:
        raise RuntimeError("No run ID returned")

    deadline = time.time() + 150
    while time.time() < deadline:
        poll = _http.get(f"{_APIFY_BASE}/actor-runs/{run_id}", headers=headers, timeout=15)
        poll.raise_for_status()
        data = poll.json().get("data", {})
        status = data.get("status", "")
        if status in ("SUCCEEDED", "FINISHED"):
            dataset_id = data.get("defaultDatasetId", dataset_id)
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Run ended: {status}")
        time.sleep(5)
    else:
        raise RuntimeError("Run timed out after 150 s")

    items_resp = _http.get(
        f"{_APIFY_BASE}/datasets/{dataset_id}/items",
        params={"limit": max_results},
        headers=headers,
        timeout=30,
    )
    items_resp.raise_for_status()
    result = items_resp.json()
    return result if isinstance(result, list) else []


def _parse_items(items: List[Dict], query: str) -> List[Dict]:
    if not items:
        return []

    logger.info("Apify first-item keys for query '%s': %s", query, sorted(items[0].keys()))
    logger.debug("Apify first-item full dump: %s", items[0])

    results = []
    for item in items:
        text = (
            item.get("content")
            or item.get("text")
            or item.get("postText")
            or item.get("commentary")
            or item.get("description")
            or item.get("body")
            or ""
        ).strip()
        if not text or len(text) < 20:
            continue

        url = (
            item.get("linkedinUrl")
            or item.get("url")
            or item.get("postUrl")
            or item.get("link")
            or "https://linkedin.com"
        )
        collected_at = _extract_post_datetime(item) or datetime.now(timezone.utc)
        likes, comments, shares = _extract_engagement(item)
        author = (
            item.get("authorName")
            or (item.get("author") or {}).get("name", "")
            or (item.get("actor") or {}).get("name", "")
            or ""
        )

        results.append(
            {
                "query": query,
                "source": "linkedin",
                "content": text[:2000],
                "url": url,
                "collected_at": collected_at,
                "metadata": {
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "author": author,
                    "hashtags": item.get("hashtags", []),
                },
            }
        )
    return results


def _extract_post_datetime(item: Dict) -> Optional[datetime]:
    candidate_keys = (
        "postedAt",
        "postedAtTimestamp",
        "postDate",
        "postedDate",
        "publishedAt",
        "createdAt",
        "date",
        "timestamp",
    )

    for key in candidate_keys:
        parsed = _parse_datetime_value(item.get(key))
        if parsed:
            return parsed

    for nested_key in ("post", "metadata", "actor"):
        nested = item.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in candidate_keys:
            parsed = _parse_datetime_value(nested.get(key))
            if parsed:
                return parsed

    return None


def _parse_datetime_value(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 1_000_000_000_000:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return _parse_datetime_value(int(raw))
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_engagement(item: Dict) -> tuple:
    likes = comments = shares = 0

    for nested_key in ("socialActivity", "socialCounts", "engagement", "stats", "counts"):
        nested = item.get(nested_key)
        if isinstance(nested, dict) and nested:
            for key, value in nested.items():
                lowered = key.lower()
                try:
                    numeric = int(value) if not isinstance(value, (int, float)) else int(value)
                except (TypeError, ValueError):
                    continue
                if numeric <= 0:
                    continue
                if likes == 0 and any(token in lowered for token in ("like", "reaction")):
                    likes = numeric
                elif comments == 0 and "comment" in lowered:
                    comments = numeric
                elif shares == 0 and any(token in lowered for token in ("repost", "share", "reshare")):
                    shares = numeric

    if likes and comments and shares:
        return likes, comments, shares

    for key, value in item.items():
        lowered = key.lower()
        numeric = 0
        if isinstance(value, (int, float)) and value > 0:
            numeric = int(value)
        elif isinstance(value, str):
            numeric = int(value) if value.strip().isdigit() else 0
        elif isinstance(value, dict) and value:
            try:
                numeric = sum(
                    int(v)
                    for v in value.values()
                    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())
                )
            except (TypeError, ValueError):
                numeric = 0

        if numeric <= 0:
            continue
        if likes == 0 and any(token in lowered for token in ("reaction", "like")):
            likes = numeric
        elif comments == 0 and "comment" in lowered:
            comments = numeric
        elif shares == 0 and any(token in lowered for token in ("repost", "share", "reshare")):
            shares = numeric

    return likes, comments, shares


def apify_status() -> Dict:
    if not is_available():
        return {"status": "disabled", "reason": "APIFY_API_TOKEN missing or invalid"}
    try:
        headers = {"Authorization": f"Bearer {_APIFY_TOKEN}"}
        resp = _http.get(f"{_APIFY_BASE}/users/me", headers=headers, timeout=10)
        resp.raise_for_status()
        user = resp.json().get("data", {})
        return {
            "status": "connected",
            "username": user.get("username"),
            "plan": user.get("plan", {}).get("id", "unknown"),
            "actor": _ACTOR_PRIMARY,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
