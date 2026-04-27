"""
validation.py
-------------
Strict validation layer for collected records before downstream processing.

Example usage:
    from app.validation import validate_collected_posts

    validated_posts = validate_collected_posts(raw_results)
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CollectedPost(BaseModel):
    source: str
    content: str
    url: str
    collected_at: datetime
    content_hash: str
    engagement_score: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # likes/comments/shares/author for LinkedIn


def validate_collected_posts(records: Iterable[Any]) -> List[CollectedPost]:
    """Validate collected records and deduplicate by content/url hash."""
    validated: List[CollectedPost] = []
    seen_hashes: set[str] = set()

    for record in records:
        data = _normalise_record(record)
        content = str(data.get("content", "")).strip()
        url = str(data.get("url", "")).strip()

        if not content:
            logger.warning(
                "Rejected record with empty content: source=%s url=%s",
                data.get("source", ""),
                url or "<missing>",
            )
            continue

        if not url:
            logger.warning(
                "Rejected record with missing URL: source=%s",
                data.get("source", ""),
            )
            continue

        content_hash = hashlib.sha256(f"{content}{url}".encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            logger.info(
                "Dropped duplicate record: source=%s url=%s",
                data.get("source", ""),
                url,
            )
            continue

        seen_hashes.add(content_hash)
        validated.append(
            CollectedPost(
                source=str(data.get("source", "")).strip(),
                content=content,
                url=url,
                collected_at=data["collected_at"],
                content_hash=content_hash,
                engagement_score=float(data.get("engagement_score", 0) or 0),
                metadata=data.get("metadata", {}),
            )
        )

    return validated


def _normalise_record(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        data = dict(record)
    elif hasattr(record, "model_dump"):
        data = record.model_dump()
    else:
        data = {
            "source": getattr(record, "source", ""),
            "content": getattr(record, "content", ""),
            "url": getattr(record, "url", ""),
            "collected_at": getattr(record, "collected_at", None),
            "engagement_score": getattr(record, "engagement_score", 0),
        }

    if not data.get("collected_at"):
        data["collected_at"] = datetime.utcnow()

    # Derive a baseline engagement_score from LinkedIn metadata when not already set.
    # This ensures posts with real traction carry a non-zero score into processor.py
    # even before the TF-IDF heuristic runs.
    if not data.get("engagement_score"):
        meta = data.get("metadata", {})
        real_eng = (
            (meta.get("likes") or 0)
            + (meta.get("comments") or 0)
            + (meta.get("shares") or 0)
        )
        if real_eng > 0:
            # Soft-cap at 100: 1 000 total interactions → score 100
            data["engagement_score"] = min(real_eng / 10.0, 100.0)

    return data