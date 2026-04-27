"""
collector.py
------------
Fetches web results for a list of search queries with source prioritization.

Strategy:
  1. Fetch from primary source first (web search / SerpAPI)
  2. If primary results are below the threshold, fetch from secondary source (NewsAPI)
  3. If neither source is configured or both fail, use deterministic mock data

Results are validated through the RawResult Pydantic schema and deduplicated.
"""

import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Tuple, Optional

import requests
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

SECONDARY_FETCH_THRESHOLD = int(os.getenv("SECONDARY_FETCH_THRESHOLD", "10"))


class RawResult(BaseModel):
    model_config = {"extra": "ignore"}
    query: str
    source: str
    content: str
    url: str
    collected_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)  # preserves LinkedIn likes/comments/shares/author

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()

    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


def collect(
    queries: List[str],
    sources: str = "web_news",
    linkedin_post_count: Optional[int] = None,
    linkedin_date_since: Optional[datetime] = None,
) -> List[RawResult]:
    """Collect results for all queries and return deduplicated RawResult list."""
    results, _ = collect_with_breakdown(
        queries,
        sources=sources,
        linkedin_post_count=linkedin_post_count,
        linkedin_date_since=linkedin_date_since,
    )
    return results


def collect_with_breakdown(
    queries: List[str],
    sources: str = "web_news",
    linkedin_post_count: Optional[int] = None,
    linkedin_date_since: Optional[datetime] = None,
) -> Tuple[List[RawResult], Dict[str, List[RawResult]]]:
    """Collect prioritized results and preserve a validated per-query breakdown."""
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    news_api_key = os.getenv("NEWS_API_KEY", "")
    include_linkedin = "linkedin" in sources

    raw_items: List[Dict] = []
    query_groups: Dict[str, List[Dict]] = {query: [] for query in queries}

    for query in queries:
        items = _collect_for_query(query, serpapi_key, news_api_key)
        query_groups[query].extend(items)
        raw_items.extend(items)

    # LinkedIn / Apify collection — runs when user selects "+ LinkedIn" source
    if include_linkedin:
        try:
            from app.apify_collector import collect_linkedin_posts
            li_items = collect_linkedin_posts(
                queries,
                max_total_results=linkedin_post_count,
                date_since=linkedin_date_since,
            )
            logger.info("Apify LinkedIn returned %d total items", len(li_items))
            for item in li_items:
                q = item.get("query", queries[0] if queries else "")
                # Exact match first; fall back to the query whose string is
                # contained in (or contains) the item's query tag — handles
                # cases where the expanded query differs slightly from the
                # stored tag.
                if q in query_groups:
                    query_groups[q].append(item)
                else:
                    # Fuzzy fallback: find the closest matching expanded query
                    matched = next(
                        (eq for eq in queries if q in eq or eq in q),
                        queries[0] if queries else None,
                    )
                    if matched:
                        query_groups[matched].append(item)
                        item["query"] = matched  # normalise so downstream stays consistent
                raw_items.append(item)
        except Exception as exc:
            logger.warning("LinkedIn collection failed (non-fatal): %s", exc)

    validated_results = _validate_and_deduplicate(raw_items)
    validated_groups = {
        query: _validate_and_deduplicate(items) for query, items in query_groups.items()
    }
    return validated_results, validated_groups


def _collect_for_query(query: str, serpapi_key: str, news_api_key: str) -> List[Dict]:
    primary_items = fetch_primary(query, serpapi_key)
    combined_items = list(primary_items)

    if len(primary_items) < SECONDARY_FETCH_THRESHOLD:
        secondary_items = fetch_secondary(query, news_api_key)
        combined_items.extend(secondary_items)

    if combined_items:
        return combined_items

    return _fetch_mock(query)


def fetch_primary(query: str, api_key: str) -> List[Dict]:
    """Primary source: web search via SerpAPI."""
    if not api_key or api_key == "your_serpapi_key_here":
        return []

    try:
        return _fetch_serpapi(query, api_key)
    except Exception as exc:
        logger.warning("Primary source failed for query '%s': %s", query, exc)
        return []


def fetch_secondary(query: str, api_key: str) -> List[Dict]:
    """Secondary source: NewsAPI, used only when primary is insufficient."""
    if not api_key or api_key == "your_newsapi_key_here":
        return []

    try:
        return _fetch_newsapi(query, api_key)
    except Exception as exc:
        logger.warning("Secondary source failed for query '%s': %s", query, exc)
        return []


def _fetch_serpapi(query: str, api_key: str) -> List[Dict]:
    params = {
        "q": query,
        "api_key": api_key,
        "num": 5,
        "hl": "en",
    }
    resp = requests.get("https://serpapi.com/search", params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    items = []
    for r in data.get("organic_results", [])[:5]:
        items.append(
            {
                "query": query,
                "source": "serpapi",
                "content": f"{r.get('title', '')}. {r.get('snippet', '')}",
                "url": r.get("link", "https://example.com"),
                "collected_at": datetime.now(timezone.utc),
            }
        )
    return items


def _fetch_newsapi(query: str, api_key: str) -> List[Dict]:
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": 5,
        "language": "en",
        "sortBy": "publishedAt",
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    items = []
    for article in data.get("articles", [])[:5]:
        content = (article.get("description") or article.get("title") or "").strip()
        if not content:
            continue
        items.append(
            {
                "query": query,
                "source": "newsapi",
                "content": f"{article.get('title', '')}. {content}",
                "url": article.get("url", "https://example.com"),
                "collected_at": datetime.now(timezone.utc),
            }
        )
    return items


_MOCK_BANK = [
    {
        "title": "Robotics Industry Sees Record Investment in 2024",
        "snippet": (
            "The global robotics market reached $62 billion in 2024, driven by automation "
            "adoption across manufacturing, logistics, and healthcare sectors. AI integration "
            "has accelerated deployment of collaborative robots (cobots) in SMEs."
        ),
        "url": "https://example.com/robotics-investment-2024",
    },
    {
        "title": "Boston Dynamics and Agility Robotics Lead Humanoid Race",
        "snippet": (
            "Humanoid robots are exiting the lab and entering warehouses. Boston Dynamics "
            "Atlas and Agility Robotics Digit are being piloted at Amazon fulfillment centers "
            "for pick-and-place tasks, with reported 40% throughput improvements."
        ),
        "url": "https://example.com/humanoid-robots-warehouses",
    },
    {
        "title": "AI-Powered Computer Vision Transforms Quality Control",
        "snippet": (
            "Machine vision systems using deep learning now achieve 99.7% defect detection "
            "accuracy in semiconductor fabrication, outperforming human inspectors by 15x. "
            "Adoption is surging in automotive and electronics manufacturing."
        ),
        "url": "https://example.com/ai-computer-vision-qc",
    },
    {
        "title": "Autonomous Mobile Robots Reshape Warehouse Logistics",
        "snippet": (
            "AMR deployments grew 120% YoY in 2024. Key players Locus Robotics, Fetch, and "
            "Geek+ are competing on payload capacity and multi-robot orchestration. "
            "Energy efficiency and charging autonomy are primary differentiators."
        ),
        "url": "https://example.com/amr-warehouse-logistics",
    },
    {
        "title": "Surgical Robotics Market Projected to Reach $24B by 2030",
        "snippet": (
            "Intuitive Surgical's da Vinci platform continues to dominate with 8,000+ installed "
            "units globally. Competitors Medtronic Hugo and CMR Surgical Versius are gaining "
            "traction with more affordable, portable systems targeting emerging markets."
        ),
        "url": "https://example.com/surgical-robotics-market",
    },
    {
        "title": "Edge AI Chips Enable Real-Time Robot Perception",
        "snippet": (
            "NVIDIA Jetson Orin and Qualcomm Robotics RB6 are powering next-gen robotic "
            "perception stacks. On-device inference enables sub-10ms latency for obstacle "
            "avoidance and manipulation planning without cloud dependency."
        ),
        "url": "https://example.com/edge-ai-robotics",
    },
    {
        "title": "Labor Shortages Accelerate Agricultural Robotics Adoption",
        "snippet": (
            "Fruit harvesting robots from Abundant Robotics and Tortuga AgTech are addressing "
            "seasonal labor gaps. Soft-gripper technology breakthroughs allow gentle handling "
            "of delicate produce with 95% pick success rates."
        ),
        "url": "https://example.com/agricultural-robotics",
    },
    {
        "title": "ROS 2 Ecosystem Matures: Enterprise-Grade Middleware for Robotics",
        "snippet": (
            "ROS 2 adoption in production environments has grown 300% since 2022. Real-time "
            "capabilities, improved security, and DDS middleware support make it viable for "
            "safety-critical applications in medical and automotive robotics."
        ),
        "url": "https://example.com/ros2-enterprise",
    },
    {
        "title": "China Emerges as Dominant Force in Industrial Robotics",
        "snippet": (
            "China installed 290,000 industrial robots in 2023, representing 70% of global "
            "deployments. Domestic brands SIASUN and Estun are challenging ABB and FANUC "
            "on price with increasingly capable 6-axis arms."
        ),
        "url": "https://example.com/china-industrial-robotics",
    },
    {
        "title": "Generative AI Enters Robot Programming: Natural Language to Motion",
        "snippet": (
            "Startups like Covariant and Physical Intelligence are using foundation models to "
            "enable robots to learn new tasks from natural language instructions and a handful "
            "of demonstrations, collapsing programming time from weeks to hours."
        ),
        "url": "https://example.com/gen-ai-robot-programming",
    },
    {
        "title": "Drone Delivery Networks Expand in Urban Corridors",
        "snippet": (
            "Wing (Alphabet) and Zipline have completed over 1 million commercial deliveries. "
            "Regulatory approvals for BVLOS operations are unlocking scalable last-mile "
            "delivery economics, with costs approaching $3 per delivery at scale."
        ),
        "url": "https://example.com/drone-delivery-expansion",
    },
    {
        "title": "Soft Robotics Enables New Applications in Food Processing",
        "snippet": (
            "Soft actuator technology allows robots to handle irregular, fragile items that "
            "were previously off-limits to automation. Soft Robotics Inc. reports 40+ food "
            "industry deployments across bakery, meat, and produce segments."
        ),
        "url": "https://example.com/soft-robotics-food",
    },
]


def _fetch_mock(query: str) -> List[Dict]:
    idx = int(hashlib.md5(query.encode()).hexdigest(), 16) % len(_MOCK_BANK)
    selected = [_MOCK_BANK[(idx + i) % len(_MOCK_BANK)] for i in range(4)]
    return [
        {
            "query": query,
            "source": "mock",
            "content": f"{item['title']}. {item['snippet']}",
            "url": item["url"],
            "collected_at": datetime.now(timezone.utc),
        }
        for item in selected
    ]


def _validate_and_deduplicate(raw_items: List[Dict]) -> List[RawResult]:
    seen_hashes: Set[str] = set()
    results: List[RawResult] = []

    for item in raw_items:
        try:
            result = RawResult(**item)
        except Exception as exc:
            logger.debug("Skipping invalid item: %s", exc)
            continue

        h = result.content_hash()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        results.append(result)

    logger.info("Collected %d unique results after deduplication", len(results))
    return results
