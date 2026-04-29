"""
langgraph_orchestrator.py
--------------------------
LangGraph-based intelligence pipeline replacing the linear analyze() flow.

Graph topology (5 nodes):

  [query_node] → [collect_node] → [route_node] ──→ [enrich_node] → [finalize_node]
                                        │
                                        └──(too few results)──→ [collect_node] (retry once)
                                        └──(no linkedin token)──→ [finalize_node] (skip enrich)

Decisions made by the router:
  - RETRY:   fewer than MIN_DOCS documents after validation → expand queries and re-collect
  - SKIP:    Apify token missing → skip LinkedIn enrichment, go straight to finalize
  - ENRICH:  Apify available → run LinkedIn enrichment pass before finalize
  - FINALIZE: already enriched once or results good → finalize

State schema (PipelineState TypedDict):
  topic, sources, queries, query_analysis,
  raw_results, query_groups, validated_results, validated_query_groups,
  processed, insights, trend_evolution, opportunity_scores,
  retry_count, enriched, error
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

MIN_DOCS          = 3    # Minimum validated docs before we consider retrying
MAX_RETRIES       = 1    # Only retry collection once
RETRY_SUFFIX      = " latest news developments market"  # appended to queries on retry


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    # Inputs
    topic:    str
    sources:  str
    t0:       float  # wall-clock start for elapsed_seconds
    linkedin_post_count: int
    linkedin_date_since: Optional[datetime]
    linkedin_region: Optional[str]

    # Intermediate
    query_analysis:           Optional[Dict[str, Any]]
    queries:                  Optional[List[str]]
    raw_results:              Optional[List[Any]]
    query_groups:             Optional[Dict[str, List[Any]]]
    validated_results:        Optional[List[Any]]
    validated_query_groups:   Optional[Dict[str, List[Any]]]
    processed:                Optional[Any]   # ProcessedData

    # Decisions
    retry_count:  int
    enriched:     bool

    # Outputs
    insights:              Optional[Dict[str, Any]]
    trend_evolution:       Optional[List[Any]]
    opportunity_scores:    Optional[List[Any]]

    # Error surface
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node 1 — Query intelligence + expansion
# ---------------------------------------------------------------------------

def query_node(state: PipelineState) -> PipelineState:
    """Classify the topic and generate search queries."""
    from app.query_intelligence import analyze_query
    from app.query_expansion import expand_topic

    topic = state["topic"]
    logger.info("[LangGraph] query_node: topic='%s'", topic)

    try:
        query_analysis = analyze_query(topic)
        effective_topic = str(query_analysis["effective_topic"])
        queries = expand_topic(effective_topic)
        logger.info("[LangGraph] query_node: %d queries generated", len(queries))
        return {**state, "query_analysis": query_analysis, "queries": queries, "error": None}
    except Exception as exc:
        logger.error("[LangGraph] query_node failed: %s", exc)
        return {**state, "error": f"Query expansion failed: {exc}"}


# ---------------------------------------------------------------------------
# Node 2 — Data collection
# ---------------------------------------------------------------------------

def collect_node(state: PipelineState) -> PipelineState:
    """Collect raw results from SerpAPI / NewsAPI / LinkedIn (Apify)."""
    from app.collector import collect_with_breakdown
    from app.validation import validate_collected_posts

    if state.get("error"):
        return state

    queries  = state["queries"] or []
    sources  = state["sources"]
    retry    = state.get("retry_count", 0)

    # On retry: broaden queries slightly
    if retry > 0:
        queries = [q + RETRY_SUFFIX for q in queries[:3]]
        logger.info("[LangGraph] collect_node RETRY %d: broadened queries", retry)

    logger.info("[LangGraph] collect_node: %d queries, sources=%s", len(queries), sources)

    try:
        raw_results, query_groups = collect_with_breakdown(
            queries,
            sources=sources.replace("_linkedin", ""),
        )
        validated_results = validate_collected_posts(raw_results)
        validated_query_groups = {
            q: validate_collected_posts(items) for q, items in query_groups.items()
        }
        logger.info("[LangGraph] collect_node: %d raw → %d validated",
                    len(raw_results), len(validated_results))
        return {
            **state,
            "queries": queries,
            "raw_results": raw_results,
            "query_groups": query_groups,
            "validated_results": validated_results,
            "validated_query_groups": validated_query_groups,
            "error": None,
        }
    except Exception as exc:
        logger.error("[LangGraph] collect_node failed: %s", exc)
        return {**state, "error": f"Collection failed: {exc}"}


# ---------------------------------------------------------------------------
# Node 3 — Router (decision node — returns next node name as string)
# ---------------------------------------------------------------------------

def route_node(state: PipelineState) -> str:
    """
    Decide what to do next based on current state.
    Returns the name of the next node as a string (used as conditional edge).
    """
    if state.get("error"):
        logger.warning("[LangGraph] route_node: error present → finalize")
        return "finalize_node"

    validated = state.get("validated_results") or []
    retry     = state.get("retry_count", 0)
    enriched  = state.get("enriched", False)

    # Too few docs and haven't retried yet → retry collection
    if len(validated) < MIN_DOCS and retry < MAX_RETRIES:
        logger.info("[LangGraph] route_node: only %d docs, retry_count=%d → RETRY collect",
                    len(validated), retry)
        return "collect_node"  # back to collect with broadened queries

    # Check if LinkedIn enrichment is available and not yet done
    try:
        from app.apify_collector import is_available as apify_available
        linkedin_in_sources = "linkedin" in (state.get("sources") or "")
        if apify_available() and linkedin_in_sources and not enriched:
            logger.info("[LangGraph] route_node: Apify available → enrich_node")
            return "enrich_node"
    except ImportError:
        pass

    logger.info("[LangGraph] route_node: %d docs → finalize_node", len(validated))
    return "finalize_node"


# ---------------------------------------------------------------------------
# Node 4 — LinkedIn enrichment (optional, only when Apify available)
# ---------------------------------------------------------------------------

def enrich_node(state: PipelineState) -> PipelineState:
    """
    Fetch additional LinkedIn posts for the top queries and merge into results.
    This runs after the initial collection so LinkedIn posts supplement — not replace — web results.
    """
    from app.apify_collector import collect_linkedin_posts
    from app.validation import validate_collected_posts

    if state.get("error"):
        return {**state, "enriched": True}

    queries         = state.get("queries") or []
    validated       = list(state.get("validated_results") or [])
    val_groups      = dict(state.get("validated_query_groups") or {})

    # Only fetch top 3 queries for LinkedIn to stay within free-tier limits
    top_queries = queries[:3]
    logger.info("[LangGraph] enrich_node: fetching LinkedIn for %d queries", len(top_queries))

    try:
        li_items = collect_linkedin_posts(
            top_queries,
            max_total_results=state.get("linkedin_post_count"),
            date_since=state.get("linkedin_date_since"),
            region=state.get("linkedin_region"),
        )
        logger.info("[LangGraph] enrich_node: got %d LinkedIn items", len(li_items))

        if li_items:
            # Group by query
            li_by_query: Dict[str, List[Any]] = {}
            for item in li_items:
                q = item.get("query", top_queries[0])
                li_by_query.setdefault(q, []).append(item)

            # Validate and merge
            extra_validated = validate_collected_posts(li_items)
            validated.extend(extra_validated)

            for q, items in li_by_query.items():
                val_groups.setdefault(q, [])
                val_groups[q].extend(validate_collected_posts(items))

            logger.info("[LangGraph] enrich_node: total validated after enrichment=%d", len(validated))

    except Exception as exc:
        logger.warning("[LangGraph] enrich_node: LinkedIn enrichment failed (non-fatal): %s", exc)

    return {
        **state,
        "validated_results": validated,
        "validated_query_groups": val_groups,
        "enriched": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node 5 — Finalize: process → insights → trend history → opportunities
# ---------------------------------------------------------------------------

def finalize_node(state: PipelineState) -> PipelineState:
    """Run the full processing, insight generation, and opportunity scoring."""
    from app.processor import process
    from app.insights import generate_insights
    from app.trend_history import load_previous_trends, save_trends, compute_trend_evolution
    from app.opportunities import score_opportunities

    if state.get("error"):
        return state

    topic             = state["topic"]
    validated_results = state.get("validated_results") or []

    if not validated_results:
        return {**state, "error": "No valid data collected for this topic."}

    # Processing
    logger.info("[LangGraph] finalize_node: processing %d documents", len(validated_results))
    try:
        processed = process(validated_results, topic=topic)
    except Exception as exc:
        return {**state, "error": f"Processing failed: {exc}"}

    if not processed.documents:
        return {**state, "error": "No posts met the engagement threshold for trend detection."}

    # Run insights generation and trend history in parallel
    insights = None
    trend_evolution = []
    opportunity_scores = []

    def _run_insights():
        return generate_insights(topic, processed)

    def _run_trend_history():
        prev = load_previous_trends(topic)
        save_trends(topic, processed.trends)
        evolution = compute_trend_evolution(processed.trends, prev)
        opportunities = score_opportunities(processed.trends, evolution)
        return evolution, opportunities

    with ThreadPoolExecutor(max_workers=2) as executor:
        insights_future = executor.submit(_run_insights)
        trend_future    = executor.submit(_run_trend_history)

        try:
            insights = insights_future.result(timeout=35)
        except Exception as exc:
            logger.warning("[LangGraph] finalize_node: insight generation failed: %s", exc)
            insights = {
                "summary": f"Analysis complete for '{topic}'.",
                "key_takeaways": ["Data collected successfully."],
                "opportunities": [],
                "confidence": 0.4,
            }

        try:
            trend_evolution, opportunity_scores = trend_future.result(timeout=10)
        except Exception as exc:
            logger.warning("[LangGraph] finalize_node: trend history failed: %s", exc)
            trend_evolution    = []
            opportunity_scores = []

    logger.info("[LangGraph] finalize_node: done — %d trends, %d opportunities",
                len(processed.trends), len(opportunity_scores))

    return {
        **state,
        "processed":           processed,
        "insights":            insights,
        "trend_evolution":     trend_evolution,
        "opportunity_scores":  opportunity_scores,
        "error":               None,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("query_node",    query_node)
    g.add_node("collect_node",  collect_node)
    g.add_node("route_node",    _route_passthrough)   # passthrough so conditional edges work
    g.add_node("enrich_node",   enrich_node)
    g.add_node("finalize_node", finalize_node)

    # Linear edges
    g.set_entry_point("query_node")
    g.add_edge("query_node",   "collect_node")
    g.add_edge("collect_node", "route_node")

    # Conditional edges from router
    g.add_conditional_edges(
        "route_node",
        lambda s: s.get("_next", "finalize_node"),  # reads pre-computed decision — no double call
        {
            "collect_node":  "collect_node",
            "enrich_node":   "enrich_node",
            "finalize_node": "finalize_node",
        },
    )

    g.add_edge("enrich_node",  "finalize_node")
    g.add_edge("finalize_node", END)

    return g.compile()


def _route_passthrough(state: PipelineState) -> PipelineState:
    """
    Passthrough node: computes routing decision ONCE and stores it in state['_next'].
    The conditional edge function reads _next directly — avoids calling route_node() twice.
    """
    next_node = route_node(state)
    new_retry = state.get("retry_count", 0)
    if next_node == "collect_node":
        new_retry += 1
    return {**state, "retry_count": new_retry, "_next": next_node}


# Compiled graph singleton — built once, reused across requests
_GRAPH = None

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Public entry point — called from main.py analyze endpoint
# ---------------------------------------------------------------------------

def run_pipeline(
    topic: str,
    sources: str = "web_news",
    linkedin_post_count: int = 25,
    linkedin_date_since: Optional[datetime] = None,
    linkedin_region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full LangGraph intelligence pipeline.

    Returns a dict with keys:
      query_analysis, queries, validated_results, validated_query_groups,
      processed, insights, trend_evolution, opportunity_scores, elapsed_seconds, error
    """
    t0 = time.perf_counter()

    initial_state: PipelineState = {
        "topic":                   topic,
        "sources":                 sources,
        "t0":                      t0,
        "linkedin_post_count":     linkedin_post_count,
        "linkedin_date_since":     linkedin_date_since,
        "linkedin_region":         linkedin_region,
        "query_analysis":          None,
        "queries":                 None,
        "raw_results":             None,
        "query_groups":            None,
        "validated_results":       None,
        "validated_query_groups":  None,
        "processed":               None,
        "retry_count":             0,
        "enriched":                False,
        "insights":                None,
        "trend_evolution":         None,
        "opportunity_scores":      None,
        "error":                   None,
    }

    logger.info("[LangGraph] run_pipeline: topic='%s' sources='%s'", topic, sources)
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("[LangGraph] run_pipeline: finished in %.2fs error=%s", elapsed, final_state.get("error"))

    return {**final_state, "elapsed_seconds": elapsed}