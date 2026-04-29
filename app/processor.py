"""
processor.py
------------
Deterministic content processing pipeline. NO LLM calls here.

Flow:
  1. Build a corpus from CollectedPost.content fields.
  2. Extract candidate keywords with TF-IDF.
  3. Score and normalize document engagement to 0-100.
  4. Filter low-engagement posts before trend detection.
  5. Recompute TF-IDF and cluster documents into multiple trend groups.
  6. Attach per-document sentiment (lexicon-based, zero extra dependencies).
"""

import re
import json
import logging
import os
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

from app.validation import CollectedPost
from app.sentiment import analyse as sentiment_analyse, aggregate as sentiment_aggregate
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

STOPWORDS = set(ENGLISH_STOP_WORDS)
LOW_SIGNAL_TERMS = {
    "new", "using", "time", "latest", "based", "used", "use", "like",
    "said", "says", "year", "years", "week", "today", "make", "made",
    "usd", "billion", "million", "trillion", "percent", "including",
    "included", "report", "news", "analysis", "results", "quarter",
    "just", "also", "well", "first", "second", "third", "fourth",
    "number", "numbers", "data", "source", "sources", "according",
}
GENERIC_LABEL_TERMS = {
    "technology", "technologies", "system", "systems", "platform", "platforms",
    "solution", "solutions", "service", "services", "industry", "market",
    "global", "company", "companies", "business", "west", "east",
}


def _llm_label_clusters(topic: str, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use OpenAI to generate meaningful labels for trend clusters."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    orchestrator_model = os.getenv("LLM_ORCHESTRATOR_MODEL", "gpt-4o-mini").strip()
    if not openai_key or openai_key in ("your_openai_key_here", ""):
        return clusters

    cluster_data = [
        {"id": i, "keywords": c["keywords"][:6]}
        for i, c in enumerate(clusters)
    ]

    system = (
        "You are a trend analyst. Given a topic and keyword clusters from content analysis, "
        "generate a short, meaningful label (3-5 words) for each cluster that captures its core theme. "
        "Labels must be specific, professional, and directly relevant to the topic. "
        "No generic terms like 'Industry Trends', 'Market Analysis', or 'New Technology'. "
        "Return ONLY a JSON array of strings, one label per cluster, in the same order. No extra text."
    )
    user = f"Topic: {topic}\nClusters: {json.dumps(cluster_data)}"

    try:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": orchestrator_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            },
            timeout=10,  # hard cap — cluster labelling must not block the pipeline
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        labels = json.loads(raw)
        if isinstance(labels, list) and len(labels) == len(clusters):
            for i, cluster in enumerate(clusters):
                if isinstance(labels[i], str) and len(labels[i]) > 2:
                    cluster["label"] = labels[i]
    except Exception as exc:
        logger.warning("LLM cluster labelling failed (using rule-based labels): %s", exc)

    return clusters
ENGAGEMENT_THRESHOLD = 5.0
SOURCE_WEIGHTS = {
    "news": 0.8,
    "blog": 0.5,
    "unknown": 0.3,
}


class ProcessedData:
    """Lightweight container; intentionally not a Pydantic model to keep
    processor.py free of I/O concerns."""

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        trends: List[Dict[str, Any]],
        top_keywords: List[str],
    ):
        self.documents = documents
        self.trends = trends
        self.top_keywords = top_keywords
        self.sentiment_summary: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "trends": self.trends,
            "top_keywords": self.top_keywords,
            "sentiment_summary": self.sentiment_summary,
        }


def process(results: List[CollectedPost], topic: str = "") -> ProcessedData:
    if not results:
        logger.warning("process() called with empty results list")
        return ProcessedData(documents=[], trends=[], top_keywords=[])

    initial_corpus = [r.content for r in results]
    initial_matrix, initial_features, _ = _build_tfidf(initial_corpus)
    candidate_keywords = extract_keywords(initial_corpus, initial_matrix, initial_features)

    scored_results = _score_posts(results, candidate_keywords)
    filtered_results = [
        item for item in scored_results
        if item["engagement_score"] >= ENGAGEMENT_THRESHOLD
        or (
            # Never drop a LinkedIn post that has real engagement data —
            # short posts naturally score low on TF-IDF but are still signal.
            item["post"].source == "linkedin"
            and sum(
                item["post"].metadata.get(k) or 0
                for k in ("likes", "comments", "shares")
            ) > 0
        )
    ]

    if not filtered_results:
        logger.warning(
            "All posts fell below the engagement threshold of %.1f",
            ENGAGEMENT_THRESHOLD,
        )
        return ProcessedData(documents=[], trends=[], top_keywords=[])

    filtered_posts = [item["post"] for item in filtered_results]
    filtered_corpus = [post.content for post in filtered_posts]
    tfidf_matrix, feature_names, _ = _build_tfidf(filtered_corpus)

    documents = []
    for idx, item in enumerate(filtered_results):
        post = item["post"]
        doc_keywords = _top_keywords_for_doc(tfidf_matrix, feature_names, idx, n=8)
        source_type = _infer_source_type(post.source, post.url)
        documents.append(
            {
                "source": post.source,
                "source_type": source_type,
                "source_weight": SOURCE_WEIGHTS[source_type],
                "url": post.url,
                "content_preview": post.content[:200] + ("..." if len(post.content) > 200 else ""),
                "engagement_score": round(item["engagement_score"], 2),
                "keywords": doc_keywords,
                "collected_at": post.collected_at.isoformat(),
                "sentiment": sentiment_analyse(post.content),
                "metadata": post.metadata,  # likes/comments/shares/author (LinkedIn)
            }
        )

    trends = get_trends(filtered_corpus, tfidf_matrix, feature_names)

    # Enrich cluster labels with LLM — runs in a background thread with a hard timeout
    # so a slow/unavailable LLM never blocks the full pipeline response.
    if topic and trends:
        with ThreadPoolExecutor(max_workers=1) as _pool:
            _fut = _pool.submit(_llm_label_clusters, topic, trends)
            try:
                trends = _fut.result(timeout=12)
            except (FuturesTimeoutError, Exception) as _exc:
                logger.warning("LLM cluster labelling skipped (timeout/error): %s", _exc)

    top_keywords = []
    for trend in trends:
        for keyword in trend["keywords"]:
            if keyword not in top_keywords:
                top_keywords.append(keyword)
    top_keywords = top_keywords[:10]

    # Aggregate sentiment across all documents
    sentiment_summary = sentiment_aggregate([d["sentiment"] for d in documents])

    # Per-source breakdown (useful for LinkedIn vs news comparison)
    source_sentiment: Dict[str, Any] = {}
    for doc in documents:
        src = doc["source"]
        if src not in source_sentiment:
            source_sentiment[src] = []
        source_sentiment[src].append(doc["sentiment"])
    sentiment_summary["by_source"] = {
        src: sentiment_aggregate(sents)
        for src, sents in source_sentiment.items()
    }

    result = ProcessedData(documents=documents, trends=trends, top_keywords=top_keywords)
    result.sentiment_summary = sentiment_summary
    return result


def _build_tfidf(corpus: List[str]):
    vectorizer = TfidfVectorizer(
        max_features=250,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return matrix, feature_names, vectorizer


def _top_keywords_for_doc(matrix, feature_names: np.ndarray, doc_idx: int, n: int = 8) -> List[str]:
    row = matrix[doc_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:n]
    return [feature_names[i] for i in top_indices if row[i] > 0 and not _is_low_signal(feature_names[i])]


def extract_keywords(
    corpus: List[str],
    matrix,
    feature_names: np.ndarray,
    max_keywords: int = 40,
) -> List[Dict[str, Any]]:
    """Score keywords using normalized TF-IDF and raw frequency."""
    if matrix.shape[1] == 0:
        return []

    mean_tfidf = np.asarray(matrix.mean(axis=0)).flatten()
    raw_frequency = np.array(
        [_count_term_frequency(corpus, term) for term in feature_names],
        dtype=float,
    )

    tfidf_norm = mean_tfidf / max(float(mean_tfidf.max()), 1e-9)
    freq_norm = raw_frequency / max(float(raw_frequency.max()), 1.0)
    combined_score = 0.65 * tfidf_norm + 0.35 * freq_norm

    ranked_indices = combined_score.argsort()[::-1]
    keywords: List[Dict[str, Any]] = []

    for index in ranked_indices:
        term = feature_names[index]
        if _is_low_signal(term):
            continue

        keywords.append(
            {
                "keyword": term,
                "tfidf_score": round(float(mean_tfidf[index]), 5),
                "frequency": int(raw_frequency[index]),
                "score": round(float(combined_score[index]), 5),
            }
        )
        if len(keywords) >= max_keywords:
            break

    return keywords


def engagement_score(content: str, keyword_items: List[Dict[str, Any]]) -> float:
    """Deterministic engagement heuristic before trend detection."""
    words = re.findall(r"[a-zA-Z0-9\-]+", content.lower())
    word_count = len(words)
    if word_count == 0:
        return 0.0

    keyword_lookup = {item["keyword"].lower(): item["score"] for item in keyword_items}
    important_keywords = {
        item["keyword"].lower()
        for item in keyword_items[:10]
        if item["score"] >= 0.3
    }

    base = min(word_count, 400) / 400.0
    weighted_hits = 0.0
    important_hits = 0
    content_lower = content.lower()

    for keyword, score in keyword_lookup.items():
        occurrences = content_lower.count(keyword)
        if occurrences:
            weighted_hits += occurrences * score
            if keyword in important_keywords:
                important_hits += 1

    density = min(weighted_hits / max(word_count, 1), 0.12) / 0.12
    importance = min(important_hits / max(len(important_keywords), 1), 1.0)
    return 100.0 * (0.5 * base + 0.3 * density + 0.2 * importance)


def _score_posts(results: List[CollectedPost], keyword_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_scores = []
    for post in results:
        text_score = engagement_score(post.content, keyword_items) * (
            0.7 + _source_weight(post.source, post.url)
        )

        # Blend real LinkedIn engagement (likes + comments + shares) into the score.
        # A post with 500 total interactions gets up to +30 pts boost, capped there.
        meta = post.metadata or {}
        real_eng = (
            (meta.get("likes") or 0)
            + (meta.get("comments") or 0)
            + (meta.get("shares") or 0)
        )
        real_boost = min(real_eng / 500.0, 1.0) * 30.0

        raw_scores.append({"post": post, "raw_score": text_score + real_boost})

    values = np.array([item["raw_score"] for item in raw_scores], dtype=float)
    min_score = float(values.min()) if len(values) else 0.0
    max_score = float(values.max()) if len(values) else 0.0

    for item in raw_scores:
        if max_score - min_score < 1e-9:
            normalized = 100.0 if item["raw_score"] > 0 else 0.0
        else:
            normalized = (item["raw_score"] - min_score) / (max_score - min_score) * 100.0
        item["engagement_score"] = round(normalized, 2)

    return raw_scores


def cluster_keywords(
    matrix,
    corpus: List[str],
    feature_names: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Cluster documents into multiple trend groups and extract representative keywords
    per cluster.
    """
    num_docs = matrix.shape[0]
    if num_docs == 0:
        return []
    if num_docs == 1:
        return [_build_cluster_trend(corpus, matrix, feature_names, np.array([0]), 0)]

    best_k = _select_optimal_k(matrix)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    trends = []
    for cluster_id in range(best_k):
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        trend = _build_cluster_trend(corpus, matrix, feature_names, member_indices, cluster_id)
        if trend["keywords"]:
            trends.append(trend)

    trends.sort(key=lambda x: x["score"], reverse=True)
    return trends


def get_trends(
    corpus: List[str],
    matrix,
    feature_names: np.ndarray,
    max_trends: int = 12,
) -> List[Dict[str, Any]]:
    clusters = cluster_keywords(matrix, corpus, feature_names)
    return clusters[:max_trends]


def _select_optimal_k(matrix) -> int:
    num_docs = matrix.shape[0]
    min_k = 3
    max_k = min(6, num_docs)

    if num_docs <= 3:
        return min(3, num_docs)

    candidate_ks = [k for k in range(min_k, max_k + 1) if k < num_docs]
    if not candidate_ks:
        return min(3, num_docs)

    best_k = candidate_ks[0]
    best_score = -1.0

    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _build_cluster_trend(
    corpus: List[str],
    matrix,
    feature_names: np.ndarray,
    member_indices: np.ndarray,
    cluster_id: int,
) -> Dict[str, Any]:
    cluster_matrix = matrix[member_indices]
    cluster_docs = [corpus[idx] for idx in member_indices]

    mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()
    raw_frequency = np.array(
        [_count_term_frequency(cluster_docs, term) for term in feature_names],
        dtype=float,
    )

    candidate_indices = mean_tfidf.argsort()[::-1]
    keywords: List[str] = []
    keyword_scores: List[float] = []

    for index in candidate_indices:
        term = feature_names[index]
        if _is_low_signal(term):
            continue
        if raw_frequency[index] <= 0:
            continue

        keywords.append(term)
        keyword_scores.append(float(mean_tfidf[index]) * float(raw_frequency[index]))
        if len(keywords) >= 8:
            break

    if not keywords:
        return {"label": f"cluster_{cluster_id}", "keywords": [], "score": 0.0}

    label = _build_cluster_label(keywords)
    avg_tfidf = float(np.mean([mean_tfidf[feature_names.tolist().index(term)] for term in keywords]))
    total_frequency = float(sum(_count_term_frequency(cluster_docs, term) for term in keywords))
    score = round(avg_tfidf * total_frequency, 5)

    return {
        "label": label,
        "keywords": keywords[:8],
        "score": score,
    }


def _count_term_frequency(corpus: List[str], term: str) -> int:
    pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
    return sum(len(pattern.findall(doc.lower())) for doc in corpus)


def _infer_source_type(source: str, url: str) -> str:
    source_name = (source or "").lower()
    host = urlparse(url).netloc.lower()
    if source_name == "newsapi" or any(part in host for part in ("news", "reuters", "techcrunch", "forbes", "wsj", "bbc", "cnn")):
        return "news"
    if any(part in host for part in ("blog", "medium", "substack", "wordpress")):
        return "blog"
    return "unknown"


def _source_weight(source: str, url: str) -> float:
    return SOURCE_WEIGHTS[_infer_source_type(source, url)]


def _build_cluster_label(keywords: List[str]) -> str:
    cleaned_phrases: List[str] = []
    seen_tokens = set()

    for keyword in keywords:
        phrase = _clean_label_phrase(keyword)
        if not phrase:
            continue

        phrase_tokens = phrase.lower().split()
        novel_tokens = [token for token in phrase_tokens if token not in seen_tokens]
        if not novel_tokens:
            continue

        cleaned_phrase = " ".join(novel_tokens)
        cleaned_phrases.append(cleaned_phrase)
        seen_tokens.update(novel_tokens)

        if len(" ".join(cleaned_phrases).split()) >= 4:
            break

    label_tokens = " ".join(cleaned_phrases).split()[:4]
    if not label_tokens:
        return "Emerging Topic"

    title = " ".join(label_tokens)
    return _title_case_label(title)


def _clean_label_phrase(keyword: str) -> str:
    tokens = []
    for token in keyword.lower().split():
        if token in STOPWORDS or token in LOW_SIGNAL_TERMS or token in GENERIC_LABEL_TERMS:
            continue
        if len(token) < 3 or re.fullmatch(r"\d+", token):
            continue
        tokens.append(token)

    if not tokens:
        return ""

    if len(tokens) > 3:
        tokens = tokens[:3]
    return " ".join(tokens)


def _title_case_label(label: str) -> str:
    acronyms = {"ai", "ml", "nlp", "ros"}
    words = []
    for word in label.split():
        if word in acronyms:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _is_low_signal(term: str) -> bool:
    cleaned = term.strip().lower()
    if len(cleaned) < 3 or re.fullmatch(r"\d+", cleaned):
        return True

    tokens = cleaned.split()
    if not tokens:
        return True

    if any(token in LOW_SIGNAL_TERMS for token in tokens):
        return True

    if all(token in STOPWORDS for token in tokens):
        return True

    return any(len(token) < 3 for token in tokens)