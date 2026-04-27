"""
sentiment.py
------------
Zero-dependency lexicon-based sentiment analysis for LinkedIn posts and
any collected content.

Why lexicon instead of a model?
  - No extra pip installs (no VADER, TextBlob, transformers)
  - Runs in microseconds per post — no latency impact
  - Tuned for business/tech language (LinkedIn, news, SerpAPI snippets)
  - Returns a consistent dict that flows into processor → API → frontend

Output schema per document:
  {
    "label":    "positive" | "negative" | "neutral",
    "score":    float  # -1.0 (very negative) … +1.0 (very positive)
    "positive": int    # raw positive signal count
    "negative": int    # raw negative signal count
  }
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicon — prefix-matched so "innovat" matches "innovation", "innovative" etc.
# Weighted: 2 = strong signal, 1 = weak signal
# ---------------------------------------------------------------------------

_POSITIVE: Dict[str, int] = {
    # Growth & momentum
    "growth": 2, "grow": 1, "soar": 2, "surge": 2, "boom": 2, "accelerat": 2,
    "expand": 1, "scale": 1, "momentum": 2, "outperform": 2,
    # Innovation
    "innovat": 2, "breakthrough": 2, "launch": 1, "pioneer": 2, "disrupt": 1,
    "transform": 1, "revolutioniz": 2, "advance": 1, "cutting-edge": 2, "state-of-the-art": 2,
    # Financial
    "profit": 2, "revenue": 1, "gain": 1, "record": 1, "invest": 1, "fund": 1,
    "unicorn": 2, "ipo": 1, "valuation": 1, "roi": 1,
    # Hiring / positive employment
    "hire": 1, "hiring": 2, "recruit": 1, "talent": 1, "promot": 1,
    # Sentiment words
    "excit": 2, "proud": 2, "thrilled": 2, "delighted": 2, "honor": 1,
    "optimis": 2, "confident": 1, "success": 2, "achiev": 2, "win": 2,
    "award": 2, "recogni": 1, "celebrat": 1,
    # Product / business
    "partner": 1, "collaborat": 1, "opportunit": 2, "adoption": 2, "leading": 1,
    "robust": 1, "efficient": 1, "improve": 1, "enhanc": 1, "empower": 1,
    "best": 1, "top": 1, "strong": 1, "excel": 2,
}

_NEGATIVE: Dict[str, int] = {
    # Job losses
    "layoff": 3, "laid off": 3, "job loss": 3, "redundan": 3, "retrench": 3,
    "downsize": 2, "furlough": 2,
    # Financial distress
    "loss": 2, "declin": 2, "drop": 1, "fall": 1, "shrink": 2, "plummet": 3,
    "bankrupt": 3, "debt": 1, "deficit": 2, "cut": 1,
    # Risk / threats
    "risk": 1, "threat": 2, "vulnerab": 2, "breach": 3, "hack": 3, "fraud": 3,
    "exploit": 2, "attack": 2, "leak": 2, "scam": 3,
    # Crisis
    "crisis": 3, "crash": 3, "collaps": 3, "fail": 2, "shut down": 2, "bankrupt": 3,
    "concern": 1, "challeng": 1, "problem": 1, "difficult": 1, "uncertain": 1,
    # Ethical / regulatory
    "regulat": 1, "fine": 1, "lawsuit": 2, "litigat": 2, "penalt": 2,
    "bias": 2, "discriminat": 2, "unethic": 3, "mislead": 2, "manipulat": 3,
    "toxic": 2, "harmful": 2, "dangerous": 2, "controversial": 1,
    # Displacement
    "displace": 2, "replac": 1, "automat": 1,  # mild — context-dependent
    "fear": 2, "warn": 1, "slow": 1, "poor": 2, "weak": 1,
}

# Negation words that flip the next sentiment token
_NEGATORS = {"not", "no", "never", "without", "lack", "lacking", "fails", "failed"}

# Intensifiers that multiply signal weight
_INTENSIFIERS = {"very", "extremely", "highly", "incredibly", "massively", "significantly"}


def analyse(text: str) -> Dict:
    """
    Return sentiment dict for *text*.

    Args:
        text: raw post/article content

    Returns:
        {"label": str, "score": float, "positive": int, "negative": int}
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "positive": 0, "negative": 0}

    text_lower = text.lower()
    # Tokenise — keep hyphenated terms together
    tokens = re.findall(r"[a-z][a-z0-9\-']*", text_lower)

    pos_total = 0
    neg_total = 0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check for 2-gram match first (e.g. "job loss", "laid off")
        bigram = f"{token} {tokens[i+1]}" if i + 1 < len(tokens) else ""

        # Determine multiplier from preceding context (look back 2 tokens)
        multiplier = 1
        negated = False
        for j in range(max(0, i - 2), i):
            if tokens[j] in _NEGATORS:
                negated = True
            if tokens[j] in _INTENSIFIERS:
                multiplier = 2

        def _match_pos(term: str) -> int:
            for prefix, weight in _POSITIVE.items():
                if term.startswith(prefix) or prefix in term:
                    return weight
            return 0

        def _match_neg(term: str) -> int:
            for prefix, weight in _NEGATIVE.items():
                if term.startswith(prefix) or prefix in term:
                    return weight
            return 0

        # Bigram check
        if bigram:
            bp = _match_pos(bigram)
            bn = _match_neg(bigram)
            if bp or bn:
                if negated:
                    neg_total += bp * multiplier
                    pos_total += bn * multiplier
                else:
                    pos_total += bp * multiplier
                    neg_total += bn * multiplier
                i += 2
                continue

        # Unigram check
        p = _match_pos(token)
        n = _match_neg(token)
        if negated:
            neg_total += p * multiplier
            pos_total += n * multiplier
        else:
            pos_total += p * multiplier
            neg_total += n * multiplier

        i += 1

    # Normalise to [-1, +1]
    total_signal = pos_total + neg_total
    if total_signal == 0:
        score = 0.0
    else:
        score = round((pos_total - neg_total) / total_signal, 3)

    label = "positive" if score > 0.08 else "negative" if score < -0.08 else "neutral"

    return {
        "label": label,
        "score": score,
        "positive": pos_total,
        "negative": neg_total,
    }


def batch_analyse(texts: list) -> list:
    """Analyse a list of texts, returns list of sentiment dicts."""
    return [analyse(t) for t in texts]


def aggregate(sentiments: list) -> Dict:
    """
    Summarise a list of sentiment dicts into aggregate stats.

    Returns:
        {
          "overall_label":    str,
          "overall_score":    float,   # mean score
          "positive_pct":     float,   # 0-100
          "negative_pct":     float,
          "neutral_pct":      float,
          "count":            int,
        }
    """
    if not sentiments:
        return {
            "overall_label": "neutral", "overall_score": 0.0,
            "positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 100.0,
            "count": 0,
        }

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    score_sum = 0.0
    for s in sentiments:
        counts[s.get("label", "neutral")] = counts.get(s.get("label", "neutral"), 0) + 1
        score_sum += s.get("score", 0.0)

    n = len(sentiments)
    mean_score = round(score_sum / n, 3)
    overall_label = "positive" if mean_score > 0.08 else "negative" if mean_score < -0.08 else "neutral"

    return {
        "overall_label": overall_label,
        "overall_score": mean_score,
        "positive_pct":  round(counts["positive"]  / n * 100, 1),
        "negative_pct":  round(counts["negative"]  / n * 100, 1),
        "neutral_pct":   round(counts["neutral"]   / n * 100, 1),
        "count": n,
    }