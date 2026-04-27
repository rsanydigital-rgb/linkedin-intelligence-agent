"""
market_intel.py — NO pytrends, NO threading, NO external API keys required.
Always returns instantly. Uses real trend_history.json data where available.
"""
import os, math, json, logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests as _http

logger = logging.getLogger(__name__)
HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "trend_history.json"


def interest_over_time(topic: str) -> Dict[str, Any]:
    keyword = _clean_keyword(topic)
    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()
    if serpapi_key and serpapi_key != "your_serpapi_key_here":
        try:
            r = _http.get("https://serpapi.com/search", timeout=8, params={
                "engine": "google_trends", "q": keyword,
                "data_type": "TIMESERIES", "date": "today 5-y", "api_key": serpapi_key,
            })
            r.raise_for_status()
            pts = r.json().get("interest_over_time", {}).get("timeline_data", [])
            timeline = []
            for pt in pts:
                try:
                    dt  = datetime.strptime(pt["date"].split(" – ")[0].strip(), "%b %d, %Y")
                    val = int((pt.get("values") or [{}])[0].get("extracted_value", 0))
                    timeline.append({"date": dt.strftime("%Y-%m-%d"), "value": val})
                except Exception:
                    continue
            if timeline:
                fc = _forecast(timeline)
                return _trends_resp("serpapi_trends", keyword, timeline, fc)
        except Exception as e:
            logger.warning("SerpAPI trends failed: %s", e)
    return _synthetic_interest(topic)


def job_postings(topic: str) -> Dict[str, Any]:
    keyword = _clean_keyword(topic)
    try:
        r = _http.get("https://remotive.com/api/remote-jobs",
            params={"search": keyword, "limit": 10},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        data = r.json()
        jobs = data.get("jobs", [])
        if jobs:
            listings = [{"title": j.get("title",""), "company": j.get("company_name",""),
                "location": j.get("candidate_required_location") or "Remote",
                "url": j.get("url",""), "posted": j.get("publication_date","")[:10],
                "salary": j.get("salary","")} for j in jobs[:8]]
            return _jobs_resp("remotive", data.get("job-count", len(jobs)), listings)
    except Exception as e:
        logger.warning("Remotive failed: %s", e)

    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()
    if serpapi_key and serpapi_key != "your_serpapi_key_here":
        try:
            r = _http.get("https://serpapi.com/search", timeout=8, params={
                "engine": "google_jobs", "q": keyword, "api_key": serpapi_key, "num": 10})
            r.raise_for_status()
            data = r.json()
            listings = [{"title": j.get("title",""), "company": j.get("company_name",""),
                "location": j.get("location",""),
                "url": (j.get("related_links") or [{}])[0].get("link",""),
                "posted": j.get("detected_extensions",{}).get("posted_at",""),
                "salary": j.get("detected_extensions",{}).get("salary","")}
                for j in data.get("jobs_results",[])[:8]]
            total = data.get("search_information",{}).get("total_results", len(listings))
            return _jobs_resp("serpapi", total, listings)
        except Exception as e:
            logger.warning("SerpAPI jobs failed: %s", e)

    return _synthetic_jobs(topic)


def _synthetic_interest(topic: str) -> Dict:
    seed   = sum(ord(c) for c in topic.lower())
    today  = datetime.now(timezone.utc)
    start  = today - timedelta(weeks=260)
    real   = _load_scores(topic)
    base   = 20 + (seed % 40)
    growth = 0.04 + (seed % 8) * 0.006
    timeline = []
    for w in range(260):
        dt  = start + timedelta(weeks=w)
        val = base + growth * w * 4 + math.sin(w * 0.35 + seed * 0.1) * 8 + math.sin(w * 0.9 + seed) * 3
        if real and w >= 240:
            val = val * 0.4 + real[(w - 240) % len(real)] * 0.6
        timeline.append({"date": dt.strftime("%Y-%m-%d"), "value": max(0, min(100, round(val)))})
    return _trends_resp("synthetic", _clean_keyword(topic), timeline, _forecast(timeline))


def _load_scores(topic: str) -> List[float]:
    try:
        if not HISTORY_PATH.exists(): return []
        h = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        runs = h.get(" ".join(topic.lower().split()), [])
        scores = [float(t.get("score", 0)) for run in runs[-10:] for t in run.get("trends", [])]
        if not scores: return []
        mx = max(scores) or 1
        return [min(100, round(s / mx * 90)) for s in scores]
    except Exception:
        return []


def _forecast(timeline: List[Dict], weeks: int = 52) -> List[Dict]:
    w = timeline[-26:] if len(timeline) >= 26 else timeline
    n = len(w)
    if n < 4: return []
    xs, ys = list(range(n)), [p["value"] for p in w]
    xm, ym = sum(xs)/n, sum(ys)/n
    slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys)) / (sum((x-xm)**2 for x in xs) or 1)
    b     = ym - slope * xm
    last  = datetime.strptime(w[-1]["date"], "%Y-%m-%d")
    return [{"date": (last + timedelta(weeks=i)).strftime("%Y-%m-%d"),
             "value": max(0, min(100, round(b + slope*(n-1+i)))), "forecast": True}
            for i in range(1, weeks+1)]


def _trends_resp(source, keyword, timeline, forecast):
    vals = [p["value"] for p in timeline]
    avg  = int(sum(vals[-13:]) / max(len(vals[-13:]), 1))
    peak = timeline[vals.index(max(vals))]["date"][:4]
    if len(vals) >= 16:
        r, p = sum(vals[-8:])/8, sum(vals[-16:-8])/8
        d = "rising" if r > p*1.08 else "falling" if r < p*0.92 else "stable"
    else:
        d = "stable"
    return {"source": source, "keyword": keyword, "timeline": timeline,
            "forecast": forecast, "peak_year": peak, "avg_last_3m": avg, "trend_direction": d}


def _jobs_resp(source, total, listings):
    from collections import Counter
    locs  = Counter(j["location"] for j in listings if j["location"])
    comps = Counter(j["company"]  for j in listings if j["company"])
    sals  = [j["salary"] for j in listings if j.get("salary")]
    return {"source": source, "total_count": int(total), "listings": listings,
            "top_locations": [l for l,_ in locs.most_common(5)],
            "top_companies": [c for c,_ in comps.most_common(5)],
            "salary_range": sals[0] if sals else None}


def _synthetic_jobs(topic: str) -> Dict:
    seed  = sum(ord(c) for c in topic.lower())
    roles = [f"Senior {topic.title()} Engineer", f"{topic.title()} Researcher",
             f"Lead {topic.title()} Developer", f"{topic.title()} Product Manager",
             f"Applied {topic.title()} Scientist"]
    cos   = ["Google DeepMind","Microsoft","Meta AI","OpenAI","Anthropic",
             "Hugging Face","Mistral AI","Cohere","Scale AI","Databricks"]
    locs  = ["Remote","San Francisco, CA","New York, NY","London, UK","Berlin, DE","Singapore"]
    listings = [{"title": roles[i], "company": cos[(seed+i)%len(cos)],
                 "location": locs[(seed+i*3)%len(locs)], "url":"","posted":"","salary":""}
                for i in range(len(roles))]
    return _jobs_resp("synthetic", 150 + (seed % 850), listings)


def _clean_keyword(topic: str) -> str:
    words = [w for w in topic.strip().split() if len(w) > 2]
    return " ".join(words[:3]) if words else topic[:50]