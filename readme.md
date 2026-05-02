# LinkedIn Intelligence Agent

An AI-assisted market and LinkedIn intelligence app for discovering topic trends, collecting supporting sources, scoring opportunities, generating strategic insights, and exporting reports.

The project has two main parts:

- A FastAPI backend in `main.py` and `app/`
- A static frontend in `frontend/` served by the backend

It can run with only Python installed. Optional services such as Redis, PostgreSQL, Apify, Notion, Slack, SendGrid, SerpAPI, NewsAPI, and n8n add caching, persistence, live data, exports, alerts, and scheduling.

## What This Project Does

Given a topic such as `AI recruiting tools` or `robotics in warehouses`, the app:

1. Classifies whether the topic is broad, vague, or specific.
2. Expands the topic into multiple focused search queries.
3. Collects source material from web search, news, mock fallback data, and optionally LinkedIn through Apify.
4. Validates, deduplicates, and normalizes collected records.
5. Extracts keywords with TF-IDF.
6. Scores source relevance and engagement.
7. Clusters documents into trend groups.
8. Computes trend evolution against previous runs.
9. Scores business opportunities.
10. Runs sentiment analysis.
11. Generates executive insights with an LLM when configured, or a deterministic fallback when not.
12. Stores, caches, exports, and notifies through optional integrations.

## Tech Stack

- Backend: FastAPI, Uvicorn, Pydantic
- Pipeline orchestration: LangGraph
- NLP and clustering: scikit-learn, NumPy, TF-IDF, KMeans
- LLM support: OpenAI-compatible chat completions or Anthropic Claude
- Data collection: SerpAPI, NewsAPI, Apify LinkedIn actors, mock fallback data
- Cache: Redis
- Persistence: PostgreSQL plus local JSON trend history
- Reports and alerts: Notion, Slack, SendGrid, n8n
- Frontend: plain HTML, CSS, and JavaScript
- Deployment: Docker and Docker Compose

## Project Structure

```text
.
|-- main.py
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .env
|-- app/
|   |-- apify_collector.py
|   |-- cache.py
|   |-- collector.py
|   |-- insights.py
|   |-- langgraph_orchestrator.py
|   |-- logging_setup.py
|   |-- market_intel_final.py
|   |-- notifier.py
|   |-- notion_output.py
|   |-- opportunities.py
|   |-- processor.py
|   |-- query_expansion.py
|   |-- query_intelligence.py
|   |-- sentiment.py
|   |-- storage.py
|   |-- trend_history.py
|   |-- validation.py
|   `-- __init__.py
|-- data/
|   `-- trend_history.json
`-- frontend/
    |-- index.html
    |-- dashboard.html
    |-- explore.html
    `-- reports.html
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create or update `.env` in the project root. The app can run with many services disabled because it has graceful fallbacks.

Minimum useful local setup:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_ORCHESTRATOR_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=20
```

If no LLM key is configured, query expansion and insights still work with rule-based fallbacks.

### 4. Run the app

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

## Run With Docker Compose

Docker Compose starts:

- The FastAPI app
- Redis
- PostgreSQL

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:8000
```

In Docker, `REDIS_URL` and `DATABASE_URL` are automatically set to point at the Compose services.

## Frontend Pages

- `frontend/index.html`: Main discovery page. Runs a topic analysis, displays query intelligence, sources, trends, insights, actions, config status, and metrics.
- `frontend/dashboard.html`: Visual dashboard for the most recent analysis stored in browser localStorage. Shows KPIs, trend charts, sentiment, market intel, jobs, and quick actions.
- `frontend/explore.html`: Deep-dive page for filtering source results by query, source, sentiment, date range, and LinkedIn engagement.
- `frontend/reports.html`: Automation and export page for n8n schedules, Notion export, email digest, Slack notification, and analysis history.

The backend serves these files directly:

- `/`
- `/index.html`
- `/dashboard.html`
- `/explore.html`
- `/reports.html`

## Backend API

### `GET /health`

Returns basic service health and version.

### `GET /config_status`

Returns whether optional integrations are configured:

- SerpAPI
- NewsAPI
- Apify
- Slack
- SendGrid
- Notion
- n8n
- Redis
- PostgreSQL
- LLM model and base URL

### `GET /metrics`

Returns cache, storage, notifier, Notion, and API usage status.

### `GET /history`

Returns recent analysis runs from PostgreSQL when configured.

### `POST /analyze`

Runs the full intelligence pipeline.

Example body:

```json
{
  "topic": "AI recruiting tools",
  "sources": "web_news",
  "linkedin_post_count": 25,
  "linkedin_timeline": "30d",
  "linkedin_region": "global"
}
```

Useful query parameter:

```text
refresh=true
```

This bypasses Redis cache and forces a fresh analysis.

Supported LinkedIn timeline values:

- `7h`
- `24h`
- `48h`
- `7d`
- `30d`
- `90d`
- `180d`

Supported LinkedIn region values:

- `global`
- `india`
- `asia_pacific`
- `southeast_asia`
- `north_america`
- `europe`
- `middle_east`

### `GET /market-intel?topic=...`

Returns market interest data and job posting data.

It tries:

- SerpAPI Google Trends when configured
- Remotive jobs API
- SerpAPI Google Jobs when configured
- Synthetic fallback data

### `GET /notion/status`

Checks whether Notion credentials and database access are working.

### `POST /export/notion`

Exports the latest/cached report for a topic to a Notion database.

```json
{
  "topic": "AI recruiting tools"
}
```

### `POST /notify/digest`

Sends an email report using SendGrid.

```json
{
  "topic": "AI recruiting tools",
  "email": "person@example.com",
  "report_data": {}
}
```

### `POST /notify/slack`

Posts the latest report summary to Slack.

```json
{
  "topic": "AI recruiting tools"
}
```

### `POST /schedule`

Sends a scheduling request to an n8n webhook.

```json
{
  "topic": "AI recruiting tools",
  "cron_expression": "0 9 * * *"
}
```

## Pipeline Architecture

The primary analysis route uses `app/langgraph_orchestrator.py`.

Pipeline flow:

```text
query_node
  -> collect_node
  -> route_node
      -> collect_node again, if too few results and retry is available
      -> enrich_node, if LinkedIn is requested and Apify is configured
      -> finalize_node
```

Finalization runs:

- deterministic processing
- insight generation
- trend history save/load
- trend evolution
- opportunity scoring

The graph is compiled once and reused across requests.

## Important Modules

### `main.py`

Creates the FastAPI app, serves frontend pages, defines request/response models, exposes API routes, handles cache lookups, invokes the LangGraph pipeline, writes successful runs to storage, and triggers non-fatal side effects such as Notion export and Slack alerts.

### `app/langgraph_orchestrator.py`

Defines the LangGraph state machine for analysis. It handles query generation, collection, routing decisions, optional LinkedIn enrichment, and final processing.

### `app/query_intelligence.py`

Classifies the input topic as broad, vague, or niche. It also suggests more focused alternatives for broad or vague topics.

### `app/query_expansion.py`

Expands a topic into focused search queries. It uses an LLM when configured and falls back to deterministic query templates when not.

### `app/collector.py`

Collects web and news data. It tries SerpAPI first, then NewsAPI if primary results are insufficient, then mock data if neither external source works.

### `app/apify_collector.py`

Collects LinkedIn posts through Apify. It supports post count, timeline cutoff, region filtering, primary/fallback actors, and engagement metadata extraction.

### `app/validation.py`

Normalizes collected records, rejects invalid records, deduplicates by content and URL hash, and preserves LinkedIn metadata.

### `app/processor.py`

Runs deterministic content analysis:

- TF-IDF keyword extraction
- engagement scoring
- low-signal filtering
- KMeans trend clustering
- optional LLM cluster labels
- source weighting
- sentiment aggregation

### `app/insights.py`

Generates executive insights. It prefers OpenAI-compatible APIs or Claude when configured. If the LLM fails or is missing, it returns rule-based insights instead of failing the request.

### `app/sentiment.py`

Runs fast lexicon-based sentiment analysis without extra ML dependencies.

### `app/trend_history.py`

Stores trend history locally in `data/trend_history.json` and compares current trends against the previous run for the same topic.

### `app/opportunities.py`

Scores trends as opportunities based on strength, growth, and uniqueness.

### `app/cache.py`

Uses Redis for 24-hour cached analysis results. If Redis is not configured or unavailable, the app continues without caching.

### `app/storage.py`

Uses PostgreSQL for run history and API usage tracking. It creates tables automatically on first use. If PostgreSQL is not configured, the app continues without database persistence.

### `app/notion_output.py`

Creates Notion report pages with summary, trends, takeaways, opportunities, confidence, and metadata.

### `app/notifier.py`

Sends optional Slack, SendGrid email, and webhook notifications. Failures are logged but do not break the analysis pipeline.

### `app/market_intel_final.py`

Provides interest-over-time and job market data. It uses real external sources when available and synthetic fallback data otherwise.

### `app/logging_setup.py`

Configures structured JSON logging for easier debugging and observability.

## Environment Variables

### LLM

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_ORCHESTRATOR_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=20
ANTHROPIC_API_KEY=
```

Notes:

- `OPENAI_BASE_URL` supports OpenAI-compatible providers.
- `LLM_ORCHESTRATOR_MODEL` is used for query expansion, insights, and cluster labels.
- `ANTHROPIC_API_KEY` is supported in `query_expansion.py` and `insights.py`.

### Data Collection

```env
SERPAPI_KEY=
NEWS_API_KEY=
SECONDARY_FETCH_THRESHOLD=10
APIFY_API_TOKEN=
APIFY_MAX_RESULTS_PER_QUERY=15
```

Notes:

- SerpAPI powers web search and optional trends/jobs lookups.
- NewsAPI is used as a secondary data source.
- Apify powers LinkedIn post search.
- Without these keys, mock/synthetic fallbacks keep the app usable.

### Cache and Storage

```env
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=86400
DATABASE_URL=postgresql://user:password@localhost:5432/linkedin_intel
```

### Reports and Notifications

```env
NOTION_API_KEY=
NOTION_DATABASE_ID=
SLACK_WEBHOOK_URL=
SENDGRID_API_KEY=
ALERT_EMAIL_TO=
ALERT_WEBHOOK_URL=
N8N_WEBHOOK_URL=
APP_BASE_URL=http://localhost:8000
```

Notes:

- Notion requires the integration to be shared with the target database.
- The Notion database only needs a `Name` title property.
- SendGrid sender email must be verified in SendGrid.
- n8n should use an active production webhook URL, not a temporary test webhook.

### Docker Compose Database Defaults

```env
POSTGRES_USER=linkedin_agent
POSTGRES_PASSWORD=changeme
POSTGRES_DB=linkedin_intel
```

## Data and Persistence

### Local trend history

`data/trend_history.json` stores trend snapshots by topic. This powers trend evolution even without PostgreSQL.

### Redis cache

When `REDIS_URL` is set, analysis results are cached using a deterministic cache key that includes:

- topic
- source selection
- LinkedIn post count
- LinkedIn timeline
- LinkedIn region

### PostgreSQL

When `DATABASE_URL` is set, these tables are created automatically:

- `trend_runs`
- `api_usage_log`
- `topic_history`

## Source Modes

The frontend can request:

- Web/news only
- Web/news plus LinkedIn

LinkedIn enrichment only runs when:

- the selected source includes LinkedIn
- `APIFY_API_TOKEN` is configured

If Apify is missing or fails, the pipeline still completes using available web/news/mock data.

## Fallback Behavior

The project is designed to degrade gracefully:

- No LLM key: rule-based query expansion and insights
- No SerpAPI or NewsAPI: deterministic mock source data
- No Apify: LinkedIn enrichment skipped
- No Redis: cache disabled
- No PostgreSQL: database history disabled
- No Notion/Slack/SendGrid/n8n: actions return helpful configuration errors
- LLM timeout or external API failure: logged and replaced with fallback behavior where possible

## Common Workflows

### Run a topic analysis from PowerShell

```powershell
$body = @{
  topic = "AI recruiting tools"
  sources = "web_news"
  linkedin_post_count = 25
  linkedin_timeline = "30d"
  linkedin_region = "global"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/analyze" -ContentType "application/json" -Body $body
```

### Force a fresh analysis

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/analyze?refresh=true" -ContentType "application/json" -Body $body
```

### Check integration status

```powershell
Invoke-RestMethod http://localhost:8000/config_status
```

### Check metrics

```powershell
Invoke-RestMethod http://localhost:8000/metrics
```

## Troubleshooting

### The app runs but results look generic

Most likely no external data keys are configured, so the app is using mock or synthetic fallback data. Add `SERPAPI_KEY`, `NEWS_API_KEY`, or `APIFY_API_TOKEN` for live collection.

### LinkedIn results are missing

Check:

- `APIFY_API_TOKEN` starts with `apify_api_`
- the selected source includes LinkedIn
- your Apify actor has access and enough credits
- the selected timeline and region are not too restrictive

### Notion export fails

Check:

- `NOTION_API_KEY` is set
- `NOTION_DATABASE_ID` is set
- the Notion integration is shared with the database
- the database has a title property named `Name`

You can test with:

```text
GET /notion/status
```

### Email digest fails

Check:

- `SENDGRID_API_KEY` is set
- `ALERT_EMAIL_TO` or the chosen sender is verified in SendGrid
- the recipient email is valid

### n8n scheduling returns 404

Use the production webhook URL and activate the workflow in n8n. Test webhook URLs often expire and do not work after the test session.

### Redis or PostgreSQL is disabled

This is okay for local development. The app continues without them. Use Docker Compose if you want both services started automatically.

### Docker app cannot reach local services

Inside Docker, `localhost` means the app container itself. Use Compose service names such as `redis` and `postgres`, or configure network-accessible hostnames.

## Development Notes

- The backend uses structured JSON logs configured in `app/logging_setup.py`.
- Most external side effects are intentionally non-fatal.
- Frontend state is stored in browser localStorage under keys such as `si_analysis`, `si_recent`, and `digest_email`.
- Generated Python caches such as `__pycache__/` should not be edited.
- Secrets belong in `.env` and should not be committed.

## Suggested Next Improvements

- Add automated tests for collector, processor, and API response contracts.
- Add a migration tool if PostgreSQL schema grows beyond the MVP tables.
- Move frontend JavaScript into separate files for easier maintenance.
- Add authentication before exposing this beyond local/internal use.
- Add rate limiting around expensive collection endpoints.
