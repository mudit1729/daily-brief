# Signal Brief

An AI-powered daily news intelligence engine that aggregates 60+ RSS sources, clusters related stories, ranks by relevance, and synthesizes a personalized daily briefing with market data, weather, investment signals, and tracked story timelines.

![Today](docs/screenshots/today.png)

## Features

### Daily Briefing
Multi-section news brief generated through a 5-stage pipeline (Acquire → Normalize → Compress → Rank → Synthesize). Each section is budget-constrained and LLM-summarized from clustered source articles across US News, AI & Tech, India, Geopolitics, Science, and Health.

News cards stack into swipeable decks — navigate with arrow buttons or reset to the first card.

![Card Stack](docs/screenshots/card_stack.png)

### Market Overview
Real-time market indices (S&P 500, Dow Jones, NASDAQ, Nifty 50, Sensex, Gold) with top market-moving story clusters ranked by impact score.

![Market](docs/screenshots/market.png)

### Timelines
Curated chronological event timelines tracking unfolding narratives across entities. LLM-powered auto-discovery finds new timelines and auto-updates existing ones with fresh events each pipeline run.

![Timelines](docs/screenshots/timelines.png)

### Tracked Stories
Long-running story threads grouped by topic (AI Regulation, Fed Interest Rates, Semiconductor Industry, etc.) with event timelines, source links, and activity tracking.

![Stories](docs/screenshots/stories.png)

### Investment Thesis
Daily investment thesis with AI hedge fund analyst signals. Integrates multiple analyst personas (Warren Buffett, Technical, Valuation, Sentiment, Risk Management) across configurable tickers.

![Thesis](docs/screenshots/thesis.png)

### Settings & Source Health
Cost dashboard with budget tracking, pipeline scheduler controls, and source health monitoring with auto-cooldown for failing feeds.

![Settings](docs/screenshots/settings.png)
![Sources](docs/screenshots/sources.png)

## Architecture

```
app/
├── integrations/       # RSS fetcher, LLM gateway, market data, weather APIs
├── pipeline/           # 5-stage pipeline: acquire → normalize → compress → rank → synthesize
├── models/             # SQLAlchemy models (Source, Article, Cluster, Brief, Timeline, etc.)
├── services/           # Business logic (clustering, ranking, embedding, cost, hedge fund)
├── routes/             # Flask blueprints (views, admin API, health, feedback)
├── jobs/               # APScheduler jobs (daily pipeline, source health)
├── templates/          # Jinja2 templates (pages, partials, macros)
└── static/             # CSS (design tokens + components), JS, images
vendor/
└── ai_hedge_fund/      # Vendored AI hedge fund multi-agent system
migrations/             # Alembic database migrations
tests/                  # pytest suite (106 tests)
```

### Pipeline

| Stage | Purpose |
|-------|---------|
| **Acquire** | Fetch RSS feeds with health tracking, retry logic, and auto-cooldown |
| **Normalize** | Extract article text, deduplicate, section-balanced selection |
| **Compress** | TF-IDF + cosine similarity clustering, representative article selection |
| **Rank** | LLM-scored relevance ranking with insight boosting |
| **Synthesize** | LLM summary generation per section, market + weather + thesis integration |

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL
- OpenAI API key

### Install

```bash
git clone https://github.com/mudit1729/daily-brief.git
cd daily-brief
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your credentials:
#   DATABASE_URL=postgresql://localhost/daily_brief
#   OPENAI_API_KEY=sk-...
#   ADMIN_API_KEY=your-admin-token
```

### Database

```bash
createdb daily_brief
flask db upgrade
```

### Seed Sources

```bash
flask shell
>>> from app.pipeline.acquire import seed_sources_from_file
>>> seed_sources_from_file('seed_sources.json')
```

### Run

```bash
flask run --port 5005
```

The scheduler runs the pipeline daily at the configured time. To trigger manually:

```bash
curl -X POST http://localhost:5005/admin/pipeline/trigger \
  -H "X-Admin-Key: your-admin-token"
```

### Tests

```bash
pytest tests/ -q
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://localhost/daily_brief` | PostgreSQL connection string |
| `OPENAI_API_KEY` | — | OpenAI API key for LLM calls |
| `ADMIN_API_KEY` | — | Token for admin API endpoints |
| `LLM_MODEL` | `gpt-5.2` | LLM model for synthesis |
| `LLM_DAILY_BUDGET_USD` | `1.00` | Daily LLM spend cap |
| `LLM_DAILY_TOKEN_BUDGET` | `100000` | Daily token limit |
| `HEDGE_FUND_TICKERS` | `AAPL,MSFT,NVDA,GOOGL,AMZN` | Tickers for hedge fund analysis |
| `SCHEDULER_ENABLED` | `true` | Enable/disable APScheduler |
| `SOURCE_FAILURE_THRESHOLD` | `3` | Consecutive failures before auto-cooldown |

## Tech Stack

- **Backend:** Flask, SQLAlchemy, Alembic, APScheduler
- **Database:** PostgreSQL
- **LLM:** OpenAI (configurable model)
- **NLP:** scikit-learn (TF-IDF clustering), numpy
- **Market Data:** yfinance
- **AI Analysts:** LangChain + LangGraph (vendored ai-hedge-fund)
- **Frontend:** Vanilla JS, CSS custom properties, Jinja2
- **Testing:** pytest (106 tests)
