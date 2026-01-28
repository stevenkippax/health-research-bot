# Viral Health Post Idea Bot

A production-ready bot that automatically generates viral health post ideas for Instagram pages **aging.ai** and **avoidaging**.

## What It Does

Twice daily, this bot:
1. **Pulls fresh content** from curated health/longevity sources (PubMed, WHO, CDC, bioRxiv, ScienceDaily, etc.)
2. **Scores content against your winner corpus** using embeddings similarity
3. **Matches viral primitives** (STUDY_SHOCK_COMPARISON, FOOD_SYMPTOM_BENEFIT, etc.)
4. **Generates ALL CAPS slide copy** matching your top-performing style
5. **Creates supporting captions** with source attribution
6. **Exports to Google Sheets** with viral scores and primitives
7. **Maintains state** to avoid repeating topics

### V3 Viral Likeness Pipeline (NEW)

The V3 pipeline introduces:
- **Winner corpus scoring**: Computes embedding similarity to your top-performing headlines
- **Viral primitives matching**: Deterministic pattern matching before LLM generation
- **ALL CAPS slide copy**: Instagram-ready on-image text matching winner style
- **Caption generation**: Supporting context with source attribution
- **Weighted final scoring**: `0.45*viral_likeness + 0.35*primitive + 0.20*clarity`

---

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Google Cloud service account (for Sheets API)
- Railway account (for hosting)

### Local Development

```bash
# Clone and enter directory
cd viral_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and fill in your values
cp .env.example .env
# Edit .env with your API keys

# Run once manually
python -m viral_bot run

# Or run with backfill
python -m viral_bot backfill --days 2

# Export to CSV
python -m viral_bot export_csv
```

---

## Google Sheets Setup (Step by Step)

### 1. Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project**
3. Name it something like "Viral Bot"
4. Click **Create**

### 2. Enable the Google Sheets API
1. In your project, go to **APIs & Services** → **Library**
2. Search for "Google Sheets API"
3. Click on it and click **Enable**

### 3. Create a Service Account
1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **Service Account**
3. Name it "viral-bot-sheets" (or similar)
4. Click **Create and Continue**
5. For role, select **Editor** (or skip this step)
6. Click **Done**

### 4. Get the Service Account Key (JSON)
1. Click on the service account you just created
2. Go to the **Keys** tab
3. Click **Add Key** → **Create new key**
4. Select **JSON** and click **Create**
5. A JSON file will download - **keep this safe!**

### 5. Share Your Google Sheet
1. Create a new Google Sheet (or use existing)
2. Copy the **Sheet ID** from the URL:
   ```
   https://docs.google.com/spreadsheets/d/SHEET_ID_IS_HERE/edit
   ```
3. Click **Share** button
4. Add the service account email (looks like `viral-bot-sheets@your-project.iam.gserviceaccount.com`)
5. Give it **Editor** access

### 6. Set Up Environment Variables
The JSON key file contents go into the `GOOGLE_SERVICE_ACCOUNT_JSON` environment variable.

You can either:
- **Option A**: Paste the entire JSON as one line (escape quotes)
- **Option B**: Base64 encode it: `cat your-key.json | base64 -w 0`

---

## Railway Deployment

### 1. Create Railway Project
1. Go to [Railway.app](https://railway.app/)
2. Click **New Project** → **Deploy from GitHub repo**
3. Connect your repository

### 2. Add Environment Variables
In Railway dashboard, go to your service → **Variables** tab and add:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `GOOGLE_SHEET_ID` | The ID from your Google Sheet URL |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The full JSON key (or base64 encoded) |
| `GOOGLE_SERVICE_ACCOUNT_BASE64` | Set to `true` if you base64 encoded the JSON |
| `DATABASE_URL` | Leave empty for SQLite, or use Railway Postgres |
| `LOG_LEVEL` | `INFO` (or `DEBUG` for more detail) |
| `FRESHNESS_HOURS` | `48` (default) |
| `MAX_OUTPUTS_PER_RUN` | `5` (default) |

### 3. Set Up Persistent Storage (for SQLite)
1. In Railway, click **New** → **Volume**
2. Mount it at `/app/data`
3. Set `DATABASE_PATH=/app/data/viral_bot.db` in variables

### 4. Configure Cron Schedule
In Railway, add a **Cron Job**:
- **Schedule**: `0 9,18 * * *` (runs at 09:00 and 18:00 UTC)
- **Command**: `python -m viral_bot run`

Alternatively, the bot includes a built-in scheduler you can enable:
```
ENABLE_SCHEDULER=true
SCHEDULE_HOURS=9,18
```

### 5. Deploy
Railway will auto-deploy when you push to your repository.

---

## Winner Corpus Setup (V3 Pipeline)

The V3 pipeline requires a Google Sheet containing your top-performing headlines.

### 1. Create Winner Headlines Sheet
1. Create a new Google Sheet (or use existing)
2. In column A, add your top-performing headlines (one per row)
3. The first row can be a header ("Headline") - it will be skipped
4. Copy the Sheet ID from the URL

### 2. Share with Service Account
Share the winners sheet with your service account email (same as output sheet).

### 3. Set Environment Variable
```bash
WINNERS_SHEET_ID=1ShjETp_vVEZWSdYrEwa3GFugwPEXPMHuRDXoyekO_RA
```

### 4. Local CSV Fallback (Dev)
For local development, you can also provide a CSV file:
```bash
# Place file in project root
AgingAI Output.csv
```
The CSV should have headlines in the first column.

---

## Environment Variables Reference

```bash
# Required
OPENAI_API_KEY=sk-...
GOOGLE_SHEET_ID=1abc123...
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}

# Optional - Google
GOOGLE_SERVICE_ACCOUNT_BASE64=false
GOOGLE_SHEET_TAB_NAME=PostIdeas

# Optional - Database
DATABASE_URL=                    # PostgreSQL URL (leave empty for SQLite)
DATABASE_PATH=./data/viral_bot.db  # SQLite path

# Optional - Bot Config
FRESHNESS_HOURS=48              # Only fetch items from last N hours
MAX_OUTPUTS_PER_RUN=5           # Max post ideas per run
MIN_VIRALITY_SCORE=40           # Minimum score to consider
DEDUP_SIMILARITY_THRESHOLD=0.85 # Semantic similarity threshold

# V3 Viral Likeness Settings
WINNERS_SHEET_ID=1ShjETp...     # Sheet ID for winner headlines corpus
WINNERS_RANGE=A:A               # Range to read winner headlines from
WINNERS_REFRESH_HOURS=24        # Hours between winner corpus refreshes
VIRAL_SIM_THRESHOLD=0.78        # Min cosine similarity to winners (0-1)
VIRAL_SIM_WEIGHT=0.45           # Weight for viral likeness in final score
PRIMITIVE_WEIGHT=0.35           # Weight for primitive score in final score
LLM_QUALITY_WEIGHT=0.20         # Weight for LLM quality in final score
PRIMITIVE_THRESHOLD=40          # Min primitive score (0-100) to pass
BANNED_NEWSWORDS=objectives,stakeholders,framework,...
ALLOW_TIER_C=true               # Allow Tier C (raw research) sources

# Optional - Server
PORT=8000                       # Health check server port
ENABLE_HEALTH_SERVER=true       # Enable FastAPI server

# Optional - Scheduler
ENABLE_SCHEDULER=false          # Enable built-in scheduler
SCHEDULE_HOURS=9,18             # Hours to run (UTC)

# Optional - Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
```

---

## CLI Commands

```bash
# Run V3 pipeline (viral likeness scoring)
python -m viral_bot run --v3

# Run V3 pipeline in dry-run mode
python -m viral_bot run --v3 --dry-run

# Run V2 pipeline (legacy story compression)
python -m viral_bot run

# Debug a single URL through the pipeline
python -m viral_bot debug-item URL --json-output

# Backfill: fetch older content
python -m viral_bot backfill --days 3

# Export latest outputs to CSV
python -m viral_bot export_csv --output latest_posts.csv

# Start the health server only (for monitoring)
python -m viral_bot serve

# Check database stats
python -m viral_bot stats

# Show current configuration
python -m viral_bot config

# List content sources with tiers
python -m viral_bot sources
```

---

## API Endpoints (Health Server)

When `ENABLE_HEALTH_SERVER=true`:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check (returns `{"status": "ok"}`) |
| `GET /latest` | Returns last run's outputs as JSON |
| `GET /stats` | Database statistics |
| `POST /feedback` | Submit feedback for an output |

### Feedback Endpoint Example
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "output_id": 123,
    "likes": 1500,
    "shares": 200,
    "saves": 450,
    "notes": "Performed very well!"
  }'
```

---

## Output Format (Google Sheet Columns)

| Column | Description |
|--------|-------------|
| run_timestamp_utc | When the bot ran |
| archetype | Category (STUDY_STAT, WARNING_RISK, etc.) |
| source_name | Where the content came from |
| source_url | Link to original |
| published_at | When source was published |
| extracted_claim | The key factual claim |
| virality_score | 0-100 score |
| confidence | 0.0-1.0 confidence |
| image_headline | The viral headline for the image |
| image_suggestion | What the image should show |
| why_it_will_work | Bullets explaining virality |
| status | NEW (changes to USED when posted) |
| feedback_likes | (fill in later) |
| feedback_shares | (fill in later) |
| feedback_saves | (fill in later) |
| feedback_notes | (fill in later) |

---

## Viral Primitives (V3)

The V3 pipeline uses deterministic primitive matching before LLM generation:

| Primitive | Description | Example |
|-----------|-------------|---------|
| `STUDY_SHOCK_COMPARISON` | Study + shocking number/comparison | "EQUIVALENT TO SMOKING 5 CIGARETTES A DAY" |
| `SIMPLE_HACK_PAIN_RELIEF` | Small physical action → symptom relief | "ROLL A TENNIS BALL UNDER YOUR FOOT FOR 2 MINUTES" |
| `FOOD_SYMPTOM_BENEFIT` | Specific food → health benefit | "EATING WALNUTS REDUCES INFLAMMATION BY 30%" |
| `PARENT_CHILD_BIO` | Baby/parent + biological markers | "SKIN-TO-SKIN CONTACT REDUCES INFANT CORTISOL BY 42%" |
| `AUTHORITY_CLASSIFICATION` | WHO/FDA/IARC classification | "NOW CLASSIFIED AS GROUP 1 CARCINOGEN BY THE WHO" |
| `CULTURE_CONTROVERSY` | Health-relevant controversy | "NEW STUDY CHALLENGES DECADES OF NUTRITION ADVICE" |
| `TIME_REVERSAL` | Age reversal, longevity | "COULD ADD 3.7 YEARS TO YOUR LIFE" |
| `BODY_PART_SPECIFIC` | Specific body part + improvement | "YOUR BRAIN SHRINKS 0.5% PER YEAR AFTER 30" |

---

## Archetypes (V2/Legacy)

The V2 pipeline categorizes content into these archetypes:

| Archetype | Example |
|-----------|---------|
| `NEWS_POLICY` | "WHO just announced..." |
| `STUDY_STAT` | "Study of 50,000 people finds..." |
| `WARNING_RISK` | "This common habit linked to 40% higher risk..." |
| `SIMPLE_HABIT` | "One daily habit that adds years to your life" |
| `IF_THEN` | "If you do X, here's what happens to Y" |
| `COUNTERINTUITIVE` | "The surprising food that helps your brain" |
| `HUMAN_INTEREST` | "This 95-year-old's secret to longevity" |

---

## Content Sources

The bot fetches from:

**Research/Preprints:**
- PubMed (aging, longevity queries)
- bioRxiv (neuroscience, aging, metabolism)
- medRxiv (public health, aging)

**Public Health:**
- WHO news feed
- CDC newsroom
- NIH news
- NIA (National Institute on Aging)

**Credible News:**
- BBC Health
- The Guardian Health
- Reuters Health
- STAT News

---

## Adding Feedback Later (Training Loop)

1. After posts perform, update the Google Sheet with actual metrics
2. Use the feedback endpoint or run:
   ```bash
   python -m viral_bot import_feedback --sheet
   ```
3. The bot stores feedback in the database for future model improvements

---

## Troubleshooting

### "No items found"
- Check if sources are accessible
- Increase `FRESHNESS_HOURS` temporarily
- Run with `LOG_LEVEL=DEBUG`

### "Google Sheets error"
- Verify sheet is shared with service account email
- Check that Sheet ID is correct
- Ensure JSON key is valid

### "OpenAI rate limit"
- The bot has built-in exponential backoff
- Consider reducing concurrent requests
- Check your OpenAI usage limits

### "Duplicate content"
- The bot deduplicates by URL and semantic similarity
- Adjust `DEDUP_SIMILARITY_THRESHOLD` (0.0-1.0)
- Check the `content_items` table for duplicates

---

## Project Structure

```
viral_bot/
├── src/viral_bot/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── main.py              # Main orchestration
│   ├── scheduler.py         # Cron/scheduler logic
│   ├── config.py            # Pydantic settings
│   ├── db.py                # Database models & operations
│   ├── dedupe.py            # Deduplication (URL + embeddings)
│   ├── openai_eval.py       # Virality predictor
│   ├── openai_generate.py   # Headline & image suggestion
│   ├── sheets.py            # Google Sheets integration
│   ├── server.py            # FastAPI health endpoints
│   ├── logging_conf.py      # Structured logging
│   └── sources/
│       ├── __init__.py
│       ├── registry.py      # Source registry
│       ├── base.py          # Base parser class
│       ├── rss.py           # RSS feed parser
│       ├── pubmed.py        # PubMed API
│       └── html.py          # HTML scraper (fallback)
├── data/                    # SQLite database (gitignored)
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Procfile                 # Railway/Heroku
├── railway.json             # Railway config
└── README.md
```

---

## License

MIT License - Use freely for your projects.
