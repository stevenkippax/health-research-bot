# Railway Deployment Guide (Idiot-Proof Edition)

Follow these steps EXACTLY. Don't skip any step.

---

## Prerequisites (Do These First!)

### 1. Get an OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. **SAVE IT SOMEWHERE** - you can't see it again!

### 2. Set Up Google Sheets (This is the tricky part)

#### Step 2a: Create a Google Cloud Project

1. Go to https://console.cloud.google.com/
2. Click the project dropdown at the top → "New Project"
3. Name it anything (e.g., "viral-bot")
4. Click "Create" and wait

#### Step 2b: Enable Google Sheets API

1. In Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for "Google Sheets API"
3. Click on it → Click "Enable"
4. Also search for "Google Drive API" and enable it too

#### Step 2c: Create a Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Name it anything (e.g., "viral-bot-sheets")
4. Click "Create and Continue"
5. Skip the optional steps, click "Done"

#### Step 2d: Get the JSON Key

1. Click on your new service account (the email address)
2. Go to the "Keys" tab
3. Click "Add Key" → "Create new key"
4. Choose "JSON" → Click "Create"
5. **A file will download - DON'T LOSE THIS!**
6. Open the file in a text editor - you'll need the contents

#### Step 2e: Share Your Google Sheet

1. Create a new Google Sheet (or use an existing one)
2. Copy the Sheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/THIS_IS_THE_SHEET_ID/edit
   ```
3. Click "Share" in the top right
4. Paste the service account email (looks like `something@something.iam.gserviceaccount.com`)
5. Give it "Editor" access
6. Click "Send"

---

## Deploy to Railway

### Step 1: Create a Railway Account

1. Go to https://railway.app/
2. Sign up (GitHub login is easiest)

### Step 2: Create a New Project

1. Click "New Project"
2. Click "Deploy from GitHub repo"
3. Connect your GitHub account if prompted
4. Select the `health-research-bot` repository
5. Click "Deploy Now"

### Step 3: Add Environment Variables (CRITICAL!)

1. Click on your newly created service
2. Go to the "Variables" tab
3. Click "New Variable" and add each of these:

| Variable Name | Value |
|--------------|-------|
| `OPENAI_API_KEY` | Your OpenAI key (starts with `sk-`) |
| `GOOGLE_SHEET_ID` | The Sheet ID from your Google Sheet URL |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The ENTIRE contents of your downloaded JSON file |
| `DATABASE_URL` | Leave empty for now (see Step 4 for PostgreSQL) |
| `LOG_LEVEL` | `INFO` |
| `LOG_JSON` | `true` |
| `PORT` | `8000` |
| `ENABLE_SCHEDULER` | `true` |
| `SCHEDULE_HOURS` | `9,18` (runs at 9 AM and 6 PM UTC) |

**IMPORTANT FOR GOOGLE_SERVICE_ACCOUNT_JSON:**
- Copy the ENTIRE JSON file contents
- It should start with `{` and end with `}`
- Include the quotes and everything
- Make sure there are no extra spaces or line breaks

### Step 4: Add PostgreSQL (Recommended for Production)

SQLite doesn't persist on Railway. Add PostgreSQL:

1. In your Railway project, click "New"
2. Click "Database" → "PostgreSQL"
3. Wait for it to provision
4. Click on the PostgreSQL service
5. Go to "Variables" tab
6. Copy the `DATABASE_URL` value
7. Go back to your bot service
8. Add a new variable: `DATABASE_URL` = the URL you copied

### Step 5: Verify Deployment

1. Click on your service
2. Go to "Deployments" tab
3. Click on the latest deployment
4. Click "View Logs"
5. You should see:
   ```
   server_starting
   scheduler_enabled
   ```
6. If you see errors, check your environment variables!

### Step 6: Test the Health Endpoint

1. Go to "Settings" tab
2. Under "Networking", click "Generate Domain"
3. Copy your URL (looks like `your-app.up.railway.app`)
4. Visit `https://your-app.up.railway.app/health` in your browser
5. You should see:
   ```json
   {"status":"ok","timestamp":"...","version":"1.0.0"}
   ```

---

## Troubleshooting

### "Invalid Google credentials" error

- Make sure `GOOGLE_SERVICE_ACCOUNT_JSON` contains the ENTIRE JSON file
- Check for missing quotes or brackets
- Try base64 encoding the JSON:
  1. Go to https://www.base64encode.org/
  2. Paste your JSON
  3. Click "Encode"
  4. Copy the result
  5. Set `GOOGLE_SERVICE_ACCOUNT_JSON` to the encoded value
  6. Add `GOOGLE_SERVICE_ACCOUNT_BASE64=true`

### "Permission denied" on Google Sheets

- Make sure you shared the sheet with the service account email
- Check that the service account has "Editor" access
- Verify the Sheet ID is correct

### "OPENAI_API_KEY not found" error

- Double-check you added the variable
- Make sure there are no extra spaces
- Verify your API key is valid at https://platform.openai.com/api-keys

### Bot runs but nothing appears in Google Sheets

- Check the logs for errors
- Verify `GOOGLE_SHEET_ID` is correct
- Make sure the tab name matches `GOOGLE_SHEET_TAB_NAME` (default: "PostIdeas")
- Check that FRESHNESS_HOURS isn't filtering out all content

### Database errors

- If using SQLite: Add PostgreSQL (Step 4 above)
- If using PostgreSQL: Check that `DATABASE_URL` is set correctly

### Deployment keeps failing

Check the build logs for errors:
1. Go to "Deployments"
2. Click the failed deployment
3. Click "Build Logs"
4. Look for red error text

---

## How to Run Manually

Want to trigger a run without waiting for the schedule?

1. Get your Railway URL (from Settings → Networking)
2. Send a POST request to `/run`:
   ```bash
   curl -X POST https://your-app.up.railway.app/run
   ```
3. Check the logs to see it running

---

## Understanding the Schedule

The bot runs automatically based on `SCHEDULE_HOURS`:

- `9,18` = Runs at 9:00 AM and 6:00 PM **UTC**
- To change the schedule, update the `SCHEDULE_HOURS` variable
- Times are in 24-hour format, UTC timezone

To convert to your timezone:
- UTC 9:00 = 4:00 AM EST / 1:00 AM PST
- UTC 18:00 = 1:00 PM EST / 10:00 AM PST

---

## Costs

### Railway
- Free tier: $5/month credit (usually enough for this bot)
- After free tier: ~$5-10/month typical usage

### OpenAI
- GPT-4o: ~$0.01-0.05 per run (depending on content volume)
- Embeddings: Very cheap (~$0.0001 per article)

### Google Sheets
- Free!

---

## Quick Reference: All Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `GOOGLE_SHEET_ID` | Yes | - | Google Sheet ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Yes | - | Service account JSON |
| `GOOGLE_SERVICE_ACCOUNT_BASE64` | No | `false` | Set `true` if JSON is base64 |
| `GOOGLE_SHEET_TAB_NAME` | No | `PostIdeas` | Tab name in sheet |
| `DATABASE_URL` | No | - | PostgreSQL URL |
| `FRESHNESS_HOURS` | No | `48` | Only fetch recent content |
| `MAX_OUTPUTS_PER_RUN` | No | `5` | Max ideas per run |
| `MIN_VIRALITY_SCORE` | No | `40` | Min score threshold |
| `ENABLE_SCHEDULER` | No | `false` | Enable auto-scheduling |
| `SCHEDULE_HOURS` | No | `9,18` | Hours to run (UTC) |
| `LOG_LEVEL` | No | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `LOG_JSON` | No | `false` | JSON log format |
| `PORT` | No | `8000` | Server port |

---

## Need Help?

1. Check the logs on Railway
2. Make sure all required variables are set
3. Verify Google Sheets permissions
4. Check that your OpenAI key has credits
