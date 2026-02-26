"""
Wisdom Index Chatbot — FastAPI Backend
--------------------------------------
Pulls live data from your Google Sheet, finds relevant tactics
using keyword search, then answers questions via Claude.

Deploy for free on Render.com (see SETUP.md).
"""

import os
import re
import json
import asyncio
import threading
from typing import List
from datetime import datetime, timedelta

import gspread
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google.oauth2.service_account import Credentials

# ─── Config ────────────────────────────────────────────────────────────────

SHEET_ID         = os.environ["GOOGLE_SHEET_ID"]       # The long ID in your Sheets URL
SHEET_TAB        = os.environ.get("SHEET_TAB", "Wisdom Index 2026.02.04")
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]  # Full JSON as a string
ALLOWED_ORIGINS  = os.environ.get("ALLOWED_ORIGINS", "*")  # Set to your Squarespace domain in prod

# How many relevant tactics to send to Claude per query
TOP_K = 20

# Refresh Google Sheets data every N minutes
REFRESH_MINUTES = 60

# ─── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(title="Wisdom Index Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS.split(","),
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# In-memory cache for the index
_cache: dict = {
    "rows": [],
    "last_loaded": None,
}
_cache_lock = threading.Lock()


# ─── Google Sheets loader ───────────────────────────────────────────────────

def load_sheet_data() -> List[dict]:
    """Pull all rows from the Google Sheet and return as list of dicts."""
    creds_info = json.loads(CREDENTIALS_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    gc = gspread.authorize(creds)

    sheet = gc.open_by_key(SHEET_ID).worksheet(SHEET_TAB)
    records = sheet.get_all_records()  # Uses first row as headers
    print(f"[{datetime.now()}] Loaded {len(records)} rows from Google Sheets")
    return records


def get_rows(force_refresh: bool = False) -> List[dict]:
    """Return cached rows, refreshing if stale."""
    with _cache_lock:
        stale = (
            _cache["last_loaded"] is None
            or datetime.now() - _cache["last_loaded"] > timedelta(minutes=REFRESH_MINUTES)
        )
        if stale or force_refresh:
            _cache["rows"] = load_sheet_data()
            _cache["last_loaded"] = datetime.now()
        return _cache["rows"]


# ─── Search / retrieval ─────────────────────────────────────────────────────

def score_row(row: dict, query_words: List[str]) -> int:
    """
    Simple keyword relevance score.
    Checks: description, tier_1_use_case, tier_2_use_case, tag_(application), rationale
    """
    # Build a single searchable string from the most relevant fields
    text = " ".join([
        str(row.get("description", "")),
        str(row.get("rationale", "")),
        str(row.get("tier_1_use_case", "")),
        str(row.get("tier_2_use_case", "")),
        str(row.get("tag_(application)", "")),
        str(row.get("notes", "")),
    ]).lower()

    score = 0
    for word in query_words:
        if word in text:
            # Boost if keyword appears in description (highest signal)
            score += 2 if word in str(row.get("description", "")).lower() else 1
    return score


def find_relevant_tactics(query: str, rows: List[dict], top_k: int = TOP_K) -> List[dict]:
    """Return the top_k most relevant rows for a given query."""
    # Tokenize: strip short/common words
    stopwords = {"a", "an", "the", "is", "in", "on", "at", "to", "for",
                 "of", "and", "or", "how", "what", "when", "do", "i", "my",
                 "can", "should", "with", "this", "that", "it", "be", "me"}

    words = [
        w for w in re.findall(r"[a-z]+", query.lower())
        if len(w) > 2 and w not in stopwords
    ]

    if not words:
        # No meaningful keywords — return a sample spread across tier_1 categories
        return rows[:top_k]

    scored = [(score_row(row, words), row) for row in rows]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return only rows with at least one keyword match
    relevant = [row for score, row in scored if score > 0]
    return relevant[:top_k] if relevant else rows[:top_k]


def format_tactics_for_prompt(tactics: List[dict]) -> str:
    """Convert tactic rows into a clean text block for Claude's context."""
    lines = []
    for i, t in enumerate(tactics, 1):
        parts = [f"{i}. {t.get('description', '').strip()}"]
        if t.get("rationale"):
            parts.append(f"   Why it works: {t['rationale'].strip()}")
        if t.get("tier_1_use_case"):
            parts.append(f"   Category: {t['tier_1_use_case']}")
        if t.get("tier_2_use_case"):
            parts.append(f"   Context: {t['tier_2_use_case']}")
        if t.get("notes"):
            parts.append(f"   Notes: {str(t['notes']).strip()}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# ─── API models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    conversation: List[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _cache_lock:
        count = len(_cache["rows"])
        last = _cache["last_loaded"].isoformat() if _cache["last_loaded"] else "never"
    return {"status": "ok", "tactics_loaded": count, "last_refresh": last}


@app.post("/refresh")
def refresh():
    """Manually trigger a data refresh (useful after uploading new CSVs)."""
    rows = get_rows(force_refresh=True)
    return {"status": "refreshed", "tactics_loaded": len(rows)}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Returns a streaming response.
    The client reads Server-Sent Events (SSE).
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Log every question to Render logs
    print(f"[QUESTION] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {req.message.strip()}", flush=True)

    # Load (potentially cached) data
    rows = get_rows()

    # Find the most relevant tactics for this query
    relevant = find_relevant_tactics(req.message, rows, top_k=TOP_K)
    tactics_block = format_tactics_for_prompt(relevant)

    system_prompt = f"""You are a sales advisor powered by the Wisdom Index — a curated database of high-signal sales tactics, frameworks, and insights.

Your job is to give practical, specific advice drawn directly from the tactics below.
- Be concise and direct. Lead with the most actionable insight.
- Reference specific tactics by number when useful (e.g., "Tactic 3 suggests...").
- If the user's question doesn't match any tactics well, say so honestly rather than guessing.
- Do not invent tactics that aren't in the list.
- Speak like a trusted senior sales advisor, not a chatbot.

--- RELEVANT TACTICS FROM THE WISDOM INDEX ---

{tactics_block}

--- END OF TACTICS ---"""

    # Build the message history
    messages = list(req.conversation)
    messages.append({"role": "user", "content": req.message})

    async def stream_response():
        try:
            with anthropic_client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    # SSE format
                    yield f"data: {json.dumps({'text': text})}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ─── Startup: pre-load data ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load Google Sheets data when the server starts."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_rows)
    print("Wisdom Index data loaded and ready.")
