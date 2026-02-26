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
from pydantic import BaseModel
from google.oauth2.service_account import Credentials

SHEET_ID         = os.environ["GOOGLE_SHEET_ID"]
SHEET_TAB        = os.environ.get("SHEET_TAB", "Wisdom Index 2026.02.04")
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]
ALLOWED_ORIGINS  = os.environ.get("ALLOWED_ORIGINS", "*")

TOP_K = 20
REFRESH_MINUTES = 60

app = FastAPI(title="Wisdom Index Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

_cache: dict = {"rows": [], "last_loaded": None}
_cache_lock = threading.Lock()


def load_sheet_data() -> List[dict]:
    creds_info = json.loads(CREDENTIALS_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SHEET_ID).worksheet(SHEET_TAB)
    records = sheet.get_all_records()
    print(f"[{datetime.now()}] Loaded {len(records)} rows from Google Sheets")
    return records


def get_rows(force_refresh: bool = False) -> List[dict]:
    with _cache_lock:
        stale = (
            _cache["last_loaded"] is None
            or datetime.now() - _cache["last_loaded"] > timedelta(minutes=REFRESH_MINUTES)
        )
        if stale or force_refresh:
            _cache["rows"] = load_sheet_data()
            _cache["last_loaded"] = datetime.now()
        return _cache["rows"]


def score_row(row: dict, query_words: List[str]) -> int:
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
            score += 2 if word in str(row.get("description", "")).lower() else 1
    return score


def find_relevant_tactics(query: str, rows: List[dict], top_k: int = TOP_K) -> List[dict]:
    stopwords = {"a", "an", "the", "is", "in", "on", "at", "to", "for",
                 "of", "and", "or", "how", "what", "when", "do", "i", "my",
                 "can", "should", "with", "this", "that", "it", "be", "me"}
    words = [
        w for w in re.findall(r"[a-z]+", query.lower())
        if len(w) > 2 and w not in stopwords
    ]
    if not words:
        return rows[:top_k]
    scored = [(score_row(row, words), row) for row in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [row for score, row in scored if score > 0]
    return relevant[:top_k] if relevant else rows[:top_k]


def format_tactics_for_prompt(tactics: List[dict]) -> str:
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


class ChatRequest(BaseModel):
    message: str
    conversation: List[dict] = []


@app.get("/health")
def health():
    with _cache_lock:
        count = len(_cache["rows"])
        last = _cache["last_loaded"].isoformat() if _cache["last_loaded"] else "never"
    return {"status": "ok", "tactics_loaded": count, "last_refresh": last}


@app.post("/refresh")
def refresh():
    rows = get_rows(force_refresh=True)
    return {"status": "refreshed", "tactics_loaded": len(rows)}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    rows = get_rows()
    relevant = find_relevant_tactics(req.message, rows, top_k=TOP_K)
    tactics_block = format_tactics_for_prompt(relevant)

    system_prompt = f"""You are a sales advisor powered by the Wisdom Index — a curated database of high-signal sales tactics, frameworks, and insights.

Your job is to give practical, specific advice drawn directly from the tactics below.
- Be concise and direct. Lead with the most actionable insight.
- NEVER reference tactic numbers (do not say "Tactic 3" or "Tactic 15" etc.) — just weave the insights naturally into your answer.
- Quote specific language from the tactics directly when it's useful, but don't attribute it to a number.
- If the user's question doesn't match any tactics well, say so honestly rather than guessing.
- Do not invent tactics that aren't in the list.
- Be sure to inlcude the rationale behind WHY a tactic is used
- If there are certain signals or cues that a user should be mindful of when deciding what tactic to use, articulate it.
- Speak like a trusted senior sales advisor, not a chatbot.

--- RELEVANT TACTICS FROM THE WISDOM INDEX ---

{tactics_block}

--- END OF TACTICS ---"""

    messages = list(req.conversation)
    messages.append({"role": "user", "content": req.message})

    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )

    return {"text": response.content[0].text}


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_rows)
    print("Wisdom Index data loaded and ready.")
