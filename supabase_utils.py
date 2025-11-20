import os
import logging
import asyncio
from typing import Optional, Set, Any, List
from pathlib import Path
from dotenv import load_dotenv

# load .env explicitly from this file's directory
here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=here / ".env")

from supabase import create_client, Client as SupabaseClient

# --- Load environment variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase_client: Optional[SupabaseClient] = None

# --- Initialize Supabase ---
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Supabase client initialized successfully.")
    except Exception as e:
        logging.error(f"❌ Supabase init failed: {e}")
        supabase_client = None
else:
    logging.warning("⚠️ SUPABASE_URL or SUPABASE_SERVICE_KEY not set; Supabase disabled.")

# --- Table names (must exist in Supabase) ---
TABLE_SESSION_MAP = "clara_session_map"
TABLE_PREFS = "clara_user_tool_preferences"
TABLE_HISTORY = "clara_conversation_history"

# --- Map session → user_id ---
async def set_session_user(session_id: str, user_id: str):
    """Link session_id to a user_id (email or Supabase UUID)."""
    if not supabase_client:
        raise RuntimeError("Supabase not configured.")
    def _upsert():
        return supabase_client.table(TABLE_SESSION_MAP).upsert(
            {"session_id": session_id, "user_id": user_id}
        ).execute()
    await asyncio.to_thread(_upsert)

async def get_user_for_session(session_id: str) -> Optional[str]:
    """Retrieve the user_id for a given session_id."""
    if not supabase_client:
        return None
    def _get():
        res = supabase_client.table(TABLE_SESSION_MAP).select("user_id").eq("session_id", session_id).single().execute()
        data = getattr(res, "data", None)
        if data:
            return data.get("user_id")
        return None
    return await asyncio.to_thread(_get)

# --- Fetch enabled tools for this session ---
async def get_enabled_tools_for_session(session_id: str) -> Set[str]:
    """Return names of tools enabled for this session_id."""
    if not supabase_client:
        return set()
    def _query():
        res = supabase_client.table(TABLE_PREFS).select("tool_name, is_enabled").eq("session_id", session_id).execute()
        return getattr(res, "data", []) or []
    rows = await asyncio.to_thread(_query)
    enabled = set()
    for r in rows:
        if r.get("is_enabled") is True and r.get("tool_name"):
            enabled.add(r["tool_name"])
    return enabled

# --- Save conversation history (async wrapper) ---
async def save_history_records_async(records: List[dict]) -> Any:
    """
    Insert chat records into Supabase history table.
    Returns the raw supabase response object (data & error attributes).
    """
    if not supabase_client:
        raise RuntimeError("Supabase not configured.")
    def _insert():
        return supabase_client.table(TABLE_HISTORY).insert(records).execute()
    return await asyncio.to_thread(_insert)

# Optional synchronous helper kept for backwards compatibility
def save_history_records(records: List[dict]) -> Any:
    if not supabase_client:
        return None
    try:
        return supabase_client.table(TABLE_HISTORY).insert(records).execute()
    except Exception as e:
        logging.error(f"Failed saving history: {e}")
        return None