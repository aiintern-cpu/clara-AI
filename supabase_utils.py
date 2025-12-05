import os
import logging
import asyncio
from typing import Set, Any, List, Optional, Dict
from pathlib import Path
from dotenv import load_dotenv

from supabase import create_client, Client as SupabaseClient


here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=here / ".env")


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase_client: Optional[SupabaseClient] = None


if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Supabase client initialized successfully.")
    except Exception as e:
        logging.error(f"❌ Supabase init failed: {e}")
        supabase_client = None
else:
    logging.warning("⚠️ SUPABASE_URL or SUPABASE_SERVICE_KEY not set; Supabase disabled.")


TABLE_PREFS = "clara_user_tool_preferences"
TABLE_HISTORY = "clara_conversation_history"


async def get_enabled_tools_for_session(user_id: str) -> Set[str]:
    """Return names of tools enabled for this user_id."""
    if not supabase_client:
        return set()
    
    def _query():
        res = supabase_client.table(TABLE_PREFS).select("tool_name, is_enabled").eq("user_id", user_id).execute()
        return getattr(res, "data", []) or []
    
    rows = await asyncio.to_thread(_query)
    enabled = set()
    for r in rows:
        if r.get("is_enabled") is True and r.get("tool_name"):
            enabled.add(r["tool_name"])
    return enabled


async def get_all_tool_status_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all tool preference records (tool_name, is_enabled) for a user.
    Returns: List of dictionaries [{'tool_name': str, 'is_enabled': bool}].
    """
    if not supabase_client:
        return []
    
    def _query():

        res = supabase_client.table(TABLE_PREFS).select("tool_name, is_enabled").eq("user_id", user_id).execute()

        return getattr(res, "data", []) or []
    
    return await asyncio.to_thread(_query)



async def set_tool_preference(user_id: str, tool_name: str, is_enabled: bool) -> None:
    """Upsert a tool preference record for a given user."""
    if not supabase_client:
        raise RuntimeError("Supabase not configured.")
    
    record = {
        "user_id": user_id,
        "tool_name": tool_name,
        "is_enabled": is_enabled
    }
    
    def _upsert():
        return supabase_client.table(TABLE_PREFS).upsert(
            record, 
            on_conflict="user_id, tool_name"
        ).execute()

    await asyncio.to_thread(_upsert)


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


def save_history_records(records: List[dict]) -> Any:
    if not supabase_client:
        return None
    try:
        return supabase_client.table(TABLE_HISTORY).insert(records).execute()
    except Exception as e:
        logging.error(f"Failed saving history: {e}")
        return None