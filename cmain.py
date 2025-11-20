import os
import logging
import datetime
import uuid
import asyncio
import copy
import json
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

from fastmcp import Client as MCPClient

import supabase_utils

# load .env explicitly from project folder (same directory as main.py expected)
here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=here / ".env")

# Basic config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPABASE_HISTORY_TABLE = "clara_conversation_history"
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "10"))
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8080/mcp")

# init Gemini client
gemini_client = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY missing from environment")
    gemini_client = genai.Client(api_key=gemini_api_key)
    logging.info("Gemini client initialized.")
except Exception as e:
    logging.error("Failed initializing Gemini client: %s", e)
    gemini_client = None

# MCP client (used to call tools)
mcp_client: Optional[MCPClient] = None
try:
    mcp_client = MCPClient(transport=MCP_SERVER_URL)
    logging.info("FastMCP client initialized for %s", MCP_SERVER_URL)
except Exception as e:
    logging.error("Failed initializing MCP client: %s", e)
    mcp_client = None

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logging.info("REQUEST (sess=%s): %s", session_id, req.message[:120])

    if not gemini_client:
        raise HTTPException(500, "Gemini client not available. Check GEMINI_API_KEY.")
    if not mcp_client:
        raise HTTPException(500, "MCP client not available. Is the tools server running?")

    # --- DYNAMIC USER LOGIC ---
    # 1. Look up the user's real email FIRST
    mapped_user = None
    if supabase_utils.supabase_client:
        try:
            mapped_user = await supabase_utils.get_user_for_session(session_id)
        except Exception as e:
            logging.warning("Could not map user for session %s: %s", session_id, e)
    
    # 2. Use the mapped email, or a fallback if no user is linked yet
    user_email_for_prompt = mapped_user or "unknown_user" # For the AI prompt
    user_id_for_tools = mapped_user or session_id      # For tool calls & history saving
    
    logging.info("Handling request for user: %s", user_email_for_prompt)

    # 3. Load conversation history (Supabase)
    loaded_history = []
    if supabase_utils.supabase_client:
        try:
            resp = await asyncio.to_thread(lambda: supabase_utils.supabase_client
                                          .table(SUPABASE_HISTORY_TABLE)
                                          .select("role, text")
                                          .eq("session_id", session_id)
                                          .order("created_at", desc=False)
                                          .limit(HISTORY_LIMIT)
                                          .execute())
            past = getattr(resp, "data", []) or []
            for m in past:
                role = m.get("role"); text = m.get("text")
                if role in ("user", "model") and text:
                    loaded_history.append({"role": role, "parts": [{"text": text}]})
            logging.info("Loaded %d past messages", len(loaded_history))
        except Exception as e:
            logging.warning("Could not load history: %s", e)

    tools_called_this_turn: List[str] = []

    # --- UPDATED SYSTEM PROMPT ---
    today = datetime.date.today().isoformat()
    system_instruction = (
        f"You are Clara assistant. The user's email is '{user_email_for_prompt}'. Today's date is EXACTLY {today}. "
        "You are a helpful assistant capable of managing calendar, tasks, Notion notes, and **reading/sending Gmail**. "
        "When asked about availability, call both get_calendar_events and get_todo_tasks first.\n\n"
        
        "*** CRITICAL RULES FOR HISTORY ***\n"
        "1. The conversation history provided to you is for CONTEXT ONLY (memory).\n"
        "2. PAST COMMANDS ARE ALREADY DONE. If you see a request in the history (e.g., 'send email'), assume it was successfully completed. DO NOT execute it again.\n"
        "3. ONLY execute tools for the VERY LAST message from the user.\n"
        "4. NEVER combine an old request with a new request."
    )
    # --- END UPDATED SYSTEM PROMPT ---

    async with mcp_client:
        all_tools = await mcp_client.list_tools()
        tool_names = {t.name for t in all_tools}
        logging.info("MCP exposes tools: %s", ", ".join(sorted(tool_names)))

        # attempt to load user prefs (which tools enabled)
        enabled_tools = await supabase_utils.get_enabled_tools_for_session(session_id)
        if enabled_tools:
            final_tools = [t for t in all_tools if t.name in enabled_tools]
            logging.info("Using user prefs to enable tools: %s", ", ".join(sorted(enabled_tools)))
        else:
            final_tools = all_tools

        # convert MCP tool model_dump -> Gemini tool declarations
        gemini_tool_decls = []
        # --- IMPORTANT ---
        # We need mcp_tool_dumps later, so we define it here
        mcp_tool_dumps = [t.model_dump() for t in final_tools] 
        
        for d in mcp_tool_dumps:
            if not d.get("name"): continue
            decl = {"name": d["name"], "description": d.get("description", "")}
            params_schema = d.get("inputSchema")
            if params_schema:
                params_schema = copy.deepcopy(params_schema)
                if params_schema.get("type") == "object": params_schema["type"] = "OBJECT"
                if "properties" in params_schema:
                    for prop in params_schema["properties"].values():
                        prop.pop("default", None)
                        if "type" in prop:
                            tmap = {"integer":"INTEGER","string":"STRING","number":"NUMBER","boolean":"BOOLEAN","object":"OBJECT","array":"ARRAY"}
                            prop["type"] = tmap.get(prop["type"], prop["type"].upper())
                decl["parameters"] = params_schema
            gemini_tool_decls.append(decl)

        tools_config = [types.Tool(function_declarations=gemini_tool_decls)] if gemini_tool_decls else None
        generation_config = types.GenerateContentConfig(tools=tools_config, system_instruction=system_instruction)

        # Prepare contents: history + this user message
        contents = loaded_history + [{"role": "user", "parts": [{"text": req.message}]}]

        # Generate and iterate possibly-many function calls until final response
        while True:
            response = await gemini_client.aio.models.generate_content(
                model=MODEL_NAME, contents=contents, config=generation_config
            )

            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content.parts or not candidate.content.parts[0].function_call:
                break

            function_call = candidate.content.parts[0].function_call
            fname = function_call.name
            args = function_call.args or {}
            logging.info("LLM requested tool: %s with args %s", fname, args)

            # --- START OF FIX ---
            # Smartly add user_id ONLY to tools that need it.
            
            tool_needs_user_id = False
            for tool_dump in mcp_tool_dumps:
                if tool_dump.get("name") == fname:
                    # Check if 'user_id' is a parameter in its inputSchema
                    if "user_id" in tool_dump.get("inputSchema", {}).get("properties", {}):
                        tool_needs_user_id = True
                    break

            # Only add user_id if the tool's signature has it AND the LLM didn't provide one.
            if tool_needs_user_id and "user_id" not in args:
                # We already have user_id_for_tools from the top of the function
                args["user_id"] = user_id_for_tools 
            
            # --- END OF FIX ---

            tools_called_this_turn.append(fname)

            # call the tool on MCP and capture result
            try:
                tool_result_obj = await mcp_client.call_tool(fname, args)
                tool_result = tool_result_obj.data
                logging.info("Tool %s returned: %s", fname, str(tool_result)[:200])
            except Exception as te:
                logging.error("Tool %s failed: %s", fname, te, exc_info=True)
                tool_result = f"Error executing tool {fname}: {te}"

            # Add the tool response into the conversation as a function response
            function_response_part = types.Part.from_function_response(name=fname, response={"result": tool_result})
            contents.append(candidate.content)
            contents.append(types.Content(role="function", parts=[function_response_part]))

        final_text = response.text if response.candidates else "Sorry, I couldn't generate a response."
        logging.info("FINAL (sess=%s): %s", session_id, final_text[:200])

        # persist history (attempt, best-effort)
        try:
            if supabase_utils.supabase_client:
                # Ensure type_of_tool_called is either a list or None (Postgres text[])
                tt = tools_called_this_turn or None

                user_turn = {
                    "session_id": session_id,
                    "user_id": user_id_for_tools, # Use the consistent ID
                    "role": "user",
                    "text": req.message,
                    "type_of_tool_called": tt
                }
                ai_turn = {
                    "session_id": session_id,
                    "user_id": user_id_for_tools, # Use the consistent ID
                    "role": "model",
                    "text": final_text,
                    "type_of_tool_called": tt
                }

                resp = await supabase_utils.save_history_records_async([user_turn, ai_turn])
                logging.info("Supabase insert response data=%s error=%s", getattr(resp, "data", None), getattr(resp, "error", None))
            else:
                logging.warning("Supabase client not configured; skipping history save.")
        except Exception as e:
            logging.exception("Failed saving history: %s", e)

        return {"reply": final_text, "session_id": session_id}

@app.post("/link_user")
async def link_user(session_id: str, user_id: str):
    """
    Associate the current session_id -> a concrete user_id (e.g. email).
    This is useful after user does OAuth on their browser and we know user_id.
    """
    try:
        await supabase_utils.set_session_user(session_id, user_id)
        return {"status": "ok", "session_id": session_id, "user_id": user_id}
    except Exception as e:
        raise HTTPException(500, str(e))

# Debug helper: try inserting a test record into Supabase
@app.post("/debug_insert_history")
async def debug_insert_history():
    if not supabase_utils.supabase_client:
        raise HTTPException(500, "Supabase not configured")
    rec = {
        "session_id": "dbg-session",
        "user_id": "dbg-user",
        "role": "user",
        "text": "debug test from debug_insert_history",
        "type_of_tool_called": []
    }
    resp = await supabase_utils.save_history_records_async([rec])
    return {"data": getattr(resp, "data", None), "error": getattr(resp, "error", None)}

@app.get("/")
def root():
    return {"message": "Clara main API running"}