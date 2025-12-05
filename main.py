import os
import logging
import datetime
import asyncio
import copy
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from google import genai
from google.genai import types

from fastmcp import Client as MCPClient 

import supabase_utils


here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=here / ".env")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPABASE_HISTORY_TABLE = "clara_conversation_history"
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "10"))
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8080/mcp")


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
    user_id: str 


class PreferenceRequest(BaseModel):
    user_id: str
    tool_name: str 
    is_enabled: bool 


class ToolStatus(BaseModel):
    tool_name: str
    is_enabled: bool

@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id 
    logging.info("REQUEST (user=%s): %s", user_id, req.message[:120])

    if not gemini_client:
        raise HTTPException(500, "Gemini client not available. Check GEMINI_API_KEY.")
    if not mcp_client:
        raise HTTPException(500, "MCP client not available. Is the tools server running?")

    user_email_for_prompt = user_id 
    user_id_for_tools = user_id      
    
    logging.info("Handling request for user: %s", user_email_for_prompt)


    loaded_history = []
    if supabase_utils.supabase_client:
        try:
            resp = await asyncio.to_thread(lambda: supabase_utils.supabase_client
                                          .table(SUPABASE_HISTORY_TABLE)
                                          .select("role, text")
                                          .eq("user_id", user_id) 
                                          .order("created_at", desc=False)
                                          .limit(HISTORY_LIMIT)
                                          .execute())
            past = getattr(resp, "data", []) or []
            for m in past:
                role = m.get("role"); text = m.get("text")
                if role in ("user", "model") and text:
                    loaded_history.append({"role": role, "parts": [{"text": text}]})
            logging.info("Loaded %d past messages for user %s", len(loaded_history), user_id)
        except Exception as e:
            logging.warning("Could not load history for user %s: %s", user_id, e)

    tools_called_this_turn: List[str] = []


    today = datetime.date.today().isoformat()
    system_instruction = (
        f"You are Clara, a professional and proactive personal assistant. The user is '{user_email_for_prompt}'. Today's date is {today}.\n\n"
        
        "**CAPABILITIES & PROTOCOL:**\n"
        "1.  **Goal:** Your primary goal is to efficiently execute calendar, task, and communication actions (Tasks, Calendar, Notion, Gmail).\n"
        "2.  **Tool Execution:** When a tool call is required, execute it immediately without confirming the parameters with the user.\n"
        
        "3.  **Proactivity (FINAL FIX):** ONLY call `get_calendar_events` or `get_todo_tasks` IF the user EXPLICITLY asks about their SCHEDULE, FREE TIME, AVAILABILITY, or WORKLOAD. \n"
        "4.  **Tool Exclusion:** NEVER use scheduling tools (e.g., `get_todo_tasks`, `get_calendar_events`) when the user asks about: your identity ('who am i'), your capabilities ('what can you do?'), or general chat topics.\n" 
        "5.  **Clarity:** Always acknowledge the tool execution. If successful, provide a brief, clear confirmation. If the tool reports an error (e.g., 'Error: No valid credentials' or 'Google API error'), **translate the technical error into a simple, helpful English explanation** (e.g., 'I couldn't complete the task. Please ensure the Google Tasks integration is enabled in your settings.').\n\n"
        
        "**CRITICAL HISTORY RULES:**\n"
        "1.  The conversation history is for CONTEXT ONLY (memory).\n"
        "2.  PAST COMMANDS ARE ALREADY COMPLETED. Do NOT re-execute any command seen in the history.\n"
        "3.  ONLY execute tools based on the content of the VERY LAST user message.\n"
        "4.  NEVER combine or infer context from old and new requests."
    )


    async with mcp_client:
        all_tools = await mcp_client.list_tools()
        tool_names = {t.name for t in all_tools}
        logging.info("MCP exposes tools: %s", ", ".join(sorted(tool_names)))

        enabled_tools = await supabase_utils.get_enabled_tools_for_session(user_id) 
        if enabled_tools:
            final_tools = [t for t in all_tools if t.name in enabled_tools]
            logging.info("Using user prefs to enable tools: %s", ", ".join(sorted(enabled_tools)))
        else:
            final_tools = all_tools


        gemini_tool_decls = []
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


        contents = loaded_history + [{"role": "user", "parts": [{"text": req.message}]}]


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


            tool_needs_user_id = False
            for tool_dump in mcp_tool_dumps:
                if tool_dump.get("name") == fname:
                    if "user_id" in tool_dump.get("inputSchema", {}).get("properties", {}):
                        tool_needs_user_id = True
                    break

            if tool_needs_user_id and "user_id" not in args:
                args["user_id"] = user_id_for_tools 


            tools_called_this_turn.append(fname)


            try:
                tool_result_obj = await mcp_client.call_tool(fname, args)
                tool_result = tool_result_obj.data
                logging.info("Tool %s returned: %s", fname, str(tool_result)[:200])
            except Exception as te:
                logging.error("Tool %s failed: %s", fname, te, exc_info=True)
                tool_result = f"Error executing tool {fname}: {te}"


            function_response_part = types.Part.from_function_response(name=fname, response={"result": tool_result})
            contents.append(candidate.content)
            contents.append(types.Content(role="function", parts=[function_response_part]))

        final_text = response.text if response.candidates else "Sorry, I couldn't generate a generate response."
        logging.info("FINAL (user=%s): %s", user_id, final_text[:200])


        try:
            if supabase_utils.supabase_client:
                tt = tools_called_this_turn or None

                user_turn = {
                    "session_id": user_id, 
                    "user_id": user_id, 
                    "role": "user",
                    "text": req.message,
                    "type_of_tool_called": tt
                }
                ai_turn = {
                    "session_id": user_id, 
                    "user_id": user_id, 
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

        return {"reply": final_text, "user_id": user_id}


@app.post("/set_tool_preference")
async def set_tool_preference_api(req: PreferenceRequest):
    """
    Sets a single tool's enabled/disabled state for a specific user.
    """
    try:
        logging.info("Setting preference for user %s: tool=%s, enabled=%s", 
                     req.user_id, req.tool_name, req.is_enabled)
        
        await supabase_utils.set_tool_preference(
            user_id=req.user_id, 
            tool_name=req.tool_name, 
            is_enabled=req.is_enabled
        )
        
        return {
            "status": "ok", 
            "message": f"Preference for {req.tool_name} set to {req.is_enabled}"
        }
    except Exception as e:
        logging.error("Failed to set tool preference: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_all_tool_status", response_model=List[ToolStatus])
async def get_all_tool_status_api(user_id: str = Query(..., description="The ID of the user.")) -> List[Dict[str, Any]]:
    """
    Retrieves the enabled/disabled status of ALL tools for a given user.
    This is used to initialize the front-end toggle switch states.
    """
    try:

        db_results = await supabase_utils.get_all_tool_status_for_user(user_id)
        

        if not mcp_client:
             raise HTTPException(500, "MCP client not available to fetch tool definitions.")
        

        async with mcp_client:
            all_mcp_tools = await mcp_client.list_tools()
            all_tool_names = {t.name for t in all_mcp_tools}
        

        user_prefs = {r['tool_name']: r['is_enabled'] for r in db_results}
        
        final_status_list = []
        

        for tool_name in sorted(list(all_tool_names)):

            is_enabled = user_prefs.get(tool_name, True) 
            final_status_list.append(ToolStatus(tool_name=tool_name, is_enabled=is_enabled))
            

        return final_status_list
        
    except Exception as e:
        logging.error("Failed to retrieve tool statuses: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Clara running"}