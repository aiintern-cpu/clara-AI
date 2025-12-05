import os
import json
import re
import tempfile
import shutil
import threading
import logging
import datetime
import random
import requests
import uuid
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from dateutil import parser as date_parser 


here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=here / ".env")

from fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from fastapi import FastAPI, Query, HTTPException
import uvicorn


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
TOKEN_DIR = os.environ.get("TOKEN_DIR", "user_tokens")
os.makedirs(TOKEN_DIR, exist_ok=True)

CREDS_PATH = os.environ.get("CREDS_PATH", "credentials.json")
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")


def _safe_filename(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9@._-]", "_", user_id)
    return f"token_user_{safe}.json"

def _token_path(user_id: str) -> str:
    return os.path.join(TOKEN_DIR, _safe_filename(user_id))

def _atomic_write(path: str, data: str):
    fd, tmp = tempfile.mkstemp(dir=TOKEN_DIR, prefix=".tmp_", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def save_creds_for_user(user_id: str, creds: Credentials):
    """Save credentials JSON to per-user token file."""
    path = _token_path(user_id)
    _atomic_write(path, creds.to_json())
    logging.info("Saved token for %s -> %s", user_id, path)

def load_creds_for_user(user_id: str) -> Optional[Credentials]:
    path = _token_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            info = json.load(f)
        creds = Credentials.from_authorized_user_info(info, SCOPES)
        return creds
    except Exception as e:
        logging.error("Failed loading creds for %s: %s", user_id, e)
        return None

def ensure_creds_refreshed(user_id: str) -> Optional[Credentials]:
    creds = load_creds_for_user(user_id)
    if not creds:
        return None
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                save_creds_for_user(user_id, creds)
                logging.info("Refreshed token for %s", user_id)
            except Exception as e:
                logging.error("Failed to refresh token for %s: %s", user_id, e)
                return None
        else:
            return None
    return creds


def get_service(api_name: str, api_version: str, user_id: str):
    """
    Returns a googleapiclient service object for given user_id.
    Will refresh tokens automatically. If no token exists, returns None.
    """
    creds = ensure_creds_refreshed(user_id)
    if not creds:
        logging.warning("No valid credentials for user: %s", user_id)
        return None
    try:
        service = build(api_name, api_version, credentials=creds)
        return service
    except Exception as e:
        logging.error("Error building service %s v%s for user %s: %s", api_name, api_version, user_id, e)
        return None


def get_youtube_service():
    """Returns a YouTube service object using the global API key."""
    if not YOUTUBE_API_KEY:
        return None
    try:
        service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        return service
    except Exception as e:
        logging.error("Error building YouTube service: %s", e)
        return None


mcp = FastMCP(name="Productivity")

@mcp.tool
def search_youtube(query: str, limit: int = 3) -> str:
    """
    Searches YouTube for videos based on a query.
    Returns a list of the top 'limit' results.
    """
    service = get_youtube_service()
    if not service:
        return "Error: YouTube service is not configured (missing API key)."
    
    try:
        search_response = service.search().list(
            q=query,
            part="snippet",
            maxResults=limit,
            type="video"
        ).execute()

        items = search_response.get("items", [])
        if not items:
            return f"No YouTube videos found for '{query}'."

        lines = []
        for item in items:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            lines.append(f"- {title}: {video_url}")
        
        return "YouTube search results:\n" + "\n".join(lines)
    
    except HttpError as he:
        return f"Google API error (YouTube Search): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error searching YouTube: {e}"

@mcp.tool
def create_calendar_event(user_id: str, topic: str, date: str, time: str, attendees: List[str]) -> str:
    service = get_service("calendar", "v3", user_id)
    if not service:
        return "Error: Could not connect to Google Calendar. No credentials for user."
    try:

        full_datetime_str = f"{date} {time}" if time else date
        

        try:
            start_dt = date_parser.parse(full_datetime_str, fuzzy=True)
        except date_parser.ParserError:
            return f"Error: Could not parse date/time '{full_datetime_str}'. Please specify clearly."

        end_dt = start_dt + datetime.timedelta(hours=1)
        attendees_list = [{"email": a.strip()} for a in attendees if "@" in a]
        

        timezone = "Asia/Kolkata" 
        event = {
            "summary": topic,
            "description": "Created by Clara.",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone},
            "attendees": attendees_list,
            "reminders": {"useDefault": True},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        link = created.get("htmlLink")
        return f"Success: The meeting is booked! View it here: {link or ''}"
    except HttpError as he:
        return f"Google API error (Calendar Create): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error creating calendar event: {e}"

@mcp.tool
def get_calendar_events(user_id: str, date: str) -> str:
    service = get_service("calendar", "v3", user_id)
    if not service:
        return "Error: Could not connect to Google Calendar. No credentials for user."
    try:
        start = datetime.datetime.fromisoformat(f"{date}T00:00:00")
        end = datetime.datetime.fromisoformat(f"{date}T23:59:59")

        time_min = start.isoformat() + "Z"
        time_max = end.isoformat() + "Z"
        events_result = service.events().list(calendarId="primary", timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime").execute()
        items = events_result.get("items", [])
        if not items:
            return f"No events found on {date}."
        lines = []
        for ev in items:
            st = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", "All day"))
            title = ev.get("summary", "No Title")
            lines.append(f"- {st}: {title}")
        return "Events on " + date + ":\n" + "\n".join(lines)
    except HttpError as he:
        return f"Google API error (Calendar Read): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error fetching events: {e}"

@mcp.tool
def add_todo_task(user_id: str, task: str, due_date: Optional[str] = None, description: Optional[str] = None) -> str:
    """
    Adds a todo task. If a due date/time is provided, it is parsed flexibly.
    """
    service = get_service("tasks", "v1", user_id)
    if not service:
        return "Error: Could not connect to Google Tasks. No credentials for user."
    try:
        body = {"title": task, "notes": description or "Created by Clara."}
        
        if due_date:
            try:
                parsed_dt = date_parser.parse(due_date, fuzzy=True, dayfirst=True)
                

                if parsed_dt.hour != 0 or parsed_dt.minute != 0 or parsed_dt.second != 0:

                    if parsed_dt.tzinfo is None:

                        local_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
                        parsed_dt = parsed_dt.replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)
                    else:
                        parsed_dt = parsed_dt.astimezone(datetime.timezone.utc)
                        
                    body["due"] = parsed_dt.isoformat().replace('+00:00', 'Z')
                    
                else:

                    body["due"] = parsed_dt.strftime("%Y-%m-%dT00:00:00.000Z")

            except date_parser.ParserError as pe:
                logging.error(f"Date Parsing Failed: {pe} for input '{due_date}'")
                return f"Error: Could not parse the due date/time '{due_date}'. Please simplify the date phrase."

        created = service.tasks().insert(tasklist="@default", body=body).execute()
        return f"Success: Task '{task}' was added to your Google Tasks list."
    except HttpError as he:

        if "Invalid value for: Invalid format" in str(he):
             return f"Error executing tool: The date or time format provided was rejected by Google Tasks. Please simplify or check the date: {due_date}"
        return f"Google API error (Tasks Create): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error adding task: {e}"

@mcp.tool
def get_todo_tasks(user_id: str, due_date: Optional[str] = None) -> str:
    service = get_service("tasks", "v1", user_id)
    if not service:
        return "Error: Could not connect to Google Tasks. No credentials for user."
    try:
        res = service.tasks().list(tasklist="@default", showCompleted=False, showHidden=False, maxResults=100).execute()
        items = res.get("items", [])
        if not items:
            return "No active tasks."
        lines = []
        for it in items:
            due = it.get("due")
            title = it.get("title", "No Title")
            

            if due_date and due:
                if due.startswith(due_date):
                    lines.append(f"- {title} (Due: {due[:10]})")
            else:
                lines.append(f"- {title}" + (f" (Due: {due[:10]})" if due else ""))
        
        return "Tasks:\n" + "\n".join(lines)
    except HttpError as he:
        return f"Google API error (Tasks Read): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error fetching tasks: {e}"

@mcp.tool
def send_email(user_id: str, to: List[str], subject: str, body: str) -> str:
    service = get_service("gmail", "v1", user_id)
    if not service:
        return "Error: Could not connect to Gmail. No credentials for user."
    try:
        from email.message import EmailMessage
        import base64
        msg = EmailMessage()
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Success: Email sent (id={sent.get('id')})"
    except HttpError as he:
        return f"Google API error (Gmail Send): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error sending email: {e}"

@mcp.tool
def create_meet_link(user_id: str, topic: str) -> str:
    service = get_service("calendar", "v3", user_id)
    if not service:
        return "Error: Could not connect to Google Calendar. No credentials for user."
    try:
        now = datetime.datetime.now()
        start = now + datetime.timedelta(minutes=5)
        end = start + datetime.timedelta(minutes=15)
        conference_request_id = f"clara-meet-{uuid.uuid4()}"
        timezone = "Asia/Kolkata"
        event = {
            "summary": topic,
            "description": "Temporary event by Clara.",
            "start": {"dateTime": start.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": timezone},
            "conferenceData": {"createRequest": {"requestId": conference_request_id, "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
        }
        created = service.events().insert(calendarId="primary", body=event, conferenceDataVersion=1).execute()
        conf = created.get("conferenceData", {})
        entry_points = conf.get("entryPoints", [])
        meet_link = None
        for ep in entry_points:
            if ep.get("entryPointType") == "video":
                meet_link = ep.get("uri")
                break
        try:

            service.events().delete(calendarId="primary", eventId=created.get("id")).execute()
        except Exception:
            pass
        if meet_link:
            return f"Success: Meet link: {meet_link}"
        return "Warning: Could not get meet link from Google response."
    except HttpError as he:
        return f"Google API error (Create Meet): {getattr(he, 'content', str(he))}"
    except Exception as e:
        return f"Error creating Meet link: {e}"
    
@mcp.tool
def add_note_to_notion(content: str) -> str:
    """
    Adds a new note (as a new page) to the 'Clara's Inbox' database in Notion.
    The 'content' will be the title of the new page.
    """
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return "Error: Notion API Key or Database ID is not configured."

    notion_version = "2022-06-28"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": notion_version,
    }
    
    payload = {
        "parent": { "database_id": NOTION_DATABASE_ID },
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": content
                        }
                    }
                ]
            }
        }
    }

    try:
        url = "https://api.notion.com/v1/pages"
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        
        page_url = response.json().get("url", "No URL returned")
        return f"Success! Note added to Notion. You can see it here: {page_url}"

    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if e.response:
            try:
                error_body = e.response.json()
                error_message = error_body.get('message', json.dumps(error_body))
                logging.error(f"Notion API error (JSON): {error_body}")
            except json.JSONDecodeError:
                error_message = e.response.text
                logging.error(f"Notion API error (raw text): {error_message}")
        else:
            logging.error(f"Notion API error (no response): {e}")
            
        return f"Error adding note to Notion: {error_message}"
    except Exception as e:
        logging.error(f"Unexpected error in Notion tool: {e}")
        return f"An unexpected error occurred: {e}"
    
@mcp.tool
def get_notes_from_notion(limit: int = 5) -> str:
    """
    Retrieves the most recent notes from the 'Clara's Inbox' database in Notion.
    It will return up to 'limit' notes, defaulting to 5.
    """
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return "Error: Notion API Key or Database ID is not configured."

    notion_version = "2022-06-28"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": notion_version,
    }

    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    
    payload = {
        "page_size": limit,
        "sorts": [
            {
                "property": "Created time",
                "direction": "descending"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])

        if not results:
            return "Your 'Clara's Inbox' in Notion is empty."

        note_list = []
        for page in results:
            note_title = page.get("properties", {}).get("Name", {}).get("title", [{}])[0].get("text", {}).get("content", "Untitled Note")
            note_list.append(f"- {note_title}")

        return "Here are your latest notes from Notion:\n" + "\n".join(note_list)

    except requests.exceptions.RequestException as e:
        error_body = e.response.json() if e.response else str(e)
        logging.error(f"Notion API error: {error_body}")
        return f"Error reading from Notion: {error_body.get('message', str(e))}"
    except Exception as e:
        logging.error(f"Unexpected error in Notion tool: {e}")
        return f"An unexpected error occurred: {e}"

@mcp.tool
def read_recent_emails(user_id: str, sender_email: str = None, limit: int = 5) -> str:
    """
    Reads the most recent emails from your Gmail inbox.
    Args:
        sender_email: (Optional) Filter by a specific sender's email address.
        limit: (Optional) Number of emails to retrieve (default 5).
    """
    service = get_service("gmail", "v1", user_id)
    if not service:
        return "Error: Could not connect to Gmail. No credentials for user."

    try:
        query = f"from:{sender_email}" if sender_email else ""
        
        results = service.users().messages().list(userId='me', q=query, maxResults=limit).execute()
        messages = results.get('messages', [])

        if not messages:
            return f"No emails found{' from ' + sender_email if sender_email else ''}."

        email_list = []
        for msg in messages:
            txt = service.users().messages().get(userId='me', id=msg['id']).execute()
            
            headers = txt['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            snippet = txt.get('snippet', 'No content preview')

            email_list.append(f"From: {sender}\nSubject: {subject}\nSnippet: {snippet}\n---")

        return "Here are the recent emails:\n" + "\n".join(email_list)

    except Exception as e:
        return f"Error reading emails: {e}"


auth_app = FastAPI(title="Clara OAuth helper")

@auth_app.get("/auth/google")
def auth_google(user_id: str = Query(..., description="user identifier (e.g. email)")):
    """
    Trigger a local OAuth flow for a given user_id. This opens a browser for consent,
    then stores credentials to user_tokens/token_user_{safe_user_id}.json
    Note: intended for local/dev usage (run_local_server opens a browser).
    """
    if not os.path.exists(CREDS_PATH):
        raise HTTPException(status_code=500, detail=f"Missing {CREDS_PATH} in project root.")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
        creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
        # Save token under safe filename
        save_creds_for_user(user_id, creds)
        return {"status": "ok", "user_id": user_id, "token_file": _token_path(user_id)}
    except Exception as e:
        logging.exception("OAuth failed for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))

def run_auth_app():

    uvicorn.run(auth_app, host="127.0.0.1", port=8081, log_level="info")


if __name__ == "__main__":
    print("Starting OAuth FastAPI on http://127.0.0.1:8081 (for /auth/google) and MCP tools on http://127.0.0.1:8080/mcp")

    auth_thread = threading.Thread(target=run_auth_app, daemon=True)
    auth_thread.start()


    try:
        mcp.run(transport="http", host="127.0.0.1", port=8080)
    except KeyboardInterrupt:
        print("Shutting down.")
