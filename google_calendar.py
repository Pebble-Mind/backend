# Handles Google Calendar authentication and event retrieval.

import os.path
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
service = None  # global Calendar API service instance


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def init():
    """
    Initialize the Google Calendar API service.

    - Loads existing OAuth token from 'token.json' if available.
    - Refreshes expired credentials or triggers a new OAuth flow using 'credentials.json'.
    - Saves new/updated credentials to 'token.json' for reuse.

    Sets the global `service` variable.
    """
    global service

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    creds = None

    # Load token if it exists
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # Handle missing or invalid credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save refreshed/new credentials
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # Build Calendar API service client
    service = build("calendar", "v3", credentials=creds)


# -----------------------------------------------------------------------------
# Event Retrieval
# -----------------------------------------------------------------------------
def get_upcoming_week_events():
    """
    Retrieve events from the user's primary Google Calendar for the upcoming week.

    Returns:
        list[dict]: A list of event dictionaries, each including summary, start, and end info.
    """
    if service is None:
        raise RuntimeError("Google Calendar service not initialized. Call init() first.")

    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=7)).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    return events_result.get("items", [])
