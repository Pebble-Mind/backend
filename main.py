from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import agent
import google_calendar

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.url_map.strict_slashes = False  # treat /path and /path/ the same
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Initialize integrations
google_calendar.init()
graph = agent.get_agent_graph()


# -----------------------------------------------------------------------------
# Helpers (no behavior change; just here for readability)
# -----------------------------------------------------------------------------
def _lc_messages_from_payload(payload: dict):
    """
    Convert front-end message payload into LangChain message objects.

    Expected payload shape:
    {
        "messages": [
            {"from": "user" | "ai", "content": "text"},
            ...
        ]
    }
    """
    messages = []
    for message in payload["messages"]:
        if message["from"] == "user":
            messages.append(HumanMessage(message["content"]))
        else:
            messages.append(AIMessage(message["content"]))
    return messages


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/api/chat-with-pebble", methods=["POST"])
def chat_with_pebble():
    """
    Chat endpoint.
    - Accepts a list of prior messages from the front end
    - Invokes the agent graph
    - Returns the last AI message for rendering
    """
    data = request.get_json()
    messages = _lc_messages_from_payload(data)

    result = graph.invoke({"messages": messages})

    return jsonify({
        "from": "ai",
        "content": result["messages"][-1].content
    })


@app.route("/api/get-upcoming-week-events")
def get_upcoming_week_events():
    """
    Calendar endpoint.
    - Returns upcoming week events from Google Calendar
    """
    return jsonify({
        "events": google_calendar.get_upcoming_week_events()
    })


@app.route("/api/create-study-plan", methods=["POST"])
def create_study_plan():
    """
    Study plan endpoint.
    - Accepts a list of tasks
    - Sends a system prompt to the agent to produce a Markdown plan
    - Returns the plan as { studyPlan: string }
    """
    data = request.get_json()

    # System prompt kept as-is (content/logic unchanged)
    system_prompt = f"""
You are Pebble, a planning assistant. Create a practical study plan.

TASK
Build a day-by-day plan (today → Sunday) allocating focused work sessions and short breaks.

SCHEDULING RULES
1) Always schedule work BEFORE its due date.
2) Order of scheduling within each day:
   - Higher priority first, then earlier due dates, then remaining tasks.
3) Alternate motivation: do not schedule two low-motivation items back-to-back; interleave with higher-motivation items when possible.
4) Breaks: insert brief breaks between sessions and alternate types among: meditation, exercise, reading, journaling.
5) Session sizing: break tasks into focused blocks of ~50 minutes (or ~25 if hours are small) with a 10–15 minute break between blocks, until the task’s estimated hours are fully allocated across the week.
6) Reasonable day limits: avoid overloading any single day. If spillover is needed, continue on the next day (still before due when possible).
7) No scheduling in the past (relative to “now” above). If “today” has already partly elapsed, start from the next reasonable slot.
8) Format times in 12-hour local time (e.g., 3:00 PM–3:50 PM). No seconds. Round to 5-minute increments.
9) Keep it concise and actionable. No extra commentary outside the plan.
10) Assume that the user is unavailable to work during sleep times (10:00 PM-7:00 AM) and school times (7:00 AM-5:00 PM)
11) If there is no work on a certain day, say something like "Woohoo, no work!" for weekdays and "Enjoy the weekend!" for weekends.

OUTPUT FORMAT (Markdown only)
For each day from today to Sunday, output:
- A bullet point with the weekday and date: `### Monday (YYYY-MM-DD)`
- Bulleted items alternating WORK and BREAK. For WORK items use:
  `- [3:00 PM–3:50 PM] WORK — <Task Name>
For BREAK items use:
  `- [3:50 PM–4:05 PM] BREAK — <Meditation|Exercise|Reading|Journaling>`
After the bullet points, put a double newline at the end

EXAMPLE (one day only)
### **Monday (2025-10-06)**
- [3:00 PM–3:50 PM] WORK — AP Euro notes
- [3:50 PM–4:05 PM] BREAK — Meditation
- [4:05 PM–4:55 PM] WORK — Math problem set
- [4:55 PM–5:10 PM] BREAK — Reading

DATA
{str(data["tasks"])}
""".strip()

    result = graph.invoke({
        "messages": [SystemMessage(system_prompt)]
    })

    return jsonify({
        "studyPlan": result["messages"][-1].content
    })


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
