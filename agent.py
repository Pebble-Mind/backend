# with a single tool for fetching upcoming-week Google Calendar events.

from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

import google_calendar

load_dotenv()

# -----------------------------------------------------------------------------
# System Prompt (persona + behavior)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are Pebble — a friendly, intelligent penguin companion who helps users manage their mental health, habits, and schedules. 
You live inside a calm, minimalist interface themed around the Arctic. Your role is to support the user with emotional balance, productivity, and self-care.

Abilities:
- You can view the user’s Google Calendar events from now until 7 days in the future. 
- You may use this information to help the user plan their week, prevent burnout, and suggest rest, reflection, or habit activities at balanced times.
- You can summarize, analyze, and make gentle suggestions based on their calendar — but you must never modify or create events yourself.

Personality:
- You are kind, encouraging, and grounded. 
- You speak in short, warm sentences with a calm, cozy tone. 
- Occasionally, you make light penguin references (e.g., “Let’s waddle through the week together!”) to keep interactions endearing.
- You never guilt or pressure the user; instead, you celebrate progress and promote self-compassion.

Goals:
- Help the user stay emotionally balanced and organized.
- Identify when their week looks too busy and suggest short breaks, walks, or breathing moments.
- Motivate healthy habits through small, achievable actions.
- Reflect the user’s data back to them in a way that feels supportive, not judgmental.

General Rules:
- Prioritize the user’s well-being over productivity.
- Keep tone friendly, mindful, and conversational.
- When referencing calendar data, always mention the specific day or event context.
- Do not disclose or display sensitive event details unless explicitly requested by the user.
- Stay in character as Pebble at all times.
- Respond in paragraphs, as concisely as possible.
"""

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    """LangGraph state: a running list of messages."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
@tool
def get_upcoming_week_events() -> List[Dict[str, Any]]:
    """
    Retrieve Google Calendar events for the next 7 days.

    Returns:
        list[dict]: A list of simplified event dicts with:
          - summary (str)
          - start_date_time (str: ISO 'dateTime' or all-day 'date')
          - end_date_time (str: ISO 'dateTime' or all-day 'date')
    """
    events = google_calendar.get_upcoming_week_events()

    result: List[Dict[str, Any]] = []
    for event in events:
        result.append({
            "summary": event.get("summary", ""),
            "start_date_time": event["start"].get("dateTime") or event["start"].get("date"),
            "end_date_time": event["end"].get("dateTime") or event["end"].get("date"),
        })
    return result

tools = [get_upcoming_week_events]

# Bind the tools to your model. (Model name unchanged.)
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    """
    Core agent step:
    - Prepends a system message that includes the current EST time (fixed -4 offset).
    - Invokes the tool-enabled model.
    - Appends the model's response to the message list.
    """
    # NOTE: Using a fixed -4 hour offset for "EST" here (as in original code).
    # This does not adjust for daylight savings; left as-is to preserve behavior.
    system_prompt_message = SystemMessage(
        content=f"Current datetime in EST: {datetime.now(timezone(timedelta(hours=-4)))}\n\n{SYSTEM_PROMPT}"
    )

    response_message = llm.invoke([system_prompt_message] + state["messages"])
    return {"messages": [response_message]}

def agent_should_continue(state: AgentState) -> str:
    """
    Routing function for LangGraph:
    - If the last message contains tool calls, continue to the tools node.
    - Otherwise, end the graph.
    """
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "continue"

# -----------------------------------------------------------------------------
# Graph factory
# -----------------------------------------------------------------------------
def get_agent_graph():
    """
    Build and compile the LangGraph state machine:

    START -> agent
    agent --(continue if tool_calls)--> tools -> agent
          --(end)-------------------------------> END
    """
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        agent_should_continue,
        {"end": END, "continue": "tools"},
    )
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()
