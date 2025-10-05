from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from datetime import datetime, timezone, timedelta

import google_calendar

load_dotenv()

SYSTEM_PROMPT = f"""

Current datetime in EST: {datetime.now(timezone(timedelta(hours=-4)))}

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
- Respond in paragraphs.

"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_upcoming_week_events():

    """
    Retrieves all Google Calendar events occurring within the next 7 days.
    Returns a list of simplified event objects containing the summary, start time,
    and end time for each event.
    """

    events = google_calendar.get_upcoming_week_events()

    result = []

    for event in events:

        result.append({
            "summary": event.get("summary", ""),
            "start_date_time": event["start"].get("dateTime") or event["start"].get("date"),
            "end_date_time": event["end"].get("dateTime") or event["end"].get("date"),
        })

    return result


tools = [get_upcoming_week_events]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:

    system_prompt_message = SystemMessage(content=SYSTEM_PROMPT)

    response_message = llm.invoke([system_prompt_message] + state["messages"])
    return {"messages": [response_message]}


def agent_should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "continue"


def get_agent_graph():

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools=tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        agent_should_continue,
        {
            "end": END,
            "continue": "tools"
        }
    )
    graph_builder.add_edge("tools", "agent")

    graph = graph_builder.compile()
    return graph