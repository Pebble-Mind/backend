from langchain_core.messages import HumanMessage

import agent
import google_calendar

def main():

    google_calendar.init()

    graph = agent.get_agent_graph()

    result = graph.invoke({
        "messages": [
            HumanMessage("what should i do right now?")
        ]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()