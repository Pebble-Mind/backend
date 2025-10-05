from flask import Flask, request
from langchain_core.messages import HumanMessage

import agent
import google_calendar

app = Flask(__name__)

google_calendar.init()
graph = agent.get_agent_graph()

@app.route("/api/chat-with-pebble", methods=["POST"])
def chat_with_pebble():

    data = request.get_json()

    messages = []

    for message in data["messages"]:
        if message["from"] == "user":
            messages.append(HumanMessage(message["content"]))
        else:
            messages.append( (message["content"]))

    result = graph.invoke({
        "messages": messages
    })




if __name__ == "__main__":
    app.run(debug=True)