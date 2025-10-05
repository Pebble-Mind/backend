from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage

import agent
import google_calendar

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

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
            messages.append(AIMessage(message["content"]))

    result = graph.invoke({
        "messages": messages
    })

    return jsonify({
        "from": "ai", 
        "content": result["messages"][-1].content
    })


if __name__ == "__main__":
    app.run()