# PebbleMind Backend

A lightweight Flask backend that powers **Pebble**, a planning assistant. It:
- Chats via a LangGraph agent (LangChain + OpenAI).
- Reads Google Calendar events for the next 7 days.
- Generates a concise Markdown study plan from a list of tasks.

## Run It

1) **Install deps**
```bash
pip install -r requirements.txt
```

2) **Set env**
```
# .env
OPENAI_API_KEY=your_openai_api_key
```

3) **Google OAuth**
- Create a Google Cloud OAuth Desktop client and download credentials.
- Save as `credentials.json` in the project root.
- First run will open a browser and create `token.json`.

4) **Google OAuth**
- Create a Google Cloud OAuth Desktop client and download credentials.
- Save as `credentials.json` in the project root.
- First run will open a browser and create `token.json`.