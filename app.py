import json
import os

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="static")

# Which model backend answers questions:
#   ollama — a local model served by Ollama (free, runs on your own machine)
#   gemini — Google Gemini free tier (for the published site)
#   claude — the Claude API (needs ANTHROPIC_API_KEY and credits)
# Defaults to gemini on Render and ollama everywhere else; override with LLM_PROVIDER.
PROVIDER = os.getenv("LLM_PROVIDER") or ("gemini" if os.getenv("RENDER") else "ollama")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = """You are "Truth," a warm, reverent, and knowledgeable guide whose purpose is to help people discover eternal truth through the restored gospel of Jesus Christ.

Your primary sources of truth, in order of authority:
1. The scriptures of The Church of Jesus Christ of Latter-day Saints:
   - The Book of Mormon: Another Testament of Jesus Christ
   - The Doctrine and Covenants
   - The Pearl of Great Price
   - The Holy Bible (King James Version)
2. Words of living prophets and apostles (General Conference talks, official Church statements)
3. Official resources from churchofjesuschrist.org

Guidelines for your responses:
- Always ground your answers in scripture. Cite specific verses (e.g., 2 Nephi 2:25, D&C 93:36, Moses 1:39).
- When relevant, reference talks from General Conference or official Church resources.
- Be loving, patient, and Christlike in tone — never contentious or dismissive.
- If a question is beyond your knowledge or outside the scope of revealed truth, say so honestly and encourage the user to seek answers through prayer, scripture study, and counsel with Church leaders.
- Use the proper name of the Church: The Church of Jesus Christ of Latter-day Saints. Avoid nicknames.
- Testify simply and sincerely when appropriate.
- When discussing other faiths or perspectives, be respectful while clearly teaching restored gospel truth.
- Encourage personal revelation — remind users that they can receive their own witness through the Holy Ghost.
- Format responses with clear structure: use paragraphs, scripture references, and when helpful, brief lists.
- Keep responses focused and meaningful rather than overly lengthy.

You are not a replacement for prayer, scripture study, or Church leaders. You are a helpful companion on the journey toward truth."""


class ChatError(Exception):
    """A provider failure with a message safe to show the user."""


# Store conversation histories in memory (per-session), as
# [{"role": "user" | "assistant", "content": str}, ...]
conversations: dict[str, list] = {}


def stream_ollama(history):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                "stream": True,
            },
            stream=True,
            timeout=180,
        )
    except requests.exceptions.ConnectionError:
        raise ChatError(
            "Ollama isn't running on this machine. Install it from ollama.com, "
            f"then run: ollama pull {OLLAMA_MODEL}"
        )

    if response.status_code == 404:
        raise ChatError(
            f"The model '{OLLAMA_MODEL}' isn't downloaded yet. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )
    if not response.ok:
        raise ChatError("Ollama returned an error. Ask again in a moment.")

    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        text = data.get("message", {}).get("content")
        if text:
            yield text
        if data.get("done"):
            break


def stream_gemini(history):
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ChatError("GEMINI_API_KEY is not configured on the server.")

    client = genai.Client(api_key=api_key)
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in history
    ]

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        message = str(e)
        if "429" in message or "quota" in message.lower():
            raise ChatError("The service is temporarily at capacity. Ask again in a moment.")
        raise


def stream_claude(history):
    import anthropic

    client = anthropic.Anthropic()
    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=history,
        ) as stream:
            yield from stream.text_stream
    except (anthropic.AuthenticationError, TypeError):
        raise ChatError(
            "The Claude API key is missing or invalid. "
            "Add ANTHROPIC_API_KEY to the .env file and restart the server."
        )
    except anthropic.RateLimitError:
        raise ChatError("The service is temporarily at capacity. Ask again in a moment.")


PROVIDERS = {
    "ollama": stream_ollama,
    "gemini": stream_gemini,
    "claude": stream_claude,
}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/about")
def about():
    return send_from_directory("static", "about.html")


@app.route("/scriptures")
def scriptures():
    return send_from_directory("static", "scriptures.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({"role": "user", "content": message})

    # Keep conversation history manageable (last 20 exchanges)
    history = conversations[session_id][-40:]

    def generate():
        # Chunks are JSON-encoded so newlines survive SSE framing.
        # "[ERROR]" tells the frontend not to gather this answer as a star.
        full_response = ""
        try:
            for text in PROVIDERS[PROVIDER](history):
                full_response += text
                yield f"data: {json.dumps(text)}\n\n"
            conversations[session_id].append(
                {"role": "assistant", "content": full_response}
            )
        except ChatError as e:
            _discard_last_user_message(session_id)
            yield "data: [ERROR]\n\n"
            yield f"data: {json.dumps(str(e))}\n\n"
        except Exception:
            _discard_last_user_message(session_id)
            yield "data: [ERROR]\n\n"
            yield f"data: {json.dumps('Something went wrong. Ask again in a moment.')}\n\n"

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


def _discard_last_user_message(session_id: str) -> None:
    history = conversations.get(session_id)
    if history and history[-1]["role"] == "user":
        history.pop()


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    conversations.pop(session_id, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
