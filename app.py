import os
from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__, static_folder="static")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

# Store conversation histories in memory (per-session)
conversations: dict[str, list] = {}


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

    conversations[session_id].append({"role": "user", "parts": [{"text": message}]})

    # Keep conversation history manageable (last 20 exchanges)
    history = conversations[session_id][-40:]

    def generate():
        full_response = ""
        try:
            response = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=history,
                config={"system_instruction": SYSTEM_PROMPT},
            )

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {chunk.text}\n\n"

            conversations[session_id].append(
                {"role": "model", "parts": [{"text": full_response}]}
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                yield "data: I'm sorry, the service is temporarily at capacity. Please try again in a moment.\n\n"
            else:
                yield f"data: An error occurred. Please try again.\n\n"

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    conversations.pop(session_id, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
