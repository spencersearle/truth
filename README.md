# truth
An artificial intelligence agent trained on teachings of the gospel of Jesus Christ to reflect true doctrines and principles found in the Church of Jesus Christ of Latter Day Saints. Responds to difficult questions on gospel topics and offers advice in how to clear up truth and error by using prophetic guidance and spiritual council. 

Render website can be found at: https://truth-0owl.onrender.com/

## How it works

The interface is a 3D night sky (Three.js). Every question kindles the central Light of Truth; every answer becomes a star in an orbiting constellation you can click to revisit.

## Model backends

The Flask backend can answer with any of three providers, chosen by the `LLM_PROVIDER` env var. If unset, it picks automatically: **gemini** on Render, **ollama** everywhere else.

| Provider | Cost | Where |
|---|---|---|
| `ollama` | Free, runs locally | Local development |
| `gemini` | Google free tier | Published site (Render) |
| `claude` | Anthropic API, per-token | Optional, needs `ANTHROPIC_API_KEY` |

## Running locally (free, with Ollama)

1. Install [Ollama](https://ollama.com) and pull a model:
   ```sh
   ollama pull llama3.2
   ```
2. Install dependencies and run:
   ```sh
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
3. Open http://localhost:8080

Copy `.env.example` to `.env` to configure providers, models, and API keys.
