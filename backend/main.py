"""Flask API for Mini-Wakili."""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

load_dotenv()

from agent.graph import run_agent, run_agent_events

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    question = (data.get("userInput") or data.get("question") or "").strip()
    if not question:
        return jsonify(
            {
                "answer": None,
                "citations": [],
                "confidence": 0.0,
                "status": "error",
                "message": "userInput is required",
                "suggested_topics": [],
            }
        ), 400

    result = run_agent(question)
    code = 200 if result.get("status") != "error" else 500
    return jsonify(result), code


# Baseline order of step ids as the agent typically emits them. Used by the
# frontend to optimistically mark the NEXT expected step as "running" until
# the backend emits its "done" frame.
EXPECTED_STEP_ORDER = [
    "moderate",
    "smalltalk",
    "retrieve_0",
    "decide_0",
    "retrieve_1",
    "decide_1",
    "generate",
    "verify",
]


@app.get("/predict/stream")
def predict_stream():
    question = (
        request.args.get("q")
        or request.args.get("userInput")
        or request.args.get("question")
        or ""
    ).strip()
    if not question:
        return jsonify({"error": "q is required"}), 400

    def generate():
        # Emit a manifest so the UI can optimistically show the next step as
        # "running" even while waiting for the backend's next frame.
        yield (
            f"data: {json.dumps({'type': 'manifest', 'steps': EXPECTED_STEP_ORDER}, ensure_ascii=False)}\n\n"
        )
        for ev in run_agent_events(question):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            # Tiny yield: forces CPython to flush the generator + lets werkzeug
            # push each frame separately so the UI sees a real progression.
            time.sleep(0)
        yield 'data: {"type": "end"}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _warm_rag():
    try:
        from rag.retriever import hybrid_search
        hybrid_search("warmup", top_k=1)
        print("RAG warmed up")
    except Exception as e:
        print(f"RAG warmup skipped: {e}")


if __name__ == "__main__":
    _warm_rag()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)