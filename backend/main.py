"""Flask API for Mini-Wakili."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

from agent.graph import run_agent

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