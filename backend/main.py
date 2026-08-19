"""
Mini-Wakili Flask API.
Keeps the original ByteGenie /predict shape so the React frontend
(and n8n) can call it with minimal change.
"""

from __future__ import annotations

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from agent.graph import run_agent

load_dotenv()

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "mini-wakili"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected body: {"userInput": "<legal question>"}
    Returns structured answer with citations and confidence.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("userInput") or data.get("query") or data.get("question")

    if not question or not str(question).strip():
        return jsonify({"error": "No query provided", "status": "error"}), 400

    try:
        result = run_agent(str(question).strip())
        return jsonify(result)
    except Exception as exc:
        return jsonify({
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "status": "error",
            "message": str(exc),
        }), 500


@app.route("/fine_tune", methods=["POST"])
def fine_tune_stub():
    """Original ByteGenie endpoint kept as a no-op for compatibility."""
    return jsonify({
        "message": "Fine-tuning is not used in Mini-Wakili. "
                   "Use RAG + system prompt instead."
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
