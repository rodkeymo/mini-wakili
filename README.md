# Mini-Wakili — Legal Research Agent (Take-Home)

A minimal but working **legal-research agent** built for the NCBA Group / Project Wakili technical assessment.

It answers legal questions over a grounded corpus of Kenyan statutes, retrieves relevant passages, generates cited answers, refuses when unsupported, and includes one agentic behaviour (re-query on low confidence).

**Stack:** Python · LangChain-style control loop · Chroma · Flask · React (optional UI)

**No n8n** — the agent runs as a self-contained Flask service.

---

## Quick Start

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: set your LLM key (OpenAI-compatible or local Ollama)
# The take-home runs without any key using a deterministic grounded answerer.
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1

# Ingest the statute corpus (creates Chroma vector store)
python -m rag.ingest

# Run the API
python main.py
```

API available at `http://127.0.0.1:5000`.

### 2. Test the agent (CLI)

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"userInput": "What is required to obtain a banking licence under the Banking Act?"}'
```

### 3. Frontend (optional)

```bash
cd frontend
npm install
npm start
```

---

## Statute corpus (anti-hallucination grounding)

The vector store is built only from these approved sources:

| Act | Cap / Year |
|-----|------------|
| Banking Act | Cap. 488 |
| Capital Markets Act | Cap. 485A |
| Foreign Investments Protection Act | Cap. 518 |
| Bills of Exchange Act | Cap. 27 |
| Access to Information Act | Cap. 7M |
| Contempt of Court Act | Cap. 8F |
| Conflict of Interest Act | No. 11 of 2025 |
| Auctioneers Act | Cap. 526 |
| Food, Drugs and Chemical Substances Act | Cap. 254 |
| Alcoholic Drinks Control Act | Cap. 121 |
| General Loan and Stock Act | 1950 |

Every answer is forced to cite retrieved passages. Unsupported claims are stripped or the agent refuses.

---

## What this demonstrates

| Requirement              | Implementation                                      |
|--------------------------|-----------------------------------------------------|
| Retrieve + cite          | Hybrid retrieval → forced citations → verification  |
| Refuse / low confidence  | Explicit confidence gate before generation          |
| Agentic behaviour        | Re-query with refined question when score is low    |
| Safety                   | Input / output moderators + citation faithfulness   |
| Code quality             | Structured packages, type hints, tests              |
| Communication            | This README + DESIGN.md                             |

---

## Project layout

```
mini-wakili/
├── README.md
├── DESIGN.md
├── backend/
│   ├── agent/          # Control loop (moderate → retrieve → decide → generate → verify)
│   ├── rag/            # Ingest, retriever
│   ├── safety/         # Moderators + confidence gate
│   ├── data/
│   │   └── legal_corpus/   # Kenyan statute excerpts (grounding only)
│   ├── tests/
│   ├── main.py         # Flask API (/predict)
│   └── requirements.txt
├── frontend/           # Minimal React chat UI
└── scripts/
```

---

## API Response shape

```json
{
  "answer": "…",
  "citations": [
    {"source_id": "banking_act_cap488_c0", "passage": "…", "score": 0.82}
  ],
  "confidence": 0.78,
  "status": "ok" | "low_confidence" | "refused" | "error",
  "message": "optional explanation"
}
```

---

## Notes

- Corpus contains key excerpts for demonstration. Replace with full approved, versioned texts for production.
- No secrets are committed. Use environment variables.
- See DESIGN.md for architecture, trade-offs, and how to harden for bank use.
