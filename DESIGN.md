# DESIGN.md — Mini-Wakili

## 1. Architecture overview

```
Client (CLI / React)
        │
        ▼
Flask /predict endpoint
        │
        ▼
┌──────────────────────────────────────────┐
│  Agent control loop                      │
│  1. Input moderator                      │
│  2. Retrieve (dense over statute corpus) │
│  3. Agentic decision (re-query?)         │
│  4. Generate with forced citations       │
│  5. Citation verifier + confidence gate  │
│  6. Output moderator                     │
└──────────────────────────────────────────┘
        │
        ▼
Chroma vector store  ←  legal_corpus (Kenyan statutes only)
```

**No n8n.** The agent is a self-contained Python service. Human-in-the-loop can be added later at the application layer (UI approval button + audit log).

## 2. How hallucination is prevented

1. **Closed corpus** — only the approved Kenyan statutes listed in README are ingested. The model never sees external web content or its own parametric knowledge as a source of law.
2. **Forced citation format** — every factual claim must be followed by a `[source_id]`. Post-processing verifies that every cited ID exists in the retrieved set.
3. **Confidence gate** — if top retrieval score or average of top-3 is below threshold, the agent returns `status: low_confidence` and no answer.
4. **Output moderator** — blocks over-confident absolute legal conclusions that claim to be final opinions.
5. **Refusal message** — when the corpus is silent, the agent explicitly says it cannot answer from the available approved sources.

## 3. Key design choices & trade-offs

| Choice | Rationale | Trade-off |
|--------|-----------|-----------|
| Statute excerpts only | Focused, auditable grounding for the take-home | Not the full text of every Act |
| Chroma (local) | Zero-infra | Swap for Azure AI Search / Pinecone in production |
| Deterministic grounded answerer (no API key required) | Runs offline for evaluation | Replace with LLM call when keys available |
| Simple control loop (not full LangGraph runtime) | Easy to read and defend line-by-line | Same nodes can be dropped into LangGraph later |
| No n8n | Simpler deliverable | HITL must be implemented in UI/backend later |

## 4. What was deliberately left out (time budget)

- Full OCR / layout-aware PDF pipeline (excerpts used instead)
- Cross-encoder re-ranker
- Strong NLI entailment check on every citation
- Multi-tenancy / matter isolation
- Persistent conversation memory
- Full text of every page of every statute

## 5. How this would be hardened for bank use

1. **Data residency** — LLM inside bank Azure tenancy (private endpoint) or on-prem open model.
2. **Corpus** — full approved, versioned statutes + internal policies; every chunk carries `status`, `effective_date`, `source_hash`.
3. **Audit trail** — request ID, user, matter_id, retrieved set, model I/O, confidence, human approver; retain ≥ 5 years.
4. **Human-in-the-loop** — product never exposes a “skip review” path for anything relied upon.
5. **Citation faithfulness** — secondary entailment model rejects unsupported claims.
6. **Isolation** — retrieval filtered by `matter_id`; no cross-matter memory.
7. **Observability** — structured logs, evaluation harness (Recall@k, citation precision).

## 6. Measuring retrieval quality

```bash
cd backend && python ../scripts/evaluate_retrieval.py
```

Golden questions over the statute corpus; check that relevant sections are retrieved.
