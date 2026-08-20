import React, { useState, useRef, useEffect } from "react";
import ncbaLogo from "./logo/ncba-logo.png";

function ArrowUpIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ display: "block", flexShrink: 0 }}
      aria-hidden="true"
    >
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  );
}

function StopIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={color}
      style={{ display: "block", flexShrink: 0 }}
      aria-hidden="true"
    >
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  );
}

/** Stages shown while the request is in flight are streamed live from the
 *  backend via GET /predict/stream (Server-Sent Events).
 *
 *  Between SSE frames — and to work around any buffering — the UI also
 *  optimistically marks the *next* expected step as "running" right after
 *  the previous step completes. The backend's real "done" event then replaces
 *  the placeholder with the authoritative label + confidence detail.
 *
 *  STEP_SCRIPT is the ordered list of what the backend *could* emit. It's
 *  used only for the optimistic "running" preview row; rows that never get a
 *  real "done" event (e.g. retrieve_1 / decide_1 on a one-pass answer) are
 *  never shown.
 */
const STEP_SCRIPT = [
  { id: "moderate",   label: "Checking input safety" },
  { id: "smalltalk",  label: "Greeting — skipped retrieval" },
  { id: "retrieve_0", label: "Searching the legal corpus" },
  { id: "decide_0",   label: "Evaluating retrieval confidence" },
  { id: "retrieve_1", label: "Searching again with refined query" },
  { id: "decide_1",   label: "Evaluating refined retrieval confidence" },
  { id: "generate",   label: "Drafting grounded answer" },
  { id: "verify",     label: "Verifying citations & output safety" },
];

const STEP_LABEL_BY_ID = STEP_SCRIPT.reduce((acc, s) => {
  acc[s.id] = s.label;
  return acc;
}, {});

function AgentActivity({ steps, liveLabel, loading }) {
  const list = steps?.length
    ? steps
    : liveLabel
      ? [{ id: "live", label: liveLabel, status: "running" }]
      : [];

  if (!list.length && !loading) return null;

  return (
    <div className="agent-activity">
      <div className="agent-activity-title">Agent activity</div>
      <ul>
        {list.map((s, i) => (
          <li key={s.id || i} className={`step step-${s.status || "done"}`}>
            <span className="step-icon">
              {s.status === "running" ? (
                <span className="spinner" />
              ) : s.status === "error" ? (
                "!"
              ) : (
                "✓"
              )}
            </span>
            <span className="step-body">
              <span className="step-label">{s.label}</span>
              {s.detail && <span className="step-detail">{s.detail}</span>}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [focused, setFocused] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [askedQuestion, setAskedQuestion] = useState("");
  const [typedAnswer, setTypedAnswer] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [liveSteps, setLiveSteps] = useState([]);
  // Mirror of liveSteps — refs avoid stale closure issues inside the
  // streaming callbacks + optimistic interval.
  const liveStepsRef = useRef([]);
  // Whether the agent has signaled a real retry (decide_0 returned retry).
  // Only in that case do we allow the optimistic preview to move past
  // decide_0 → retrieve_1 / decide_1 (so "search again" never appears on a
  // 1-pass answer).
  const retryUnlockedRef = useRef(false);
  const liveStepScriptRef = useRef(STEP_SCRIPT);

  const textareaRef = useRef(null);

  useEffect(() => {
    liveStepsRef.current = liveSteps;
  }, [liveSteps]);

  /**
   * While a request is in flight, project a single optimistic "running"
   * step placeholder for whatever the backend should do next. It is keyed
   * by `id` so the real `step:done` event from the backend replaces it
   * cleanly. On a 1-pass answer, retrieve_1 / decide_1 will never appear
   * because `retryUnlockedRef` keeps the projection clamped before them —
   * matching real agent behaviour.
   */
  useEffect(() => {
    if (!loading) return undefined;
    const label = "projected";
    const ADVANCE_MS = 900; // same cadence as the original LIVE_STAGES pulse
    const lastAdvance = { t: Date.now() };
    const tick = () => {
      const doneIds = new Set(
        liveStepsRef.current
          .filter((s) => s.status === "done" || s.status === "error")
          .map((s) => s.id)
      );
      const script = liveStepScriptRef.current;
      const retryUnlocked = retryUnlockedRef.current;
      // Find the next script id that hasn't been marked done yet, honouring
      // the retry gate for retrieve_1 / decide_1.
      let nextId = null;
      for (const s of script) {
        if (doneIds.has(s.id)) continue;
        if (!retryUnlocked && (s.id === "retrieve_1" || s.id === "decide_1")) break;
        nextId = s.id;
        break;
      }
      if (!nextId) return;
      // Only append a projected "running" row if the backend hasn't sent a
      // real `running` row for the same id yet.
      const anySameId = liveStepsRef.current.some((s) => s.id === nextId);
      if (!anySameId) {
        const projected = {
          id: nextId,
          label: STEP_LABEL_BY_ID[nextId] || nextId,
          status: "running",
          _projected: true,
        };
        setLiveSteps((prev) => {
          if (prev.some((s) => s.id === nextId)) return prev;
          return [...prev, projected];
        });
        liveStepsRef.current = [...liveStepsRef.current, projected];
      }
      lastAdvance.t = Date.now();
    };
    tick(); // first step immediately
    const id = setInterval(tick, ADVANCE_MS);
    return () => clearInterval(id);
  }, [loading]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  useEffect(() => {
    if (!result?.answer) {
      setTypedAnswer("");
      setIsTyping(false);
      return;
    }
    const text = result.answer;
    setTypedAnswer("");
    setIsTyping(true);
    let i = 0;
    const speed = text.length > 400 ? 8 : 16;
    const interval = setInterval(() => {
      i += 1;
      setTypedAnswer(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [result]);

  const ask = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setAskedQuestion(question);
    setHasStarted(true);
    setInput("");
    setLoading(true);
    setError(null);
    setResult(null);
    setLiveSteps([]);
    liveStepsRef.current = [];
    retryUnlockedRef.current = false;
    liveStepScriptRef.current = STEP_SCRIPT;

    let finished = false;
    let settled = false;

    const fallback = async () => {
      if (settled) return;
      settled = true;
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ userInput: question }),
        });
        const data = await res.json();
        if (!res.ok)
          throw new Error(data.message || data.error || "Request failed");
        // Retry hint: if the server's meta.steps contains a retrieve_1
        // entry, the user still saw a 2-pass flow in the fallback trail.
        setResult(data);
        setLiveSteps(data.meta?.steps || []);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    let url;
    try {
      url = `/predict/stream?q=${encodeURIComponent(question)}`;
    } catch (e) {
      await fallback();
      return;
    }

    let es;
    try {
      es = new EventSource(url);
    } catch (e) {
      await fallback();
      return;
    }

    es.onmessage = (evt) => {
      if (finished) return;
      let data;
      try {
        data = JSON.parse(evt.data);
      } catch {
        return;
      }

      if (data.type === "manifest" && Array.isArray(data.steps)) {
        // Backend can override the optimistic script via manifest event.
        liveStepScriptRef.current = data.steps.map((id) => ({
          id,
          label: STEP_LABEL_BY_ID[id] || id,
        }));
      }

      if (data.type === "step") {
        const step = data.step;
        if (!step) return;
        // If decide_0 came back "done" with a retry label, unlock the
        // 2nd-pass optimistic rows so the UI can show retrieve_1 running.
        if (
          step.id === "decide_0" &&
          step.status === "done" &&
          typeof step.label === "string" &&
          /retry|search again/i.test(step.label)
        ) {
          retryUnlockedRef.current = true;
        }
        // If smalltalk came back done, skip the RAG script entirely:
        // don't show phantom retrieve/generate rows after smalltalk.
        if (step.id === "smalltalk" && step.status === "done") {
          liveStepScriptRef.current = STEP_SCRIPT.filter(
            (s) =>
              ["moderate", "smalltalk"].includes(s.id) ||
              liveStepsRef.current.some((x) => x.id === s.id)
          );
        }
        // Upsert this step by id — replaces any prior placeholder with the
        // real backend-sent row.
        setLiveSteps((prev) => {
          const without = prev.filter((s) => s.id !== step.id);
          const next = [...without, step];
          liveStepsRef.current = next;
          return next;
        });
      }

      if (data.type === "result") {
        finished = true;
        setResult(data.result);
        setLiveSteps(data.result?.meta?.steps || []);
        setLoading(false);
        try {
          es.close();
        } catch {}
      }

      if (data.type === "end") {
        setLoading(false);
        try {
          es.close();
        } catch {}
      }
    };

    es.onerror = () => {
      try {
        es.close();
      } catch {}
      if (!finished) {
        fallback();
      }
    };
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  const realSteps = (() => {
    const raw = liveSteps.length ? liveSteps : result?.meta?.steps || [];
    if (!raw?.length) return raw;
    const order = new Map(STEP_SCRIPT.map((s, i) => [s.id, i]));
    return [...raw].sort((a, b) => {
      const ai = order.has(a.id) ? order.get(a.id) : 9999;
      const bi = order.has(b.id) ? order.get(b.id) : 9999;
      return ai - bi;
    });
  })();
  const attempts = result?.meta?.attempts ?? 0;

  const composer = (
    <div className="composer-wrap">
      <div className={`composer ${focused ? "focused" : ""}`}>
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask a legal research question over the sample corpus…"
        />
        <button
          className="send-btn"
          onClick={ask}
          disabled={!input.trim() || loading}
          aria-label={loading ? "Researching" : "Send"}
        >
          {loading ? (
            <StopIcon size={14} color="#A8A49D" />
          ) : (
            <ArrowUpIcon size={18} color="#FFFFFF" />
          )}
        </button>
      </div>
      <div className="hint">
        Press Enter to send, Shift+Enter for a new line
      </div>
    </div>
  );

  return (
    <div className="app">
      <style>{`
        * { box-sizing: border-box; }
        html, body, #root { height: 100%; margin: 0; }

        .app {
          margin: 0 auto;
          padding: 1.25rem 1.25rem 1.5rem;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #F5F4EE;
          height: 100vh;
          color: #1F1E1D;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .logo-bar {
          display: flex;
          justify-content: flex-start;
          margin-bottom: 0.5rem;
          flex-shrink: 0;
        }
        .logo-bar img {
          height: 70px;
          width: auto;
          object-fit: contain;
          display: block;
        }

        .center-stage {
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          width: 100%;
        }
        header { text-align: center; margin-bottom: 1.75rem; max-width: 560px; }
        header h1 {
          margin: 0 0 0.35rem;
          font-size: 1.85rem;
          font-weight: 600;
          letter-spacing: -0.01em;
        }
        header p { margin: 0; color: #83807C; font-size: 0.95rem; }

        .chat-stage {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-height: 0;
          width: 100%;
        }
        .results-scroll {
          flex: 1;
          overflow-y: auto;
          padding: 0.5rem 0.25rem 1rem;
        }
        .results-wrap {
          width: 100%;
          max-width: 640px;
          margin: 0 auto;
        }

        .question-echo {
          font-size: 0.95rem;
          font-weight: 600;
          background: #EFEDE6;
          border-radius: 14px;
          padding: 0.65rem 1rem;
          margin-bottom: 1rem;
          display: inline-block;
        }

        /* Agent activity panel */
        .agent-activity {
          background: #FFFFFF;
          border: 1px solid #E4E1DA;
          border-radius: 14px;
          padding: 0.9rem 1rem;
          margin-bottom: 1rem;
        }
        .agent-activity-title {
          font-size: 0.72rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #83807C;
          margin-bottom: 0.55rem;
        }
        .agent-activity ul {
          list-style: none;
          margin: 0;
          padding: 0;
        }
        .step {
          display: flex;
          gap: 0.55rem;
          align-items: flex-start;
          padding: 0.35rem 0;
          font-size: 0.88rem;
        }
        .step-icon {
          width: 1.15rem;
          height: 1.15rem;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.65rem;
          font-weight: 700;
          flex-shrink: 0;
          margin-top: 0.1rem;
        }
        .step-done .step-icon {
          background: #DCF0E4;
          color: #1E7A46;
        }
        .step-running .step-icon {
          background: #F5E6DF;
          color: #C96442;
        }
        .step-error .step-icon {
          background: #FDEEEA;
          color: #B5563A;
        }
        .step-body {
          display: flex;
          flex-direction: column;
          gap: 0.1rem;
        }
        .step-label { color: #1F1E1D; }
        .step-detail {
          font-size: 0.78rem;
          color: #83807C;
        }
        .spinner {
          width: 10px;
          height: 10px;
          border: 2px solid #C96442;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .composer-wrap {
          width: 100%;
          max-width: 640px;
          margin: 0 auto;
          padding: 0 0.5rem;
          flex-shrink: 0;
        }
        .composer {
          display: flex;
          align-items: flex-end;
          gap: 0.5rem;
          background: #FFFFFF;
          border: 1px solid #E4E1DA;
          border-radius: 1.75rem;
          padding: 0.75rem 0.75rem 0.75rem 1.25rem;
          box-shadow: 0 1px 2px rgba(31, 30, 29, 0.04);
          transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .composer.focused {
          border-color: #C96442;
          box-shadow: 0 0 0 3px rgba(201, 100, 66, 0.12);
        }
        .composer textarea {
          flex: 1;
          resize: none;
          border: none;
          outline: none;
          background: transparent;
          font-family: inherit;
          font-size: 1rem;
          line-height: 1.5;
          color: #1F1E1D;
          padding: 0.35rem 0;
          max-height: 200px;
          overflow-y: auto;
          min-height: 1.5rem;
        }
        .composer textarea::placeholder { color: #A8A49D; }
        .send-btn {
          flex-shrink: 0;
          width: 2.25rem;
          height: 2.25rem;
          border-radius: 50%;
          border: none;
          background: #C96442;
          color: #FFFFFF;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        }
        .send-btn:hover:not(:disabled) { background: #B5563A; }
        .send-btn:disabled {
          background: #E4E1DA;
          color: #A8A49D;
          cursor: not-allowed;
        }
        .hint {
          text-align: center;
          margin-top: 0.6rem;
          font-size: 0.78rem;
          color: #A8A49D;
        }

        .error {
          padding: 0.75rem 1rem;
          background: #FDEEEA;
          border: 1px solid #F3C7B9;
          border-radius: 10px;
          color: #B5563A;
          font-size: 0.9rem;
        }
        .result {
          padding: 1.25rem;
          border-radius: 14px;
          background: #FFFFFF;
          border: 1px solid #E4E1DA;
        }
        .meta {
          display: flex;
          flex-wrap: wrap;
          gap: 0.75rem 1rem;
          align-items: center;
          margin-bottom: 0.75rem;
          font-size: 0.85rem;
          color: #83807C;
        }
        .badge {
          background: #EFEDE6;
          padding: 0.2rem 0.6rem;
          border-radius: 999px;
          text-transform: uppercase;
          font-size: 0.7rem;
          font-weight: 600;
        }
        .status-ok .badge { background: #DCF0E4; color: #1E7A46; }
        .status-low_confidence .badge,
        .status-refused .badge { background: #FDEEEA; color: #B5563A; }
        .retry-badge {
          background: #F5E6DF;
          color: #C96442;
          padding: 0.2rem 0.55rem;
          border-radius: 999px;
          font-size: 0.7rem;
          font-weight: 600;
        }
        .answer h3, .citations h3 {
          margin: 0 0 0.4rem;
          font-size: 0.95rem;
          font-weight: 600;
        }
        .answer pre {
          white-space: pre-wrap;
          font-family: inherit;
          margin: 0 0 1rem;
          line-height: 1.55;
          font-size: 0.95rem;
        }
        .cursor {
          display: inline-block;
          width: 2px;
          height: 1em;
          background: #1F1E1D;
          margin-left: 1px;
          vertical-align: text-bottom;
          animation: blink 0.9s step-end infinite;
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .citations ul { padding-left: 1.1rem; margin: 0; }
        .citations li { margin-bottom: 0.75rem; font-size: 0.9rem; }
        .citations p { margin: 0.25rem 0 0; color: #83807C; }
        .message { color: #C08A2E; font-size: 0.9rem; }

        @media (max-width: 480px) {
          .logo-bar img { height: 40px; }
          header h1 { font-size: 1.5rem; }
        }
      `}</style>

      <div className="logo-bar">
        <img src={ncbaLogo} alt="NCBA" />
      </div>

      {!hasStarted ? (
        <div className="center-stage">
          <header>
            <h1>Mini-Wakili</h1>
            <p>Legal research agent — drafts only; lawyer always decides.</p>
          </header>
          {composer}
        </div>
      ) : (
        <div className="chat-stage">
          <div className="results-scroll">
            <div className="results-wrap">
              <div className="question-echo">{askedQuestion}</div>

              {/* Activity panel — steps streamed live from backend during the
                  request, then final `meta.steps` rendered as a static trail. */}
              {(loading || realSteps.length > 0) && (
                <AgentActivity
                  loading={loading}
                  steps={realSteps}
                  liveLabel={null}
                />
              )}

              {error && <div className="error">{error}</div>}

              {result && (
                <div className={`result status-${result.status}`}>
                  <div className="meta">
                    <span className="badge">{result.status}</span>
                    <span>
                      Confidence: {(result.confidence * 100).toFixed(0)}%
                    </span>
                    {(result.meta?.attempts ?? 0) > 0 && (
                      <span className="retry-badge">
                        Searched again ({result.meta.attempts} re-query)
                      </span>
                    )}
                  </div>

                  {result.message && (
                    <p className="message">{result.message}</p>
                  )}

                  {result.answer && (
                    <div className="answer">
                      <h3>Answer</h3>
                      <pre>
                        {typedAnswer}
                        {isTyping && <span className="cursor" />}
                      </pre>
                    </div>
                  )}

                  {!isTyping &&
                    result.status === "ok" &&
                    result.citations?.length > 0 && (
                      <div className="citations">
                        <h3>Citations</h3>
                        <ul>
                          {result.citations.map((c, i) => (
                            <li key={i}>
                              <strong>[{c.source_id}]</strong> (score{" "}
                              {c.score.toFixed(2)})<p>{c.passage}</p>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {result.status !== "ok" &&
                    result.suggested_topics?.length > 0 && (
                      <div className="topics">
                        <h3>
                          Topics you can ask about from the available sources
                        </h3>
                        <ol>
                          {result.suggested_topics.slice(0, 8).map((t, i) => (
                            <li key={i}>
                              <button
                                type="button"
                                className="topic-chip"
                                onClick={() => setInput(`Tell me about: ${t}`)}
                              >
                                {t}
                              </button>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}
                </div>
              )}
            </div>
          </div>
          {composer}
        </div>
      )}
    </div>
  );
}
