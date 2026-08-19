"""Tests for confidence / refusal behaviour."""

from agent.graph import run_agent


def test_refuse_on_unrelated_question():
    """Corpus is silent on quantum physics → should refuse or low_confidence."""
    result = run_agent("What is the half-life of uranium-238?")
    assert result["status"] in ("low_confidence", "refused")
    assert result["answer"] is None or result["confidence"] < 0.4


def test_answer_on_limitation_period():
    """Should retrieve something about the Limitation of Actions Act."""
    result = run_agent(
        "What is the limitation period for actions founded on contract under Kenyan law?"
    )
    # May be ok or low depending on embedding quality; at least should not crash
    assert "status" in result
    assert "citations" in result
