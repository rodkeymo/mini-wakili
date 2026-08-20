"""Suggest discussable topics from the local statute corpus."""

from __future__ import annotations

from pathlib import Path
from typing import List

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "legal_corpus"

# Friendly labels for known files (extend as you add acts)
LABELS = {
    "banking_act_cap488": "Banking Act — licensing, supervision, restrictions on business",
    "capital_markets_act_cap485a": "Capital Markets Act — markets, intermediaries, offences",
    "foreign_investments_protection_act_cap518": "Foreign Investments Protection Act — certificates, protection of investments",
    "bills_of_exchange_act_cap27": "Bills of Exchange Act — cheques, promissory notes, negotiation",
    "access_to_information_act_cap7m": "Access to Information Act — rights of access, limits, appeals",
    "alcoholic_drinks_control_act_cap121": "Alcoholic Drinks Control Act — licensing, sale, offences",
    "auctioneers_act_cap526": "Auctioneers Act — licensing and conduct of auctioneers",
    "conflict_of_interest_act": "Conflict of Interest Act — public officers, disclosure duties",
    "contempt_of_court_act": "Contempt of Court Act — contempt, procedure, sanctions",
    "food_drugs_and_chemical_substances_act": "Food, Drugs and Chemical Substances Act — standards, offences",
    "food_and_feed_safety_control_coordination_act": "Food and Feed Safety Control Co-ordination Act — coordination, enforcement",
    "general_loan_and_stock_act": "General Loan and Stock Act — public loans, stock, debentures",
}


def list_corpus_topics(limit: int = 8) -> List[str]:
    """
    Return up to `limit` human-readable topics from available .txt sources.
    Prefers LABELS; falls back to cleaned file stems.
    """
    if not CORPUS_DIR.exists():
        return []

    topics: List[str] = []
    seen = set()

    for path in sorted(CORPUS_DIR.glob("*.txt")):
        stem = path.stem.lower()
        label = LABELS.get(stem)
        if not label:
            # banking_act_cap488 → Banking Act Cap488
            label = stem.replace("_", " ").strip().title()
        if label not in seen:
            seen.add(label)
            topics.append(label)
        if len(topics) >= limit:
            break

    return topics