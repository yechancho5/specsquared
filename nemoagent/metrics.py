from __future__ import annotations

import re
from collections.abc import Iterable


QUALITY_SIGNALS_GENERAL: list[tuple[str, tuple[str, ...]]] = [
    ("problem", ("problem", "pain point", "challenge")),
    ("target user", ("target user", "customer", "doctor", "physician", "user")),
    ("solution", ("solution", "platform", "tool", "product")),
    ("specific features", ("feature", "dashboard", "workflow", "briefing", "search")),
    ("differentiation/moat", ("differentiation", "different", "advantage", "moat", "unlike")),
    ("risks or limitations", ("risk", "limitation", "mitigation")),
    ("demo or implementation plan", ("demo", "rollout", "implementation plan", "launch")),
    ("structure", ("\n", ":")),
]

QUALITY_SIGNALS_CODING: list[tuple[str, tuple[str, ...]]] = [
    ("implementation scope", ("implementation", "plan", "workflow", "orchestration")),
    ("testability", ("test", "validation", "regression")),
    ("maintainability", ("compatibility", "refactor", "modular", "readability")),
    ("risk handling", ("risk", "failure", "guardrail", "mitigation")),
    ("concrete steps", ("1.", "2.", "3.", "step", "phase")),
    ("quality controls", ("approval", "criteria", "gate", "verify")),
    ("performance/cost awareness", ("latency", "token", "benchmark", "cost")),
]

QUALITY_SIGNALS_DOC_REVIEW: list[tuple[str, tuple[str, ...]]] = [
    ("clarity improvements", ("clarity", "clearer", "ambiguous", "readability")),
    ("structure", ("structure", "section", "flow", "transition")),
    ("evidence fidelity", ("evidence", "verify", "precise", "accuracy")),
    ("risk/caveat coverage", ("risk", "caveat", "assumption", "limitation")),
    ("actionability", ("action", "next-step", "recommendation", "edits")),
    ("tone consistency", ("tone", "audience", "jargon", "terminology")),
]


def parse_critic_suggestions(critic_text: str) -> list[str]:
    suggestions: list[str] = []
    for line in critic_text.splitlines():
        clean = line.strip()
        if re.match(r"^\d+[\).\s-]+", clean):
            suggestions.append(re.sub(r"^\d+[\).\s-]+", "", clean).strip())
    return suggestions


def _keywords_for_suggestion(suggestion: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", suggestion.lower())
    stopwords = {"that", "with", "from", "this", "into", "they", "have", "their", "would", "should", "explicitly"}
    return {token for token in tokens if token not in stopwords}


def count_suggestions_incorporated(suggestions: Iterable[str], final_output: str) -> int:
    final_lower = final_output.lower()
    incorporated = 0
    for suggestion in suggestions:
        keywords = _keywords_for_suggestion(suggestion)
        if not keywords:
            continue
        matches = sum(1 for keyword in keywords if keyword in final_lower)
        if matches >= max(1, min(2, len(keywords) // 2)):
            incorporated += 1
    return incorporated


def communication_effect_score(suggestions_count: int, incorporated_count: int) -> float:
    if suggestions_count <= 0:
        return 0.0
    return round(incorporated_count / suggestions_count, 2)


def quality_score(output: str, scenario: str = "general") -> float:
    text = output.lower()
    if scenario == "coding":
        signals = QUALITY_SIGNALS_CODING
    elif scenario == "document-review":
        signals = QUALITY_SIGNALS_DOC_REVIEW
    else:
        signals = QUALITY_SIGNALS_GENERAL

    earned = 0
    for _, keywords in signals:
        if any(keyword in text for keyword in keywords):
            earned += 1
    score = (earned / len(signals)) * 10
    return round(score, 1)
