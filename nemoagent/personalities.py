from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import PersonalityName, ScenarioType


APPROVAL_PREFIX = "APPROVED:"
APPROVAL_AGENT: PersonalityName = "conscientiousness"


@dataclass(frozen=True)
class PersonalityProfile:
    name: PersonalityName
    title: str
    focus: str
    tone: str


PERSONALITY_ORDER: list[PersonalityName] = [
    "extraversion",
    "agreeableness",
    "neuroticism",
    "sensitivity",
    "self-esteem",
    "openness",
    "conscientiousness",
]

PERSONALITIES: dict[PersonalityName, PersonalityProfile] = {
    "conscientiousness": PersonalityProfile(
        name="conscientiousness",
        title="Conscientiousness Agent",
        focus="Enforce quality gates, verify completeness, and check if feedback was addressed.",
        tone="systematic and exact",
    ),
    "agreeableness": PersonalityProfile(
        name="agreeableness",
        title="Agreeableness Agent",
        focus="Protect reader alignment, usability, and collaborative clarity.",
        tone="constructive and user-centered",
    ),
    "extraversion": PersonalityProfile(
        name="extraversion",
        title="Extraversion Agent",
        focus="Improve communication impact, narrative flow, and persuasion.",
        tone="energetic and direct",
    ),
    "neuroticism": PersonalityProfile(
        name="neuroticism",
        title="Neuroticism Agent",
        focus="Find risks, edge cases, brittle assumptions, and failure modes.",
        tone="cautious and risk-focused",
    ),
    "openness": PersonalityProfile(
        name="openness",
        title="Openness Agent",
        focus="Synthesize novel but practical improvements from all feedback.",
        tone="creative and integrative",
    ),
    "self-esteem": PersonalityProfile(
        name="self-esteem",
        title="Self-Esteem Agent",
        focus="Defend strong design choices and remove weak, hedged wording.",
        tone="confident and decisive",
    ),
    "sensitivity": PersonalityProfile(
        name="sensitivity",
        title="Sensitivity Agent",
        focus="Catch subtle ambiguity, terminology drift, and context mismatch.",
        tone="nuanced and precise",
    ),
}


CODING_HINTS = (
    "Treat the task as software engineering work. Prioritize correctness, testability, maintainability, "
    "clear trade-offs, and concrete implementation details."
)

DOC_REVIEW_HINTS = (
    "Treat the task as document review/editing work. Prioritize factual precision, structure, tone, readability, "
    "and actionable revision guidance."
)

GENERAL_HINTS = (
    "Treat the task as general reasoning work. Prioritize clarity, specificity, and practical usefulness."
)


def detect_scenario(prompt: str, scenario: ScenarioType) -> ScenarioType:
    if scenario != "auto":
        return scenario

    lower = prompt.lower()

    coding_patterns = [
        r"\bcode\b",
        r"\bimplement\b",
        r"\bbug\b",
        r"\brefactor\b",
        r"\bfunction\b",
        r"\bclass\b",
        r"\bapi\b",
        r"\btest(s|ing)?\b",
        r"\bmodule\b",
        r"\brepo(sitory)?\b",
    ]
    doc_patterns = [
        r"\bdocument\b",
        r"\breview\b",
        r"\bproofread\b",
        r"\bedit\b",
        r"\bgrammar\b",
        r"\bstyle\b",
        r"\bclarity\b",
        r"\bsummary\b",
        r"\bproposal\b",
        r"\breport\b",
    ]

    coding_hits = sum(bool(re.search(pattern, lower)) for pattern in coding_patterns)
    doc_hits = sum(bool(re.search(pattern, lower)) for pattern in doc_patterns)

    if coding_hits > doc_hits and coding_hits >= 2:
        return "coding"
    if doc_hits > coding_hits and doc_hits >= 2:
        return "document-review"
    return "general"


def scenario_guidance(scenario: ScenarioType) -> str:
    if scenario == "coding":
        return CODING_HINTS
    if scenario == "document-review":
        return DOC_REVIEW_HINTS
    return GENERAL_HINTS


def reviewer_system_prompt(profile: PersonalityProfile, scenario: ScenarioType) -> str:
    return (
        f"You are the {profile.title} in a seven-personality collaboration. "
        f"Your focus: {profile.focus} "
        f"Your tone: {profile.tone}. "
        f"{scenario_guidance(scenario)} "
        "Return 2-4 concise, concrete points the synthesizer should incorporate next. "
        "Prefer numbered lines for easier tracking."
    )


def synthesizer_system_prompt(profile: PersonalityProfile, scenario: ScenarioType) -> str:
    return (
        f"You are the {profile.title} acting as synthesis lead. "
        f"Your focus: {profile.focus} "
        f"Your tone: {profile.tone}. "
        f"{scenario_guidance(scenario)} "
        "Use the full transcript and produce a stronger revised output that directly resolves current reviewer concerns. "
        "Return the revised artifact only."
    )


def approver_system_prompt(profile: PersonalityProfile, scenario: ScenarioType) -> str:
    return (
        f"You are the {profile.title} and the final quality gate. "
        f"Your focus: {profile.focus} "
        f"Your tone: {profile.tone}. "
        f"{scenario_guidance(scenario)} "
        "If the revised artifact is ready, start with 'APPROVED:' and briefly justify approval. "
        "Otherwise return numbered required fixes. Be strict and concrete."
    )


def is_approved(text: str) -> bool:
    return text.strip().lower().startswith(APPROVAL_PREFIX.lower())
