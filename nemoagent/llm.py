from __future__ import annotations

import os
import time
from collections.abc import Sequence

import httpx

from .schemas import BackendName, GenerationResult


Message = dict[str, str]


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.33))


def _extract_prompt(messages: Sequence[Message]) -> str:
    return "\n".join(message.get("content", "") for message in messages if message.get("role") != "system")


class BaseLLMBackend:
    backend_name: BackendName

    def generate(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        raise NotImplementedError


class MockLLMBackend(BaseLLMBackend):
    backend_name: BackendName = "mock"

    def generate(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        start = time.perf_counter()
        prompt = _extract_prompt(messages)
        lower = prompt.lower()

        explicit_scenario_doc = "scenario context: document-review" in lower
        explicit_scenario_coding = "scenario context: coding" in lower
        explicit_scenario_general = "scenario context: general" in lower

        is_coding = explicit_scenario_coding or (
            not explicit_scenario_doc
            and not explicit_scenario_general
            and any(token in lower for token in ("code", "implement", "function", "class", "test", "api"))
        )
        is_doc = explicit_scenario_doc or (
            not explicit_scenario_coding
            and not explicit_scenario_general
            and any(token in lower for token in ("document", "review", "edit", "proofread", "style", "clarity"))
        )

        if "this is conscientiousness approval round 2" in lower:
            text = (
                "APPROVED: The revised artifact is now specific, internally consistent, and addresses prior high-priority "
                "quality concerns with clear execution details."
            )
        elif "this is conscientiousness approval round 1" in lower:
            text = (
                "1. Tighten concrete execution details with explicit steps and ownership.\n"
                "2. Resolve vague language and convert claims into verifiable specifics.\n"
                "3. Strengthen risk handling and success criteria for evaluation."
            )
        elif "this is openness synthesis round 2" in lower:
            if is_coding:
                text = (
                    "Implementation Plan:\n"
                    "1. Introduce a personality registry and role-aware prompt policy.\n"
                    "2. Add a seven-personality orchestration branch while preserving legacy workflows.\n"
                    "3. Generalize metrics to track per-agent latency/tokens and approval rounds.\n"
                    "4. Expand mock-mode fixtures for deterministic personality dialogue tests.\n"
                    "5. Validate with regression commands and benchmark parity checks.\n\n"
                    "Code Review Notes:\n"
                    "- Keep API compatibility for existing CLI flags and run artifacts.\n"
                    "- Add focused tests around approval gating and scenario detection.\n"
                    "- Document migration and fallback behavior in README."
                )
            elif is_doc:
                text = (
                    "Revised Review Summary:\n"
                    "The document now has a clearer thesis, cleaner section transitions, and explicit action items. "
                    "Ambiguous terms were normalized, repetitive phrasing was removed, and evidence claims are easier to verify.\n\n"
                    "Remaining strengths:\n"
                    "- Reader intent is addressed early.\n"
                    "- Risk caveats are explicit and concise.\n"
                    "- The conclusion maps directly to next-step decisions."
                )
            else:
                text = (
                    "Revised output:\n"
                    "- Clarified the problem and user impact with concrete context.\n"
                    "- Converted broad claims into specific, testable statements.\n"
                    "- Added explicit risks and practical mitigations.\n"
                    "- Organized the content into a structured, decision-ready format."
                )
        elif "this is openness synthesis round 1" in lower:
            if is_coding:
                text = (
                    "Draft Implementation Direction:\n"
                    "- Build a single adaptive seven-personality workflow for coding and document review prompts.\n"
                    "- Use extraversion/agreeableness/neuroticism/sensitivity/self-esteem as reviewer personas.\n"
                    "- Use openness as synthesizer and conscientiousness as final approval gate.\n"
                    "- Preserve legacy single and two-agent-compatible modes for backward compatibility."
                )
            elif is_doc:
                text = (
                    "Draft Document Revision:\n"
                    "The draft improves readability and flow, but it still needs sharper evidence language, stronger section purpose, "
                    "and clearer revision outcomes tied to audience needs."
                )
            else:
                text = (
                    "Draft revised output:\n"
                    "The response is more structured and specific than the baseline, with clearer priorities and practical next actions."
                )
        elif "this is openness synthesis round 0" in lower:
            if is_coding:
                text = (
                    "Initial solution sketch:\n"
                    "Implement a multi-agent workflow that coordinates seven personality-driven reviewers and a synthesis step to "
                    "produce stronger coding and review outputs while retaining existing CLI behavior."
                )
            elif is_doc:
                text = (
                    "Initial review baseline:\n"
                    "The current draft communicates intent but lacks consistent structure, precision, and reader-directed actionability."
                )
            else:
                text = (
                    "Initial baseline:\n"
                    "A first draft exists but needs stronger structure, concrete details, and clearer success criteria."
                )
        elif "this is extraversion review round" in lower:
            text = (
                "1. Increase communication impact by making the lead sentence outcome-focused.\n"
                "2. Tighten section transitions so the narrative is easier to follow.\n"
                "3. Replace abstract claims with vivid, concrete examples."
            )
        elif "this is agreeableness review round" in lower:
            text = (
                "1. Improve audience alignment and reduce jargon where possible.\n"
                "2. Add one line that addresses likely stakeholder concerns.\n"
                "3. Keep recommendations collaborative and implementation-friendly."
            )
        elif "this is neuroticism review round" in lower:
            text = (
                "1. Add explicit risk cases and failure contingencies.\n"
                "2. Clarify assumptions that could break under real constraints.\n"
                "3. Define guardrails for high-impact mistakes."
            )
        elif "this is sensitivity review round" in lower:
            text = (
                "1. Resolve subtle ambiguity in terminology and scope.\n"
                "2. Ensure each claim maps cleanly to supporting context.\n"
                "3. Normalize wording to avoid interpretation drift."
            )
        elif "this is self-esteem review round" in lower:
            text = (
                "1. Remove hedging and commit to clear design choices.\n"
                "2. Defend trade-offs explicitly instead of implying them.\n"
                "3. Keep the conclusion decisive and action-oriented."
            )

        elif "editor response round 2" in lower:
            text = (
                "APPROVED: The revised solution now addresses correctness concerns, includes concrete implementation steps, "
                "and provides clear validation guidance."
            )
        elif "critic response round 2" in lower:
            text = (
                "APPROVED: The revised pitch now names the user, shows concrete workflow features, "
                "addresses outsider skepticism, and has a demo path that is specific enough for evaluation."
            )
        elif "builder response round 2" in lower and ("editor response round" in lower or "editor -> builder" in lower):
            text = (
                "Implementation Plan:\n"
                "1. Create explicit workflow routing via a user parameter (`coding` vs `scientific-paper`).\n"
                "2. Keep Builder+Editor as the coding path and remove outsider dependence.\n"
                "3. Route scientific-paper workflow to seven personalities with conscientiousness approval.\n"
                "4. Update CLI and benchmark surfaces to accept and pass workflow selection.\n"
                "5. Add regression checks for both paths in mock mode.\n\n"
                "Validation Notes:\n"
                "- Confirm coding workflow runs Builder then Editor rounds with approval stop.\n"
                "- Confirm scientific-paper workflow runs personality collaboration."
            )
        elif "builder response round 2" in lower:
            text = (
                "MediBrief gives hospital physicians a specialty-specific command center for keeping up with new clinical research.\n\n"
                "Problem: Clinicians cannot reliably track the flood of new papers while managing patient care, documentation, and team handoffs.\n"
                "Target user: Hospital physicians, specialists, and residents who need trusted, workflow-ready research updates.\n"
                "Solution: MediBrief ingests new medical papers, ranks them by specialty and relevance, and turns them into cited, reviewable summaries.\n"
                "Key features:\n"
                "- Daily specialty feeds that triage papers by clinical relevance.\n"
                "- Structured evidence summaries covering findings, limitations, and likely clinical impact.\n"
                "- Department review queues where leads can approve, annotate, or assign papers to care teams.\n"
                "Differentiation: Unlike generic AI assistants, MediBrief is built around literature surveillance, citation grounding, and clinical team review.\n"
                "Risks: Incorrect synthesis, stale evidence, and overreliance. Mitigation: source-linked claims, recency filters, confidence flags, and mandatory human approval for clinical use.\n"
                "Demo plan: Show an oncology feed, open a new paper summary, verify citations, and route it through a physician approval workflow."
            )
        elif "editor response round 1" in lower:
            text = (
                "1. Replace high-level statements with implementation-level detail.\n"
                "2. Add concrete validation and testing strategy.\n"
                "3. Clarify edge cases and failure handling.\n"
                "4. Tighten language around assumptions and constraints."
            )
        elif "outsider response round 1" in lower:
            text = (
                "1. A skeptical hospital buyer will ask how this fits existing clinical systems instead of becoming another dashboard.\n"
                "2. The pitch should explain why clinicians would trust the summaries for high-stakes work.\n"
                "3. The value is clearer if the demo shows one complete before-and-after workflow, not only product screens."
            )
        elif "critic response round 1" in lower:
            text = (
                "1. Make the target user more specific than doctors.\n"
                "2. Tie the problem to daily clinical workflow pressure.\n"
                "3. Add review or approval features that show how teams would actually use it.\n"
                "4. Strengthen the risk section with concrete mitigation steps.\n"
                "5. Make the demo plan show a complete workflow from paper intake to approval."
            )
        elif "builder response round 1" in lower and ("editor response round" in lower or "editor -> builder" in lower):
            text = (
                "Draft Technical Revision:\n"
                "- Added concrete implementation steps instead of high-level intent.\n"
                "- Added validation strategy and explicit edge-case handling.\n"
                "- Clarified constraints, assumptions, and expected outcomes."
            )
        elif "builder response round 1" in lower or "revise the pitch" in lower or ("first draft:" in lower and "critic feedback" in lower):
            text = (
                "MediBrief turns the flood of new medical papers into fast, trustworthy updates for busy clinicians.\n\n"
                "Problem: Doctors cannot keep up with new research while balancing patient care, documentation, and compliance.\n"
                "Target user: Hospital physicians, specialists, and medical residents who need clinically relevant summaries.\n"
                "Solution: MediBrief scans new papers, ranks relevance by specialty, and generates structured summaries with citations.\n"
                "Key features:\n"
                "- Specialty-specific daily briefings with paper triage.\n"
                "- Evidence summaries that highlight findings, limitations, and clinical impact.\n"
                "- Team review mode so department leads can approve summaries and route them into existing clinical workflows.\n"
                "Differentiation: Unlike generic AI assistants, MediBrief is tuned for medical literature workflows and keeps every summary grounded in source citations.\n"
                "Risks: Hallucinated conclusions, stale literature, and clinician trust. Mitigation: citation checks, human review, recency filters, and source-visible confidence flags.\n"
                "Demo plan: Show one oncology feed, a paper summary, citation verification, and a physician approval flow from intake to team handoff."
            )
        elif "numbered suggestions" in lower or "return 4-6 numbered suggestions" in lower:
            text = (
                "1. State the target user explicitly and tie the pain point to their daily workflow.\n"
                "2. Add 3 concrete product features instead of broad promises.\n"
                "3. Explain why the product is differentiated from generic AI assistants.\n"
                "4. Name 1-2 risks and how the team would mitigate them.\n"
                "5. End with a crisp demo plan or rollout path."
            )
        else:
            if is_coding:
                text = (
                    "Draft Technical Plan:\n"
                    "- Define clear module boundaries and interfaces.\n"
                    "- Implement core logic with explicit error handling and edge-case coverage.\n"
                    "- Add tests for happy path, edge cases, and regression risks.\n"
                    "- Document assumptions, constraints, and verification steps."
                )
            elif is_doc:
                text = (
                    "Initial review baseline:\n"
                    "The draft states the core idea, but needs clearer structure, evidence framing, and explicit risk language."
                )
            else:
                text = (
                    "MediBrief is an AI tool that helps doctors summarize new medical papers.\n\n"
                    "It gives physicians quick summaries of research so they can stay current without reading every paper in full.\n"
                    "The product uses AI to read medical literature, condense the findings, and present key takeaways in a simple dashboard.\n"
                    "Doctors can search by topic and review recent publications faster."
                )

        latency_ms = (time.perf_counter() - start) * 1000 + 120
        tokens_in = estimate_tokens(prompt)
        tokens_out = estimate_tokens(text)
        return GenerationResult(
            text=text[:max_tokens * 5],
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_per_second=(tokens_out / max(latency_ms / 1000, 0.001)),
            backend="mock",
            model=model,
            raw_response={"mock": True, "temperature": temperature},
        )


class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(self, backend_name: BackendName, base_url: str, api_key: str) -> None:
        self.backend_name = backend_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start = time.perf_counter()
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        latency_ms = (time.perf_counter() - start) * 1000

        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens") or estimate_tokens(_extract_prompt(messages)))
        tokens_out = int(usage.get("completion_tokens") or estimate_tokens(text))
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_per_second=(tokens_out / max(latency_ms / 1000, 0.001)),
            backend=self.backend_name,
            model=model,
            raw_response=body,
        )


class LLMClient:
    def __init__(self) -> None:
        self.mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
        self.normal_base_url = os.getenv("NORMAL_LLM_BASE_URL", "http://localhost:8000/v1")
        self.normal_api_key = os.getenv("NORMAL_LLM_API_KEY", "dummy")
        self.normal_model = os.getenv("NORMAL_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.ssd_base_url = os.getenv("SSD_LLM_BASE_URL", "http://localhost:8001/v1")
        self.ssd_api_key = os.getenv("SSD_LLM_API_KEY", "dummy")
        self.ssd_model = os.getenv("SSD_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    def default_model_for(self, backend: BackendName) -> str:
        if backend == "ssd":
            return self.ssd_model
        return self.normal_model

    def _backend(self, backend: BackendName) -> BaseLLMBackend:
        if backend == "mock":
            return MockLLMBackend()
        if backend == "normal":
            return OpenAICompatibleBackend("normal", self.normal_base_url, self.normal_api_key)
        return OpenAICompatibleBackend("ssd", self.ssd_base_url, self.ssd_api_key)

    def generate(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        backend: BackendName,
    ) -> GenerationResult:
        return self._backend(backend).generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
