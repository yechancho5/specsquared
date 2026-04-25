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

        if "critic response round 2" in lower:
            text = (
                "APPROVED: The revised pitch now names the user, shows concrete workflow features, "
                "addresses outsider skepticism, and has a demo path that is specific enough for evaluation."
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
