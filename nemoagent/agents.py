from __future__ import annotations

from .llm import LLMClient
from .schemas import AgentResult, BackendName, RunConfig


class BuilderAgent:
    name = "builder"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def draft(self, prompt: str, config: RunConfig) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Builder Agent creating a concise, convincing startup pitch. "
                        "Include problem, target user, solution, features, differentiation, risks, and a demo plan."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            backend=config.backend,
        )
        return AgentResult(
            agent_name=self.name,
            task="draft",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )

    def revise(self, prompt: str, draft: str, critique: str, config: RunConfig) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Builder Agent revising a startup pitch using critic feedback. "
                        "Incorporate the feedback directly and keep the result sharper than the original."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original prompt:\n{prompt}\n\n"
                        f"First draft:\n{draft}\n\n"
                        f"Critic feedback with numbered suggestions:\n{critique}\n\n"
                        "Revise the pitch so the suggestions are clearly incorporated."
                    ),
                },
            ],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            backend=config.backend,
        )
        return AgentResult(
            agent_name=self.name,
            task="revise",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def critique(self, prompt: str, draft: str, config: RunConfig) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Critic Agent. Review the pitch and return numbered suggestions only. "
                        "Be concrete and focus on improvements that visibly change the final output."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User prompt:\n{prompt}\n\n"
                        f"Builder draft:\n{draft}\n\n"
                        "Return 4-6 numbered suggestions that improve clarity, specificity, differentiation, risks, and demo readiness."
                    ),
                },
            ],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            backend=config.backend,
        )
        return AgentResult(
            agent_name=self.name,
            task="critique",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )
