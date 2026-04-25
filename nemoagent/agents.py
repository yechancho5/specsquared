from __future__ import annotations

from .llm import LLMClient
from .schemas import AgentMessage, AgentResult, RunConfig


def _format_transcript(messages: list[AgentMessage]) -> str:
    if not messages:
        return "(no agent messages yet)"
    lines = []
    for message in messages:
        recipient = f" -> {message.to_agent}" if message.to_agent else ""
        task = message.metadata.get("task", "message")
        lines.append(f"{message.from_agent}{recipient} [{task}]:\n{message.content}")
    return "\n\n".join(lines)


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

    def respond(self, prompt: str, messages: list[AgentMessage], config: RunConfig, round_number: int) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Builder Agent in a live collaboration with a Critic Agent and an Outsider Agent. "
                        "Use the full transcript, answer the latest concerns, and produce a stronger pitch. "
                        "Do not ignore unresolved critic feedback or outsider skepticism."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is builder response round {round_number}. "
                        "Reply with the revised pitch only, incorporating the latest Critic and Outsider messages."
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
            task=f"dialogue-revise-{round_number}",
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

    def respond(self, prompt: str, messages: list[AgentMessage], config: RunConfig, round_number: int) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Critic Agent in a live collaboration with a Builder Agent. "
                        "Review the latest builder pitch in the transcript. "
                        "If it is ready, start with 'APPROVED:' and explain briefly. "
                        "Otherwise return numbered, concrete suggestions that the Builder must address next."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is critic response round {round_number}. "
                        "Respond directly to the Builder's latest pitch."
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
            task=f"dialogue-critique-{round_number}",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )


class OutsiderAgent:
    name = "outsider"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def respond(self, prompt: str, messages: list[AgentMessage], config: RunConfig, round_number: int) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Outsider Agent: a neutral, unbiased reviewer who is not invested in the Builder's idea. "
                        "Evaluate the latest pitch like a skeptical buyer, investor, or end user. "
                        "Focus on plain-language clarity, credibility, missing context, and whether the pitch would convince someone cold."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is outsider response round {round_number}. "
                        "Give 2-4 concise, unbiased observations the Builder should account for next."
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
            task=f"dialogue-outsider-{round_number}",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )
