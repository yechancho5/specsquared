from __future__ import annotations

from .llm import LLMClient
from .personalities import (
    PERSONALITIES,
    approver_system_prompt,
    reviewer_system_prompt,
    synthesizer_system_prompt,
)
from .schemas import AgentMessage, AgentResult, PersonalityName, RunConfig, ScenarioType


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
                        "You are a Builder Agent creating a first technical solution draft. "
                        "Prioritize concrete implementation details, correctness, trade-offs, and testability."
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
                        "You are a Builder Agent revising a technical draft using editor feedback. "
                        "Incorporate required edits directly and keep the result more actionable than the original."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original prompt:\n{prompt}\n\n"
                        f"First draft:\n{draft}\n\n"
                        f"Editor feedback with numbered suggestions:\n{critique}\n\n"
                        "Revise the draft so the suggestions are clearly incorporated."
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
                        "You are the Builder Agent in a live collaboration with an Editor Agent. "
                        "Use the full transcript, address unresolved edits, and produce a stronger technical output."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is builder response round {round_number}. "
                        "Reply with the revised output only, incorporating the latest Editor feedback."
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


class EditorAgent(CriticAgent):
    name = "editor"

    def critique(self, prompt: str, draft: str, config: RunConfig) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Editor Agent for coding and technical deliverables. "
                        "Review the draft and return numbered improvement suggestions only. "
                        "Focus on correctness, clarity, maintainability, testing, and practical implementation detail."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User prompt:\n{prompt}\n\n"
                        f"Builder draft:\n{draft}\n\n"
                        "Return 4-6 numbered suggestions that improve correctness, specificity, testability, risks, and implementation quality."
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
            task="edit",
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
                        "You are the Editor Agent in a live collaboration with a Builder Agent. "
                        "Review the latest builder output in the transcript. "
                        "If it is ready, start with 'APPROVED:' and explain briefly. "
                        "Otherwise return numbered, concrete edits the Builder must address next."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is editor response round {round_number}. "
                        "Respond directly to the Builder's latest output."
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
            task=f"dialogue-edit-{round_number}",
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


class PersonalityAgent:
    def __init__(self, llm: LLMClient, name: PersonalityName) -> None:
        self.llm = llm
        self.name = name
        self.profile = PERSONALITIES[name]

    def review(
        self,
        prompt: str,
        messages: list[AgentMessage],
        config: RunConfig,
        round_number: int,
        scenario: ScenarioType,
    ) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": reviewer_system_prompt(self.profile, scenario),
                },
                {
                    "role": "user",
                    "content": (
                        f"Scenario context: {scenario}\n\n"
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is {self.name} review round {round_number}. "
                        "Respond with specific updates for the synthesizer."
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
            task=f"dialogue-{self.name}-review-{round_number}",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )

    def synthesize(
        self,
        prompt: str,
        messages: list[AgentMessage],
        config: RunConfig,
        round_number: int,
        scenario: ScenarioType,
    ) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": synthesizer_system_prompt(self.profile, scenario),
                },
                {
                    "role": "user",
                    "content": (
                        f"Scenario context: {scenario}\n\n"
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is {self.name} synthesis round {round_number}. "
                        "Produce a revised artifact that resolves recent feedback."
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
            task=f"dialogue-{self.name}-synthesis-{round_number}",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )

    def approve(
        self,
        prompt: str,
        messages: list[AgentMessage],
        config: RunConfig,
        round_number: int,
        scenario: ScenarioType,
    ) -> AgentResult:
        result = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": approver_system_prompt(self.profile, scenario),
                },
                {
                    "role": "user",
                    "content": (
                        f"Scenario context: {scenario}\n\n"
                        f"Original user prompt:\n{prompt}\n\n"
                        f"Conversation transcript:\n{_format_transcript(messages)}\n\n"
                        f"This is {self.name} approval round {round_number}. "
                        "Approve only if the output is ready. Otherwise return required fixes."
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
            task=f"dialogue-{self.name}-approval-{round_number}",
            output=result.text,
            latency_ms=result.latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            backend=result.backend,
            model=result.model,
        )
