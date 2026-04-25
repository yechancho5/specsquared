from __future__ import annotations

from difflib import unified_diff
from typing import Optional

from .agents import BuilderAgent, CriticAgent, OutsiderAgent
from .llm import LLMClient
from .metrics import (
    communication_effect_score,
    count_suggestions_incorporated,
    parse_critic_suggestions,
    quality_score,
)
from .run_logger import save_run_trace
from .schemas import AgentMessage, BenchmarkMetrics, RunArtifacts, RunConfig, RunTrace


class Orchestrator:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()
        self.builder = BuilderAgent(self.llm)
        self.critic = CriticAgent(self.llm)
        self.outsider = OutsiderAgent(self.llm)

    def run_workflow(self, config: RunConfig) -> RunTrace:
        messages: list[AgentMessage] = []
        builder_draft = None
        critic_feedback = None
        final_output = ""
        critic_results = []
        outsider_results = []
        revision_results = []

        builder_first = self.builder.draft(config.prompt, config)
        builder_draft = builder_first.output
        messages.append(
            AgentMessage(
                run_id=config.run_id,
                from_agent="builder",
                role="assistant",
                content=builder_draft,
                metadata={"task": "draft", "latency_ms": builder_first.latency_ms},
            )
        )

        critic_latency = 0.0
        outsider_latency = 0.0
        revision_latency = 0.0
        time_to_first_output_ms = builder_first.latency_ms
        communication_enabled = False

        if config.mode == "single":
            final_output = builder_draft
        else:
            communication_enabled = True
            for round_number in range(1, config.dialogue_rounds + 1):
                critic_result = self.critic.respond(config.prompt, messages, config, round_number)
                critic_results.append(critic_result)
                critic_latency += critic_result.latency_ms
                messages.append(
                    AgentMessage(
                        run_id=config.run_id,
                        from_agent="critic",
                        to_agent="builder",
                        role="assistant",
                        content=critic_result.output,
                        metadata={"task": critic_result.task, "latency_ms": critic_result.latency_ms},
                    )
                )

                critic_feedback = "\n\n".join(result.output for result in critic_results)
                if final_output and critic_result.output.lower().startswith("approved:"):
                    break

                outsider_result = self.outsider.respond(config.prompt, messages, config, round_number)
                outsider_results.append(outsider_result)
                outsider_latency += outsider_result.latency_ms
                messages.append(
                    AgentMessage(
                        run_id=config.run_id,
                        from_agent="outsider",
                        to_agent="builder",
                        role="assistant",
                        content=outsider_result.output,
                        metadata={"task": outsider_result.task, "latency_ms": outsider_result.latency_ms},
                    )
                )

                revision_result = self.builder.respond(config.prompt, messages, config, round_number)
                revision_results.append(revision_result)
                revision_latency += revision_result.latency_ms
                final_output = revision_result.output
                messages.append(
                    AgentMessage(
                        run_id=config.run_id,
                        from_agent="builder",
                        to_agent="critic",
                        role="assistant",
                        content=final_output,
                        metadata={"task": revision_result.task, "latency_ms": revision_result.latency_ms},
                    )
                )

        suggestions = parse_critic_suggestions(critic_feedback or "")
        incorporated_count = count_suggestions_incorporated(suggestions, final_output)
        diff = None
        if builder_draft and final_output and builder_draft != final_output:
            diff = "\n".join(
                unified_diff(
                    builder_draft.splitlines(),
                    final_output.splitlines(),
                    fromfile="builder_draft",
                    tofile="final_output",
                    lineterm="",
                )
            )

        metrics = BenchmarkMetrics(
            mode=config.mode,
            backend=config.backend,
            total_latency_ms=round(builder_first.latency_ms + critic_latency + outsider_latency + revision_latency, 2),
            time_to_first_output_ms=round(time_to_first_output_ms, 2),
            builder_draft_latency_ms=round(builder_first.latency_ms, 2),
            critic_latency_ms=round(critic_latency, 2),
            outsider_latency_ms=round(outsider_latency, 2),
            builder_revision_latency_ms=round(revision_latency, 2),
            tokens_in=0,
            tokens_out=0,
            tokens_per_second=0.0,
            critic_suggestions_count=len(suggestions),
            critic_suggestions_incorporated_count=incorporated_count,
            communication_enabled=communication_enabled,
            communication_effect_score=communication_effect_score(len(suggestions), incorporated_count),
            final_quality_score=quality_score(final_output),
        )

        token_in_total = builder_first.tokens_in
        token_out_total = builder_first.tokens_out
        if config.mode != "single":
            token_in_total += sum(result.tokens_in for result in critic_results)
            token_out_total += sum(result.tokens_out for result in critic_results)
            token_in_total += sum(result.tokens_in for result in outsider_results)
            token_out_total += sum(result.tokens_out for result in outsider_results)
            token_in_total += sum(result.tokens_in for result in revision_results)
            token_out_total += sum(result.tokens_out for result in revision_results)

        metrics.tokens_in = token_in_total
        metrics.tokens_out = token_out_total
        metrics.tokens_per_second = round(
            token_out_total / max(metrics.total_latency_ms / 1000, 0.001),
            2,
        )

        trace = RunTrace(
            config=config,
            messages=messages,
            builder_draft=builder_draft,
            critic_feedback=critic_feedback,
            final_output=final_output,
            metrics=metrics,
            artifacts=RunArtifacts(diff=diff),
        )
        log_path = save_run_trace(trace)
        trace.artifacts.log_path = log_path
        return trace
