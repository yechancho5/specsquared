from __future__ import annotations

from difflib import unified_diff
from typing import Optional

from .agents import BuilderAgent, EditorAgent, PersonalityAgent
from .llm import LLMClient
from .metrics import (
    communication_effect_score,
    count_suggestions_incorporated,
    parse_critic_suggestions,
    quality_score,
)
from .personalities import APPROVAL_AGENT, PERSONALITY_ORDER, detect_scenario, is_approved
from .run_logger import save_run_trace
from .schemas import AgentMessage, AgentResult, BenchmarkMetrics, RunArtifacts, RunConfig, RunTrace


class Orchestrator:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()
        self.builder = BuilderAgent(self.llm)
        self.editor = EditorAgent(self.llm)
        self.personality_agents = {name: PersonalityAgent(self.llm, name) for name in PERSONALITY_ORDER}

    @staticmethod
    def _resolve_scenario(config: RunConfig) -> str:
        if config.scenario != "auto":
            return config.scenario
        if config.workflow == "coding":
            return "coding"
        if config.workflow == "scientific-paper":
            return "document-review"
        return detect_scenario(config.prompt, config.scenario)

    @staticmethod
    def _append_message(
        messages: list[AgentMessage],
        config: RunConfig,
        result: AgentResult,
        to_agent: Optional[str] = None,
    ) -> None:
        messages.append(
            AgentMessage(
                run_id=config.run_id,
                from_agent=result.agent_name,
                to_agent=to_agent,
                role="assistant",
                content=result.output,
                metadata={"task": result.task, "latency_ms": result.latency_ms},
            )
        )

    @staticmethod
    def _accumulate_result(
        result: AgentResult,
        latency: dict[str, float],
        tokens_in: dict[str, int],
        tokens_out: dict[str, int],
    ) -> None:
        latency[result.agent_name] = latency.get(result.agent_name, 0.0) + result.latency_ms
        tokens_in[result.agent_name] = tokens_in.get(result.agent_name, 0) + result.tokens_in
        tokens_out[result.agent_name] = tokens_out.get(result.agent_name, 0) + result.tokens_out

    @staticmethod
    def _mock_latency(latency_ms: float, mode: str) -> float:
        """Simulate a faster SSD path during local mock demos."""
        multiplier = {
            "single": 1.0,
            "two-normal": 1.0,
            "two-ssd": 0.45,
        }.get(mode, 1.0)
        return round(latency_ms * multiplier, 2)

    def run_workflow(self, config: RunConfig) -> RunTrace:
        if config.workflow == "scientific-paper" or config.mode == "seven-personalities":
            return self._run_personality_workflow(config)
        return self._run_legacy_workflow(config)

    def _run_legacy_workflow(self, config: RunConfig) -> RunTrace:
        messages: list[AgentMessage] = []
        scenario = self._resolve_scenario(config)
        builder_draft = None
        editor_feedback = None
        final_output = ""
        editor_results = []
        revision_results = []

        builder_first = self.builder.draft(config.prompt, config)
        if config.backend == "mock":
            builder_first.latency_ms = self._mock_latency(builder_first.latency_ms, config.mode)
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

        editor_latency = 0.0
        revision_latency = 0.0
        time_to_first_output_ms = builder_first.latency_ms
        communication_enabled = False

        if config.mode == "single":
            final_output = builder_draft
        else:
            communication_enabled = True
            for round_number in range(1, config.dialogue_rounds + 1):
                editor_result = self.editor.respond(config.prompt, messages, config, round_number)
                if config.backend == "mock":
                    editor_result.latency_ms = self._mock_latency(editor_result.latency_ms, config.mode)
                editor_results.append(editor_result)
                editor_latency += editor_result.latency_ms
                messages.append(
                    AgentMessage(
                        run_id=config.run_id,
                        from_agent="editor",
                        to_agent="builder",
                        role="assistant",
                        content=editor_result.output,
                        metadata={"task": editor_result.task, "latency_ms": editor_result.latency_ms},
                    )
                )

                editor_feedback = "\n\n".join(result.output for result in editor_results)
                if final_output and editor_result.output.lower().startswith("approved:"):
                    break

                revision_result = self.builder.respond(config.prompt, messages, config, round_number)
                if config.backend == "mock":
                    revision_result.latency_ms = self._mock_latency(revision_result.latency_ms, config.mode)
                revision_results.append(revision_result)
                revision_latency += revision_result.latency_ms
                final_output = revision_result.output
                messages.append(
                    AgentMessage(
                        run_id=config.run_id,
                        from_agent="builder",
                        to_agent="editor",
                        role="assistant",
                        content=final_output,
                        metadata={"task": revision_result.task, "latency_ms": revision_result.latency_ms},
                    )
                )

        suggestions = parse_critic_suggestions(editor_feedback or "")
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
            total_latency_ms=round(builder_first.latency_ms + editor_latency + revision_latency, 2),
            time_to_first_output_ms=round(time_to_first_output_ms, 2),
            builder_draft_latency_ms=round(builder_first.latency_ms, 2),
            critic_latency_ms=round(editor_latency, 2),
            outsider_latency_ms=0.0,
            builder_revision_latency_ms=round(revision_latency, 2),
            tokens_in=0,
            tokens_out=0,
            tokens_per_second=0.0,
            critic_suggestions_count=len(suggestions),
            critic_suggestions_incorporated_count=incorporated_count,
            communication_enabled=communication_enabled,
            communication_effect_score=communication_effect_score(len(suggestions), incorporated_count),
            final_quality_score=quality_score(final_output, scenario),
            scenario=scenario,
        )

        token_in_total = builder_first.tokens_in
        token_out_total = builder_first.tokens_out
        if config.mode != "single":
            token_in_total += sum(result.tokens_in for result in editor_results)
            token_out_total += sum(result.tokens_out for result in editor_results)
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
            critic_feedback=editor_feedback,
            final_output=final_output,
            metrics=metrics,
            artifacts=RunArtifacts(diff=diff),
        )
        log_path = save_run_trace(trace)
        trace.artifacts.log_path = log_path
        return trace

    def _run_personality_workflow(self, config: RunConfig) -> RunTrace:
        messages: list[AgentMessage] = []
        scenario = self._resolve_scenario(config)
        latency_by_agent: dict[str, float] = {}
        tokens_in_by_agent: dict[str, int] = {}
        tokens_out_by_agent: dict[str, int] = {}

        synthesizer_name = "openness"
        approver_name = APPROVAL_AGENT
        reviewer_names = [name for name in config.personality_order if name not in {synthesizer_name, approver_name}]

        synthesizer = self.personality_agents[synthesizer_name]
        approver = self.personality_agents[approver_name]

        first_result = synthesizer.synthesize(
            prompt=config.prompt,
            messages=messages,
            config=config,
            round_number=0,
            scenario=scenario,
        )
        self._accumulate_result(first_result, latency_by_agent, tokens_in_by_agent, tokens_out_by_agent)
        self._append_message(messages, config, first_result)

        initial_output = first_result.output
        final_output = initial_output
        all_feedback: list[str] = []
        approver_required_fixes: list[str] = []
        approval_round: Optional[int] = None

        for round_number in range(1, config.dialogue_rounds + 1):
            for reviewer_name in reviewer_names:
                review_result = self.personality_agents[reviewer_name].review(
                    prompt=config.prompt,
                    messages=messages,
                    config=config,
                    round_number=round_number,
                    scenario=scenario,
                )
                self._accumulate_result(review_result, latency_by_agent, tokens_in_by_agent, tokens_out_by_agent)
                self._append_message(messages, config, review_result, to_agent=synthesizer_name)
                all_feedback.append(review_result.output)

            synthesis_result = synthesizer.synthesize(
                prompt=config.prompt,
                messages=messages,
                config=config,
                round_number=round_number,
                scenario=scenario,
            )
            self._accumulate_result(synthesis_result, latency_by_agent, tokens_in_by_agent, tokens_out_by_agent)
            self._append_message(messages, config, synthesis_result, to_agent=approver_name)
            final_output = synthesis_result.output

            approval_result = approver.approve(
                prompt=config.prompt,
                messages=messages,
                config=config,
                round_number=round_number,
                scenario=scenario,
            )
            self._accumulate_result(approval_result, latency_by_agent, tokens_in_by_agent, tokens_out_by_agent)
            self._append_message(messages, config, approval_result, to_agent=synthesizer_name)
            all_feedback.append(approval_result.output)

            if is_approved(approval_result.output):
                approval_round = round_number
                break
            approver_required_fixes.append(approval_result.output)

        critic_feedback = "\n\n".join(all_feedback) if all_feedback else None
        suggestions_source = "\n\n".join(approver_required_fixes) or (critic_feedback or "")
        suggestions = parse_critic_suggestions(suggestions_source)
        incorporated_count = count_suggestions_incorporated(suggestions, final_output)

        diff = None
        if initial_output != final_output:
            diff = "\n".join(
                unified_diff(
                    initial_output.splitlines(),
                    final_output.splitlines(),
                    fromfile="initial_output",
                    tofile="final_output",
                    lineterm="",
                )
            )

        total_latency = sum(latency_by_agent.values())
        total_tokens_in = sum(tokens_in_by_agent.values())
        total_tokens_out = sum(tokens_out_by_agent.values())

        metrics = BenchmarkMetrics(
            mode=config.mode,
            backend=config.backend,
            total_latency_ms=round(total_latency, 2),
            time_to_first_output_ms=round(first_result.latency_ms, 2),
            builder_draft_latency_ms=round(first_result.latency_ms, 2),
            critic_latency_ms=round(latency_by_agent.get(approver_name, 0.0), 2),
            outsider_latency_ms=round(latency_by_agent.get("sensitivity", 0.0), 2),
            builder_revision_latency_ms=round(max(0.0, latency_by_agent.get(synthesizer_name, 0.0) - first_result.latency_ms), 2),
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            tokens_per_second=round(total_tokens_out / max(total_latency / 1000, 0.001), 2),
            critic_suggestions_count=len(suggestions),
            critic_suggestions_incorporated_count=incorporated_count,
            communication_enabled=True,
            communication_effect_score=communication_effect_score(len(suggestions), incorporated_count),
            final_quality_score=quality_score(final_output, scenario),
            scenario=scenario,
            approval_agent=approver_name,
            approval_round=approval_round,
            agent_latency_ms={key: round(value, 2) for key, value in latency_by_agent.items()},
            agent_tokens_in=tokens_in_by_agent,
            agent_tokens_out=tokens_out_by_agent,
        )

        trace = RunTrace(
            config=config,
            messages=messages,
            builder_draft=initial_output,
            critic_feedback=critic_feedback,
            final_output=final_output,
            metrics=metrics,
            artifacts=RunArtifacts(diff=diff),
        )
        log_path = save_run_trace(trace)
        trace.artifacts.log_path = log_path
        return trace
