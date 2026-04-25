from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .llm import LLMClient
from .orchestrator import Orchestrator
from .run_logger import save_benchmark_summary
from .schemas import BenchmarkSummaryRow, RunConfig, ScenarioType, WorkflowMode, WorkflowType


def load_prompts(prompt_file: str) -> list[str]:
    content = Path(prompt_file).read_text(encoding="utf-8")
    prompts = [chunk.strip() for chunk in content.splitlines() if chunk.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def run_benchmark(
    prompt_file: str,
    modes: list[WorkflowMode],
    runs: int,
    backend_override: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    dialogue_rounds: int = 2,
    scenario: ScenarioType = "auto",
    workflow: WorkflowType = "coding",
) -> tuple[list[BenchmarkSummaryRow], str]:
    llm = LLMClient()
    orchestrator = Orchestrator(llm=llm)
    prompts = load_prompts(prompt_file)
    results: dict[tuple[str, str], list] = {}

    for prompt in prompts:
        for mode in modes:
            backend = backend_override or ("ssd" if mode == "two-ssd" else "normal")
            for _ in range(runs):
                config = RunConfig(
                    mode=mode,
                    workflow=workflow,
                    backend=backend,  # type: ignore[arg-type]
                    prompt=prompt,
                    model=model or llm.default_model_for(backend),  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    dialogue_rounds=dialogue_rounds,
                    scenario=scenario,
                )
                trace = orchestrator.run_workflow(config)
                results.setdefault((mode, backend), []).append(trace.metrics)

    rows: list[BenchmarkSummaryRow] = []
    serializable_rows: list[dict] = []
    for (mode, backend), metrics_list in results.items():
        count = len(metrics_list)
        row = BenchmarkSummaryRow(
            mode=mode,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            average_total_latency_ms=round(sum(m.total_latency_ms for m in metrics_list) / count, 2),
            average_quality_score=round(sum(m.final_quality_score for m in metrics_list) / count, 2),
            average_tokens_per_second=round(sum(m.tokens_per_second for m in metrics_list) / count, 2),
            average_communication_effect_score=round(
                sum(m.communication_effect_score for m in metrics_list) / count,
                2,
            ),
            average_suggestions_incorporated=round(
                sum(m.critic_suggestions_incorporated_count for m in metrics_list) / count,
                2,
            ),
            runs=count,
        )
        rows.append(row)
        serializable_rows.append(row.model_dump())

    rows.sort(key=lambda row: (row.mode, row.backend))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_file": prompt_file,
        "runs_per_prompt": runs,
        "rows": serializable_rows,
    }
    saved_path = save_benchmark_summary(summary)
    return rows, saved_path
