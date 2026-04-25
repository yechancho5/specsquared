from __future__ import annotations

from typing import Optional

import typer
from dotenv import load_dotenv

from .benchmark import run_benchmark
from .llm import LLMClient
from .orchestrator import Orchestrator
from .render import render_benchmark, render_run
from .schemas import BackendName, RunConfig, ScenarioType, WorkflowMode, WorkflowType


app = typer.Typer(help="CLI NemoAgent demo for legacy and personality-based multi-agent refinement.")


def resolve_backend(mode: WorkflowMode, backend: Optional[BackendName]) -> BackendName:
    if backend is not None:
        return backend
    if mode == "two-ssd":
        return "ssd"
    if mode in {"single", "two-normal", "seven-personalities"}:
        return "normal"
    return "mock"


@app.command()
def run(
    mode: WorkflowMode = typer.Option(..., help="Workflow mode."),
    workflow: WorkflowType = typer.Option("coding", help="Logic workflow: coding or scientific-paper."),
    prompt: str = typer.Option(..., help="User prompt."),
    backend: Optional[BackendName] = typer.Option(None, help="Inference backend."),
    model: Optional[str] = typer.Option(None, help="Model name."),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    max_tokens: int = typer.Option(700, help="Max output tokens."),
    dialogue_rounds: int = typer.Option(2, min=1, help="Dialogue rounds for collaborative modes."),
    scenario: ScenarioType = typer.Option(
        "auto",
        help="Context hint used by personality mode: auto, coding, document-review, or general.",
    ),
) -> None:
    load_dotenv()
    llm = LLMClient()
    resolved_backend = resolve_backend(mode, backend)
    config = RunConfig(
        mode=mode,
        workflow=workflow,
        backend=resolved_backend,
        prompt=prompt,
        model=model or llm.default_model_for(resolved_backend),
        temperature=temperature,
        max_tokens=max_tokens,
        dialogue_rounds=dialogue_rounds,
        scenario=scenario,
    )
    trace = Orchestrator(llm=llm).run_workflow(config)
    render_run(trace)


@app.command()
def bench(
    prompt_file: str = typer.Option(..., help="Path to prompt file."),
    modes: str = typer.Option("single,two-normal,two-ssd,seven-personalities", help="Comma-separated modes."),
    workflow: WorkflowType = typer.Option("coding", help="Logic workflow: coding or scientific-paper."),
    runs: int = typer.Option(3, help="Runs per prompt and mode."),
    backend: Optional[BackendName] = typer.Option(None, help="Optional backend override."),
    model: Optional[str] = typer.Option(None, help="Optional model override."),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    max_tokens: int = typer.Option(700, help="Max output tokens."),
    dialogue_rounds: int = typer.Option(2, min=1, help="Dialogue rounds for collaborative modes."),
    scenario: ScenarioType = typer.Option(
        "auto",
        help="Scenario hint for personality mode: auto, coding, document-review, or general.",
    ),
) -> None:
    load_dotenv()
    selected_modes = [mode.strip() for mode in modes.split(",") if mode.strip()]
    rows, saved_path = run_benchmark(
        prompt_file=prompt_file,
        modes=selected_modes,  # type: ignore[arg-type]
        runs=runs,
        backend_override=backend,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        dialogue_rounds=dialogue_rounds,
        scenario=scenario,
        workflow=workflow,
    )
    render_benchmark(rows, saved_path=saved_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
