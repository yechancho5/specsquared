from __future__ import annotations

from typing import Optional

import typer
from dotenv import load_dotenv

from .benchmark import run_benchmark
from .llm import LLMClient
from .orchestrator import Orchestrator
from .render import render_benchmark, render_run
from .schemas import BackendName, RunConfig, WorkflowMode


app = typer.Typer(help="CLI NemoAgent demo for multi-agent pitch refinement.")


def resolve_backend(mode: WorkflowMode, backend: Optional[BackendName]) -> BackendName:
    if backend is not None:
        return backend
    if mode == "two-ssd":
        return "ssd"
    if mode in {"single", "two-normal"}:
        return "normal"
    return "mock"


@app.command()
def run(
    mode: WorkflowMode = typer.Option(..., help="Workflow mode."),
    prompt: str = typer.Option(..., help="User prompt."),
    backend: Optional[BackendName] = typer.Option(None, help="Inference backend."),
    model: Optional[str] = typer.Option(None, help="Model name."),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    max_tokens: int = typer.Option(700, help="Max output tokens."),
    dialogue_rounds: int = typer.Option(2, min=1, help="Critic/Builder dialogue rounds for two-agent modes."),
) -> None:
    load_dotenv()
    llm = LLMClient()
    resolved_backend = resolve_backend(mode, backend)
    config = RunConfig(
        mode=mode,
        backend=resolved_backend,
        prompt=prompt,
        model=model or llm.default_model_for(resolved_backend),
        temperature=temperature,
        max_tokens=max_tokens,
        dialogue_rounds=dialogue_rounds,
    )
    trace = Orchestrator(llm=llm).run_workflow(config)
    render_run(trace)


@app.command()
def bench(
    prompt_file: str = typer.Option(..., help="Path to prompt file."),
    modes: str = typer.Option("single,two-normal,two-ssd", help="Comma-separated modes."),
    runs: int = typer.Option(3, help="Runs per prompt and mode."),
    backend: Optional[BackendName] = typer.Option(None, help="Optional backend override."),
    model: Optional[str] = typer.Option(None, help="Optional model override."),
    temperature: float = typer.Option(0.2, help="Sampling temperature."),
    max_tokens: int = typer.Option(700, help="Max output tokens."),
    dialogue_rounds: int = typer.Option(2, min=1, help="Critic/Builder dialogue rounds for two-agent modes."),
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
    )
    render_benchmark(rows, saved_path=saved_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
