from __future__ import annotations

from typing import Any

from .schemas import BenchmarkSummaryRow, RunTrace

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    Console = Any  # type: ignore[assignment]
    Panel = Any  # type: ignore[assignment]
    Table = Any  # type: ignore[assignment]
    RICH_AVAILABLE = False


def _console() -> Console | None:
    if not RICH_AVAILABLE:
        return None
    return Console()


def render_run(trace: RunTrace) -> None:
    console = _console()
    if console is None:
        print(f"Mode: {trace.config.mode}")
        print(f"Backend: {trace.config.backend}")
        print(f"Model: {trace.config.model}")
        print("\nBuilder Draft:\n")
        print(trace.builder_draft or "")
        if trace.critic_feedback:
            print("\nCritic Feedback:\n")
            print(trace.critic_feedback)
        print("\nFinal Output:\n")
        print(trace.final_output)
        if trace.artifacts.diff:
            print("\nDiff:\n")
            print(trace.artifacts.diff)
        print("\nMetrics:")
        print(trace.metrics.model_dump_json(indent=2))
        print(f"\nRun log: {trace.artifacts.log_path}")
        return

    console.print(
        Panel.fit(
            f"[bold]Mode:[/bold] {trace.config.mode}\n"
            f"[bold]Backend:[/bold] {trace.config.backend}\n"
            f"[bold]Model:[/bold] {trace.config.model}",
            title="NemoAgent Run",
        )
    )
    console.print(Panel(trace.builder_draft or "", title="Builder Draft", border_style="cyan"))
    if trace.critic_feedback:
        console.print(Panel(trace.critic_feedback, title="Critic Feedback", border_style="yellow"))
    console.print(Panel(trace.final_output, title="Final Output", border_style="green"))
    if trace.artifacts.diff:
        console.print(Panel(trace.artifacts.diff, title="Before / After Diff", border_style="magenta"))

    table = Table(title="Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    metrics = trace.metrics.model_dump()
    for key, value in metrics.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"[bold]Run log:[/bold] {trace.artifacts.log_path}")


def render_benchmark(rows: list[BenchmarkSummaryRow], saved_path: str | None = None) -> None:
    console = _console()
    if console is None:
        print("Benchmark Summary")
        for row in rows:
            print(row.model_dump())
        if saved_path:
            print(f"\nSaved summary: {saved_path}")
        return

    table = Table(title="Benchmark Summary")
    table.add_column("Mode")
    table.add_column("Backend")
    table.add_column("Avg Total Latency")
    table.add_column("Avg Quality")
    table.add_column("Avg Suggestions Used")
    table.add_column("Avg Tokens/sec")
    table.add_column("Avg Comm Effect")
    for row in rows:
        table.add_row(
            row.mode,
            row.backend,
            f"{row.average_total_latency_ms / 1000:.2f}s",
            f"{row.average_quality_score:.2f}",
            f"{row.average_suggestions_incorporated:.2f}",
            f"{row.average_tokens_per_second:.2f}",
            f"{row.average_communication_effect_score:.2f}",
        )
    console.print(table)
    if saved_path:
        console.print(f"[bold]Saved summary:[/bold] {saved_path}")
