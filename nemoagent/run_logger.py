from __future__ import annotations

import json
from pathlib import Path

from .schemas import RunTrace


def save_run_trace(trace: RunTrace, runs_dir: str = "runs") -> str:
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = trace.config.timestamp.replace(":", "-")
    filename = f"{timestamp}_{trace.config.mode}_{trace.config.backend}_{trace.config.run_id}.json"
    target = path / filename
    target.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return str(target)


def save_benchmark_summary(summary: dict, runs_dir: str = "runs") -> str:
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"benchmark_{summary['timestamp'].replace(':', '-')}.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(target)
