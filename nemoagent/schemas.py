from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


WorkflowMode = Literal["single", "two-no-comm", "two-normal", "two-ssd"]
BackendName = Literal["mock", "normal", "ssd"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid4().hex[:12]


class GenerationResult(BaseModel):
    text: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    tokens_per_second: float
    backend: BackendName
    model: str
    raw_response: dict[str, Any] | None = None


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    from_agent: str
    to_agent: str | None = None
    role: str
    content: str
    timestamp: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_name: str
    task: str
    output: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    backend: BackendName
    model: str


class RunConfig(BaseModel):
    run_id: str = Field(default_factory=new_run_id)
    mode: WorkflowMode
    backend: BackendName
    prompt: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 700
    timestamp: str = Field(default_factory=utc_now)


class BenchmarkMetrics(BaseModel):
    mode: WorkflowMode
    backend: BackendName
    total_latency_ms: float
    time_to_first_output_ms: float
    builder_draft_latency_ms: float = 0.0
    critic_latency_ms: float = 0.0
    builder_revision_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_per_second: float = 0.0
    critic_suggestions_count: int = 0
    critic_suggestions_incorporated_count: int = 0
    communication_enabled: bool = False
    communication_effect_score: float = 0.0
    final_quality_score: float = 0.0


class RunArtifacts(BaseModel):
    log_path: str | None = None
    diff: str | None = None


class RunTrace(BaseModel):
    config: RunConfig
    messages: list[AgentMessage] = Field(default_factory=list)
    builder_draft: str | None = None
    critic_feedback: str | None = None
    final_output: str
    metrics: BenchmarkMetrics
    artifacts: RunArtifacts = Field(default_factory=RunArtifacts)


class BenchmarkSummaryRow(BaseModel):
    mode: WorkflowMode
    backend: BackendName
    average_total_latency_ms: float
    average_quality_score: float
    average_tokens_per_second: float
    average_communication_effect_score: float
    average_suggestions_incorporated: float
    runs: int
