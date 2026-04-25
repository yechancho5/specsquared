# specsquared

CLI MVP for a NemoAgent-style multi-agent demo that compares single-agent output, multi-agent collaboration, and SSD-backed latency improvements through an OpenAI-compatible vLLM interface.

## What This Project Is

This project implements a small Python CLI that runs a pitch-refinement workflow with a Builder agent and a Critic agent. It is designed for hackathon demos where you want to show three things clearly:

- A two-agent workflow can produce a better answer than a single agent.
- Inter-agent communication changes the final answer in a visible way.
- An SSD-backed inference endpoint can reduce end-to-end latency compared with a normal endpoint.

The code does not implement SSD itself. It treats SSD as a swappable backend that exposes the same OpenAI-compatible API shape as the normal vLLM service.

## Architecture

```text
User Prompt
   |
   v
CLI (Typer)
   |
   v
Orchestrator
   |------------------------|
   v                        v
Builder Agent           Critic Agent
   |                        |
   |------ LLM Client ------|
             |
   -------------------------------
   |             |               |
Mock Backend   Normal vLLM   SSD-backed vLLM
   |
   v
Metrics + Diff + JSON Run Logs + Benchmark Tables
```

## Project Structure

```text
specsquared/
├── .env.example
├── README.md
├── requirements.txt
├── nemoagent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agents.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── llm.py
│   ├── metrics.py
│   ├── orchestrator.py
│   ├── render.py
│   ├── run_logger.py
│   └── schemas.py
├── prompts/
│   └── demo_prompts.txt
└── runs/
    └── .gitkeep
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

`MOCK_MODE=true` forces the mock backend for quick local development.

Normal endpoint:

- `NORMAL_LLM_BASE_URL`
- `NORMAL_LLM_API_KEY`
- `NORMAL_LLM_MODEL`

SSD endpoint:

- `SSD_LLM_BASE_URL`
- `SSD_LLM_API_KEY`
- `SSD_LLM_MODEL`

## Workflow Modes

- `single`: one Builder agent writes the final answer directly.
- `two-no-comm`: Builder drafts, Critic reviews, but the final output ignores critique.
- `two-normal`: Builder drafts, Critic reviews, Builder revises using critique on the normal backend.
- `two-ssd`: same as `two-normal`, but routed to the SSD-backed backend.

## Running Locally In Mock Mode

```bash
python -m nemoagent run \
  --mode single \
  --backend mock \
  --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

```bash
python -m nemoagent run \
  --mode two-normal \
  --backend mock \
  --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

## Running Against Normal vLLM On Brev

```bash
python -m nemoagent run \
  --mode two-normal \
  --backend normal \
  --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

## Running Against SSD-backed vLLM On Brev

```bash
python -m nemoagent run \
  --mode two-ssd \
  --backend ssd \
  --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

## Benchmark Command

```bash
python -m nemoagent bench \
  --prompt-file prompts/demo_prompts.txt \
  --modes single,two-no-comm,two-normal,two-ssd \
  --runs 3
```

## Example Commands

```bash
python -m nemoagent run --mode single --backend normal --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-no-comm --backend normal --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-normal --backend normal --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-ssd --backend ssd --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

## Example Benchmark Output

```text
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Mode         ┃ Backend ┃ Avg Total Latency ┃ Avg Quality ┃ Avg Suggestions Used  ┃ Avg Tokens/sec ┃ Avg Comm Effect ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ single       │ normal  │ 4.20s             │ 6.50        │ 0.00                  │ 42.10          │ 0.00            │
│ two-no-comm  │ normal  │ 7.80s             │ 6.70        │ 0.00                  │ 40.80          │ 0.00            │
│ two-normal   │ normal  │ 12.90s            │ 8.40        │ 3.00                  │ 39.50          │ 0.75            │
│ two-ssd      │ ssd     │ 6.10s             │ 8.30        │ 3.00                  │ 82.20          │ 0.75            │
└──────────────┴─────────┴───────────────────┴─────────────┴───────────────────────┴────────────────┴─────────────────┘
```

## What The Demo Proves

- Single-agent output is faster but weaker.
- Critique without incorporation is not enough.
- Two-agent revision improves quality.
- SSD preserves the collaborative workflow while lowering latency.

## Notes

- Mock mode works immediately and is deterministic enough for local development.
- Each run saves a JSON artifact in `runs/`.
- The terminal output includes agent sections, diff output, and metric summaries.
