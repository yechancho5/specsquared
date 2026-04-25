# specsquared

CLI MVP for a NemoAgent-style multi-agent demo that compares single-agent output, multi-agent collaboration, and SSD-backed latency improvements through an OpenAI-compatible vLLM interface.

## What This Project Is

This project implements a small Python CLI that runs a pitch-refinement workflow with a Builder agent, a Critic agent, and an Outsider agent. It is designed for hackathon demos where you want to show three comparisons clearly:

- A single-agent baseline.
- A two-agent workflow on the normal backend.
- The same two-agent workflow on the SSD-backed backend.

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
   |------------------------|------------------------|
   v                        v                        v
Builder Agent           Critic Agent           Outsider Agent
   |                        |                        |
   |------------------ LLM Client ------------------|
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
- `two-normal`: Builder, Critic, and Outsider exchange multiple transcript-aware messages on the normal backend before the final answer.
- `two-ssd`: same agent dialogue as `two-normal`, but routed to the SSD-backed backend.

Multi-agent modes use `--dialogue-rounds` to control how many Critic/Outsider/Builder exchanges happen after the first Builder draft. The default is `2`.

## Running Locally In Mock Mode

Best local demo sequence:

```bash
python -m nemoagent run --mode single --backend mock --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-no-comm --backend mock --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-normal --backend mock --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-ssd --backend mock --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

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
  --modes single,two-normal,two-ssd \
  --runs 3
```

## Example Commands

```bash
python -m nemoagent run --mode single --backend normal --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-normal --backend normal --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
python -m nemoagent run --mode two-ssd --backend ssd --prompt "Create a pitch for an AI tool that helps doctors summarize medical papers."
```

## Example Benchmark Output

```text
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Mode         ┃ Backend ┃ Avg Total Latency ┃ Avg Quality ┃ Avg Suggestions Used  ┃ Avg Tokens/sec ┃ Avg Comm Effect ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ single       │ normal  │ 4.20s             │ 6.50        │ 0.00                  │ 42.10          │ 0.00            │
│ two-normal   │ normal  │ 14.80s            │ 8.70        │ 4.00                  │ 38.20          │ 0.80            │
│ two-ssd      │ ssd     │ 7.10s             │ 8.70        │ 4.00                  │ 81.50          │ 0.80            │
└──────────────┴─────────┴───────────────────┴─────────────┴───────────────────────┴────────────────┴─────────────────┘
```

## What The Demo Proves

- Single-agent output is the baseline.
- Multi-turn Builder/Critic/Outsider communication changes the final answer in a visible way.
- SSD preserves the same collaborative workflow while lowering latency.

## Local Demo Notes

- In `mock` mode, the content is deterministic so the workflow is easy to demo repeatedly.
- `two-ssd` in `mock` mode simulates a faster latency profile so the benchmark still shows the intended SSD speedup story locally.
- Real `normal` and `ssd` comparisons require live OpenAI-compatible endpoints.

## Notes

- Mock mode works immediately and is deterministic enough for local development.
- Each run saves a JSON artifact in `runs/`.
- The terminal output includes the agent conversation, diff output, and metric summaries.
