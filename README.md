# specsquared

CLI MVP for a NemoAgent-style multi-agent demo that compares single-agent output, legacy multi-agent collaboration, personality-driven multi-agent collaboration, and SSD-backed latency improvements through an OpenAI-compatible vLLM interface.

## What This Project Is

This project implements a small Python CLI with an explicit workflow selector:

- `coding`: legacy Builder + Editor collaboration.
- `scientific-paper`: seven-personality collaboration for scientific-paper review.

- The coding workflow keeps legacy Builder/Editor behavior.
- The scientific-paper workflow uses the new Big Seven personality system.

It is designed for demos where you want to compare multiple approaches clearly:

- A single-agent baseline.
- A legacy collaborative workflow on the normal backend.
- A seven-personality collaborative workflow on the normal backend.
- The same workflows on the SSD-backed backend (where applicable).

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
Builder/Editor Agent Set   or   Seven Personality Agent Set
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

## Workflow Selector

- `--workflow coding`: uses Builder + Editor logic.
- `--workflow scientific-paper`: uses seven-personality review logic.

This selector controls which agent logic executes.

## Mode Behavior

- `single`: one Builder agent writes the final answer directly.
- `two-normal`: collaborative rounds for the selected workflow on the normal backend.
- `two-ssd`: same as `two-normal`, routed to SSD backend.
- `seven-personalities`: forces seven-personality path (also used automatically when `--workflow scientific-paper` is selected).

Collaborative modes use `--dialogue-rounds` to control how many refinement rounds run. The default is `2`.

For scientific-paper workflow, `--scenario` can be set to `auto`, `coding`, `document-review`, or `general`. `auto` is default and infers context from the prompt.

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
  --workflow coding \
  --backend mock \
  --prompt "Implement a robust Python module and include testing strategy."
```

```bash
python -m nemoagent run \
  --mode two-normal \
  --workflow scientific-paper \
  --backend mock \
  --scenario auto \
  --prompt "Review this scientific abstract for clarity, evidence quality, and methodological risks."
```

## Running Against Normal vLLM On Brev

```bash
python -m nemoagent run \
  --mode two-normal \
  --workflow coding \
  --backend normal \
  --prompt "Implement a robust Python module and include testing strategy."
```

## Running Against SSD-backed vLLM On Brev

```bash
python -m nemoagent run \
  --mode two-ssd \
  --workflow coding \
  --backend ssd \
  --prompt "Implement a robust Python module and include testing strategy."
```

## Benchmark Command

```bash
python -m nemoagent bench \
  --prompt-file prompts/demo_prompts.txt \
  --workflow coding \
  --modes single,two-normal,two-ssd,seven-personalities \
  --runs 3
```

## Example Commands

```bash
python -m nemoagent run --mode two-normal --workflow coding --backend normal --prompt "Implement a robust Python module and include testing strategy."
python -m nemoagent run --mode two-ssd --workflow coding --backend ssd --prompt "Implement a robust Python module and include testing strategy."
python -m nemoagent run --mode two-normal --workflow scientific-paper --backend normal --scenario auto --prompt "Review and improve a scientific paper abstract for clarity and risk coverage."
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
- Legacy multi-turn Builder/Critic/Outsider communication changes the final answer in a visible way.
- Seven-personality collaboration provides broader critique coverage and a strict approval gate.
- SSD preserves collaborative workflows while lowering latency.

## Local Demo Notes

- In `mock` mode, the content is deterministic so the workflow is easy to demo repeatedly.
- `two-ssd` in `mock` mode simulates a faster latency profile so the benchmark still shows the intended SSD speedup story locally.
- Real `normal` and `ssd` comparisons require live OpenAI-compatible endpoints.

## Notes

- Mock mode works immediately and is deterministic enough for local development.
- Each run saves a JSON artifact in `runs/`.
- The terminal output includes the agent conversation, diff output, and metric summaries.

## Additional Documentation

- Seven personalities origin and MAPS attribution: [docs/seven-personalities-origin.md](docs/seven-personalities-origin.md)
