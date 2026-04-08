---
title: PipePulse OpenEnv
emoji: 💧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - civic-tech
  - water-utilities
  - incident-response
---

# PipePulse — Municipal Water Operations OpenEnv

PipePulse is a deterministic RL environment for city water operations where an agent must:
- dispatch leak repair crews,
- operate isolation valves,
- preserve service continuity for critical facilities,
- and contain contamination spread.

## Why this is real-world

Utilities handle leaks, isolation valves, and contamination response under budget and SLA pressure. PipePulse models those tradeoffs as a strict OpenEnv environment with deterministic grading.

## OpenEnv API

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grade`
- `GET /health`

Typed Pydantic spaces are defined in `models.py`:
- `Observation`
- `Action`
- `Reward`

## Action space

`Action.action_type` supports:
- `assign_crew`
- `reroute_crew`
- `hold`
- `open_valve`
- `close_valve`
- `flush_segment`

## Observation highlights

Each observation includes:
- active leaks and crew states
- network `segments` + `valves`
- `service_outage_units` and `critical_facility_outages`
- contamination status (`contamination_risk_index`, `contaminated_segments`)

## Tasks (easy → hard+)

1. `easy_single_crew` — single-crew leak response baseline.
2. `medium_valve_tradeoff` — valve isolation vs outage tradeoffs.
3. `hard_burst_fairness_budget` — burst propagation + fairness + budget pressure.
4. `hard_plus_contamination_containment` — contamination spread and containment under concurrent leaks.

All rewards and grader scores are bounded to `[0.0, 1.0]`.

## Reward and grader design

Step reward combines:
- leak resolution progress
- SLA and critical-zone handling
- outage control and critical-facility uptime
- contamination control
- fairness and budget discipline

Penalties:
- invalid actions
- idle/no-progress loops
- budget overrun

`grader.py` computes deterministic task scores using aligned metrics:
- completion, SLA compliance, priority coverage
- outage control, critical uptime
- contamination control / containment effectiveness
- fairness, budget discipline, validity

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

## Baseline inference (`inference.py`)

### Mandatory env-var contract

```python
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional
```

- Defaults only for `API_BASE_URL` and `MODEL_NAME`
- `HF_TOKEN` has no default
- LLM calls use `from openai import OpenAI`

Run:

```bash
python inference.py
```

Logs follow strict required format:
- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=...`

### Baseline scores

Baseline policy uses deterministic task-specific action plans with fallback to lookahead/LLM selection.

| Task | Score |
|---|---:|
| `easy_single_crew` | `0.701` |
| `medium_valve_tradeoff` | `0.761` |
| `hard_burst_fairness_budget` | `0.842` |
| `hard_plus_contamination_containment` | `0.863` |
| **Average** | **`0.792`** |

## Docker

```bash
docker build -t pipepulse-openenv .
docker run -p 7860:7860 pipepulse-openenv
```

## Validation

```bash
python prevalidate.py
```

Checks include:
- required files and manifest task count
- reward/grader bounds
- reset/step/state behavior
- valve and flush action paths
- inference env-var and log-helper contract snippets
