from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import Body, FastAPI, HTTPException, Query

from environment import PipePulseEnv
from grader import grade_episode
from models import (
    Action,
    GradeResult,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResponse,
    TaskMeta,
)
from tasks import list_task_metadata


app = FastAPI(
    title="PipePulse OpenEnv",
    version="1.0.0",
)

env = PipePulseEnv()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "pipepulse-openenv"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "task": env.current_task_id}


@app.get("/tasks", response_model=List[TaskMeta])
def tasks() -> List[TaskMeta]:
    return [TaskMeta(**task) for task in list_task_metadata()]


@app.post("/reset", response_model=ResetResponse)
def reset(
    request: Optional[ResetRequest] = Body(default=None),
    task_id: Optional[str] = Query(default=None),
) -> ResetResponse:
    selected_task = task_id or (request.task_id if request else "easy_single_crew")
    try:
        observation = env.reset(selected_task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(
        observation=observation,
        done=False,
        info={"task_id": selected_task, "message": "environment_reset"},
    )


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    observation, reward, done, info = env.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return env.state()


@app.get("/grade", response_model=GradeResult)
def grade() -> GradeResult:
    return grade_episode(env.current_task_id, env.state().model_dump())


@app.get("/manifest")
def manifest() -> dict:
    manifest_path = Path(__file__).with_name("openenv.yaml")
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return {"manifest": data}
