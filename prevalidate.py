from __future__ import annotations

from pathlib import Path

import yaml

from environment import PipePulseEnv
from grader import grade_episode
from models import Action
from tasks import list_task_metadata


ROOT = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_required_files() -> None:
    required = [
        "app.py",
        "environment.py",
        "models.py",
        "grader.py",
        "openenv.yaml",
        "inference.py",
        "Dockerfile",
        "README.md",
    ]
    for file_name in required:
        _assert((ROOT / file_name).exists(), f"Missing required file: {file_name}")


def check_manifest() -> None:
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    _assert(len(tasks) >= 4, "openenv.yaml should contain 4 tasks for V2 scope")
    for task in tasks:
        reward_range = task.get("reward_range", [0.0, 1.0])
        _assert(len(reward_range) == 2, f"Invalid reward_range for task {task.get('id')}")
        _assert(reward_range[0] >= 0.0 and reward_range[1] <= 1.0, "Reward range must be within [0.0, 1.0]")


def check_environment_and_graders() -> None:
    env = PipePulseEnv()
    for task in list_task_metadata():
        task_id = task["task_id"]
        obs = env.reset(task_id)
        _assert(obs.task_id == task_id, f"Reset failed for task {task_id}")
        _assert(len(obs.valves) > 0, f"Task {task_id} should expose valves")
        _assert(len(obs.segments) > 0, f"Task {task_id} should expose segments")

        # Core action path
        hold_action = Action(action_type="hold", crew_id=obs.crews[0].crew_id)
        obs2, reward, _, _ = env.step(hold_action)
        _assert(0.0 <= reward.total <= 1.0, "Step reward must be in [0.0, 1.0]")
        _assert(obs2.task_id == task_id, "Observation task_id mismatch after hold step")

        # Valve action path
        valve_action = Action(action_type="close_valve", valve_id=obs.valves[0].valve_id)
        _, valve_reward, _, _ = env.step(valve_action)
        _assert(0.0 <= valve_reward.total <= 1.0, "Valve step reward must be in [0.0, 1.0]")

        # Flush path (for contamination task, this should be valid eventually)
        if task_id == "hard_plus_contamination_containment":
            flush_action = Action(
                action_type="flush_segment",
                crew_id=obs.crews[0].crew_id,
                segment_id=obs.segments[-1].segment_id,
            )
            _, flush_reward, _, _ = env.step(flush_action)
            _assert(0.0 <= flush_reward.total <= 1.0, "Flush step reward must be in [0.0, 1.0]")

        grade = grade_episode(task_id, env.state().model_dump())
        _assert(0.0 <= grade.score <= 1.0, "Grader score must be in [0.0, 1.0]")


def check_inference_contract() -> None:
    inference_text = (ROOT / "inference.py").read_text(encoding="utf-8")
    required_snippets = [
        'API_BASE_URL = os.getenv("API_BASE_URL",',
        'MODEL_NAME = os.getenv("MODEL_NAME",',
        'HF_TOKEN = os.getenv("HF_TOKEN")',
        'LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")',
        "from openai import OpenAI",
        "def log_start(",
        "def log_step(",
        "def log_end(",
    ]
    for snippet in required_snippets:
        _assert(snippet in inference_text, f"inference.py missing required snippet: {snippet}")


def main() -> None:
    check_required_files()
    check_manifest()
    check_environment_and_graders()
    check_inference_contract()
    print("prevalidate: PASS")


if __name__ == "__main__":
    main()
