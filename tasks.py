from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


CRITICAL_ZONES = {"hospital", "school"}


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    max_ticks: int
    initial_budget: float
    crew_count: int
    description: str
    water_loss_norm: float
    source_segments: List[str]
    segments: List[Dict]
    valves: List[Dict]
    leaks: List[Dict]
    burst_events: Dict[int, List[Dict]] = field(default_factory=dict)
    contamination_events: Dict[int, List[Dict]] = field(default_factory=dict)


TASKS: Dict[str, TaskConfig] = {
    "easy_single_crew": TaskConfig(
        task_id="easy_single_crew",
        difficulty="easy",
        max_ticks=16,
        initial_budget=2200.0,
        crew_count=1,
        description="Single-crew leak response with basic SLA pressure and minimal network controls.",
        water_loss_norm=3000.0,
        source_segments=["E0"],
        segments=[
            {"id": "E0", "ward": "ward_hub", "demand_units": 0.0, "critical_facility": False, "location_index": 0},
            {"id": "E1", "ward": "ward_a", "demand_units": 44.0, "critical_facility": False, "location_index": 2},
            {"id": "E2", "ward": "ward_b", "demand_units": 33.0, "critical_facility": True, "location_index": 6},
            {"id": "E3", "ward": "ward_c", "demand_units": 41.0, "critical_facility": True, "location_index": 10},
        ],
        valves=[
            {"id": "EV1", "from_segment": "E0", "to_segment": "E1", "initially_closed": False},
            {"id": "EV2", "from_segment": "E1", "to_segment": "E2", "initially_closed": False},
            {"id": "EV3", "from_segment": "E1", "to_segment": "E3", "initially_closed": False},
        ],
        leaks=[
            {"id": "EL1", "segment_id": "E1", "ward": "ward_a", "zone_type": "residential", "severity": 2, "flow_rate": 22.0, "sla_ticks": 7, "repair_difficulty": 2, "location_index": 2},
            {"id": "EL2", "segment_id": "E2", "ward": "ward_b", "zone_type": "school", "severity": 3, "flow_rate": 30.0, "sla_ticks": 5, "repair_difficulty": 2, "location_index": 6},
            {"id": "EL3", "segment_id": "E3", "ward": "ward_c", "zone_type": "hospital", "severity": 4, "flow_rate": 35.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 10},
        ],
    ),
    "medium_valve_tradeoff": TaskConfig(
        task_id="medium_valve_tradeoff",
        difficulty="medium",
        max_ticks=22,
        initial_budget=4600.0,
        crew_count=2,
        description="Dispatch with valve isolation decisions balancing leak control vs service outage continuity.",
        water_loss_norm=5200.0,
        source_segments=["M0"],
        segments=[
            {"id": "M0", "ward": "ward_hub", "demand_units": 0.0, "critical_facility": False, "location_index": 0},
            {"id": "M1", "ward": "ward_a", "demand_units": 38.0, "critical_facility": False, "location_index": 2},
            {"id": "M2", "ward": "ward_b", "demand_units": 36.0, "critical_facility": True, "location_index": 5},
            {"id": "M3", "ward": "ward_c", "demand_units": 31.0, "critical_facility": False, "location_index": 8},
            {"id": "M4", "ward": "ward_d", "demand_units": 35.0, "critical_facility": True, "location_index": 11},
            {"id": "M5", "ward": "ward_e", "demand_units": 28.0, "critical_facility": False, "location_index": 14},
        ],
        valves=[
            {"id": "MV1", "from_segment": "M0", "to_segment": "M1", "initially_closed": False},
            {"id": "MV2", "from_segment": "M1", "to_segment": "M2", "initially_closed": False},
            {"id": "MV3", "from_segment": "M2", "to_segment": "M3", "initially_closed": False},
            {"id": "MV4", "from_segment": "M3", "to_segment": "M4", "initially_closed": False},
            {"id": "MV5", "from_segment": "M3", "to_segment": "M5", "initially_closed": False},
            {"id": "MV6", "from_segment": "M1", "to_segment": "M4", "initially_closed": False},
        ],
        leaks=[
            {"id": "ML1", "segment_id": "M1", "ward": "ward_a", "zone_type": "residential", "severity": 2, "flow_rate": 24.0, "sla_ticks": 7, "repair_difficulty": 2, "location_index": 2},
            {"id": "ML2", "segment_id": "M2", "ward": "ward_b", "zone_type": "hospital", "severity": 4, "flow_rate": 36.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 5},
            {"id": "ML3", "segment_id": "M3", "ward": "ward_c", "zone_type": "industrial", "severity": 3, "flow_rate": 32.0, "sla_ticks": 6, "repair_difficulty": 2, "location_index": 8},
            {"id": "ML4", "segment_id": "M4", "ward": "ward_d", "zone_type": "school", "severity": 3, "flow_rate": 31.0, "sla_ticks": 5, "repair_difficulty": 2, "location_index": 11},
            {"id": "ML5", "segment_id": "M5", "ward": "ward_e", "zone_type": "residential", "severity": 2, "flow_rate": 26.0, "sla_ticks": 8, "repair_difficulty": 2, "location_index": 14},
        ],
        burst_events={
            7: [
                {"id": "MLB1", "segment_id": "M3", "ward": "ward_c", "zone_type": "industrial", "severity": 4, "flow_rate": 38.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 8}
            ]
        },
    ),
    "hard_burst_fairness_budget": TaskConfig(
        task_id="hard_burst_fairness_budget",
        difficulty="hard",
        max_ticks=28,
        initial_budget=6400.0,
        crew_count=3,
        description="Multi-crew coordination with burst propagation, valve tradeoffs, budget pressure, and ward fairness.",
        water_loss_norm=7200.0,
        source_segments=["H0"],
        segments=[
            {"id": "H0", "ward": "ward_hub", "demand_units": 0.0, "critical_facility": False, "location_index": 0},
            {"id": "H1", "ward": "ward_a", "demand_units": 34.0, "critical_facility": True, "location_index": 2},
            {"id": "H2", "ward": "ward_b", "demand_units": 37.0, "critical_facility": False, "location_index": 5},
            {"id": "H3", "ward": "ward_c", "demand_units": 32.0, "critical_facility": False, "location_index": 8},
            {"id": "H4", "ward": "ward_d", "demand_units": 39.0, "critical_facility": True, "location_index": 11},
            {"id": "H5", "ward": "ward_e", "demand_units": 28.0, "critical_facility": False, "location_index": 14},
            {"id": "H6", "ward": "ward_f", "demand_units": 31.0, "critical_facility": False, "location_index": 16},
            {"id": "H7", "ward": "ward_g", "demand_units": 27.0, "critical_facility": False, "location_index": 18},
        ],
        valves=[
            {"id": "HV1", "from_segment": "H0", "to_segment": "H1", "initially_closed": False},
            {"id": "HV2", "from_segment": "H1", "to_segment": "H2", "initially_closed": False},
            {"id": "HV3", "from_segment": "H2", "to_segment": "H3", "initially_closed": False},
            {"id": "HV4", "from_segment": "H3", "to_segment": "H4", "initially_closed": False},
            {"id": "HV5", "from_segment": "H4", "to_segment": "H5", "initially_closed": False},
            {"id": "HV6", "from_segment": "H5", "to_segment": "H6", "initially_closed": False},
            {"id": "HV7", "from_segment": "H6", "to_segment": "H7", "initially_closed": False},
            {"id": "HV8", "from_segment": "H2", "to_segment": "H5", "initially_closed": False},
            {"id": "HV9", "from_segment": "H1", "to_segment": "H4", "initially_closed": False},
        ],
        leaks=[
            {"id": "HL1", "segment_id": "H1", "ward": "ward_a", "zone_type": "hospital", "severity": 4, "flow_rate": 40.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 2},
            {"id": "HL2", "segment_id": "H2", "ward": "ward_b", "zone_type": "residential", "severity": 3, "flow_rate": 31.0, "sla_ticks": 6, "repair_difficulty": 2, "location_index": 5},
            {"id": "HL3", "segment_id": "H3", "ward": "ward_c", "zone_type": "industrial", "severity": 3, "flow_rate": 34.0, "sla_ticks": 6, "repair_difficulty": 2, "location_index": 8},
            {"id": "HL4", "segment_id": "H4", "ward": "ward_d", "zone_type": "school", "severity": 4, "flow_rate": 38.0, "sla_ticks": 5, "repair_difficulty": 3, "location_index": 11},
            {"id": "HL5", "segment_id": "H5", "ward": "ward_e", "zone_type": "residential", "severity": 2, "flow_rate": 24.0, "sla_ticks": 8, "repair_difficulty": 2, "location_index": 14},
            {"id": "HL6", "segment_id": "H7", "ward": "ward_g", "zone_type": "industrial", "severity": 2, "flow_rate": 26.0, "sla_ticks": 9, "repair_difficulty": 2, "location_index": 18},
        ],
        burst_events={
            5: [
                {"id": "HLB1", "segment_id": "H3", "ward": "ward_c", "zone_type": "industrial", "severity": 5, "flow_rate": 43.0, "sla_ticks": 3, "repair_difficulty": 4, "location_index": 8}
            ],
            11: [
                {"id": "HLB2", "segment_id": "H6", "ward": "ward_f", "zone_type": "residential", "severity": 4, "flow_rate": 36.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 16}
            ],
        },
    ),
    "hard_plus_contamination_containment": TaskConfig(
        task_id="hard_plus_contamination_containment",
        difficulty="hard_plus",
        max_ticks=32,
        initial_budget=7200.0,
        crew_count=3,
        description="Contain contamination spread while handling leaks, outages, and fairness under strict operational constraints.",
        water_loss_norm=7800.0,
        source_segments=["C0"],
        segments=[
            {"id": "C0", "ward": "ward_hub", "demand_units": 0.0, "critical_facility": False, "location_index": 0},
            {"id": "C1", "ward": "ward_a", "demand_units": 35.0, "critical_facility": True, "location_index": 2},
            {"id": "C2", "ward": "ward_b", "demand_units": 33.0, "critical_facility": False, "location_index": 4},
            {"id": "C3", "ward": "ward_c", "demand_units": 30.0, "critical_facility": False, "location_index": 7},
            {"id": "C4", "ward": "ward_d", "demand_units": 40.0, "critical_facility": True, "location_index": 10},
            {"id": "C5", "ward": "ward_e", "demand_units": 29.0, "critical_facility": False, "location_index": 13},
            {"id": "C6", "ward": "ward_f", "demand_units": 32.0, "critical_facility": False, "location_index": 16},
            {"id": "C7", "ward": "ward_g", "demand_units": 27.0, "critical_facility": False, "location_index": 19},
        ],
        valves=[
            {"id": "CV1", "from_segment": "C0", "to_segment": "C1", "initially_closed": False},
            {"id": "CV2", "from_segment": "C1", "to_segment": "C2", "initially_closed": False},
            {"id": "CV3", "from_segment": "C2", "to_segment": "C3", "initially_closed": False},
            {"id": "CV4", "from_segment": "C3", "to_segment": "C4", "initially_closed": False},
            {"id": "CV5", "from_segment": "C4", "to_segment": "C5", "initially_closed": False},
            {"id": "CV6", "from_segment": "C5", "to_segment": "C6", "initially_closed": False},
            {"id": "CV7", "from_segment": "C6", "to_segment": "C7", "initially_closed": False},
            {"id": "CV8", "from_segment": "C2", "to_segment": "C5", "initially_closed": False},
            {"id": "CV9", "from_segment": "C1", "to_segment": "C4", "initially_closed": False},
            {"id": "CV10", "from_segment": "C3", "to_segment": "C6", "initially_closed": False},
        ],
        leaks=[
            {"id": "CL1", "segment_id": "C1", "ward": "ward_a", "zone_type": "hospital", "severity": 4, "flow_rate": 39.0, "sla_ticks": 4, "repair_difficulty": 3, "location_index": 2},
            {"id": "CL2", "segment_id": "C2", "ward": "ward_b", "zone_type": "residential", "severity": 3, "flow_rate": 30.0, "sla_ticks": 6, "repair_difficulty": 2, "location_index": 4},
            {"id": "CL3", "segment_id": "C4", "ward": "ward_d", "zone_type": "school", "severity": 4, "flow_rate": 37.0, "sla_ticks": 5, "repair_difficulty": 3, "location_index": 10},
            {"id": "CL4", "segment_id": "C5", "ward": "ward_e", "zone_type": "industrial", "severity": 3, "flow_rate": 33.0, "sla_ticks": 6, "repair_difficulty": 2, "location_index": 13},
            {"id": "CL5", "segment_id": "C6", "ward": "ward_f", "zone_type": "residential", "severity": 2, "flow_rate": 25.0, "sla_ticks": 8, "repair_difficulty": 2, "location_index": 16},
        ],
        burst_events={
            10: [
                {"id": "CLB1", "segment_id": "C3", "ward": "ward_c", "zone_type": "industrial", "severity": 5, "flow_rate": 42.0, "sla_ticks": 3, "repair_difficulty": 4, "location_index": 7}
            ]
        },
        contamination_events={
            3: [{"segment_id": "C5", "level": 0.78}],
            8: [{"segment_id": "C3", "level": 0.62}],
        },
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return TASKS[task_id]


def list_task_metadata() -> List[Dict]:
    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "max_steps": task.max_ticks,
            "description": task.description,
            "reward_range": [0.0, 1.0],
        }
        for task in TASKS.values()
    ]
