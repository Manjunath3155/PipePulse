from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ZoneType(str, Enum):
    RESIDENTIAL = "residential"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    INDUSTRIAL = "industrial"


class LeakStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"


class CrewStatus(str, Enum):
    AVAILABLE = "available"
    EN_ROUTE = "en_route"
    IN_REPAIR = "in_repair"


class ActionType(str, Enum):
    ASSIGN_CREW = "assign_crew"
    REROUTE_CREW = "reroute_crew"
    HOLD = "hold"
    OPEN_VALVE = "open_valve"
    CLOSE_VALVE = "close_valve"
    FLUSH_SEGMENT = "flush_segment"


class RepairMode(str, Enum):
    PATCH = "patch"
    FULL_REPAIR = "full_repair"
    ISOLATE_LINE = "isolate_line"


class LeakNode(BaseModel):
    leak_id: str
    ward: str
    segment_id: str
    zone_type: ZoneType
    severity: int = Field(ge=1, le=5)
    flow_rate: float = Field(ge=0.0)
    age: int = Field(ge=0)
    sla_ticks: int = Field(ge=1)
    sla_breached: bool
    repair_difficulty: int = Field(ge=1, le=5)
    status: LeakStatus
    location_index: int = Field(ge=0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    assigned_crew_id: Optional[str] = None
    flow_multiplier: float = Field(ge=0.0, le=1.0)


class CrewState(BaseModel):
    crew_id: str
    status: CrewStatus
    location_index: int = Field(ge=0)
    current_leak_id: Optional[str] = None
    current_task_segment_id: Optional[str] = None
    ticks_to_available: int = Field(ge=0)
    mode: Optional[RepairMode] = None


class SegmentState(BaseModel):
    segment_id: str
    ward: str
    demand_units: float = Field(ge=0.0)
    critical_facility: bool = False
    isolated: bool = False
    contamination_level: float = Field(ge=0.0, le=1.0)
    outage_units: float = Field(ge=0.0)


class ValveState(BaseModel):
    valve_id: str
    from_segment: str
    to_segment: str
    is_closed: bool


class Observation(BaseModel):
    task_id: str
    tick: int = Field(ge=0)
    max_ticks: int = Field(ge=1)
    budget_remaining: float
    total_water_loss: float = Field(ge=0.0)
    sla_breaches: int = Field(ge=0)
    active_leaks: List[LeakNode]
    crews: List[CrewState]
    segments: List[SegmentState]
    valves: List[ValveState]
    next_critical_deadline: Optional[int] = None
    ward_fairness_gap: float = Field(ge=0.0, le=1.0)
    pressure_risk_index: float = Field(ge=0.0, le=1.0)
    service_outage_units: float = Field(ge=0.0)
    critical_facility_outages: int = Field(ge=0)
    contamination_risk_index: float = Field(ge=0.0, le=1.0)
    contaminated_segments: int = Field(ge=0)
    done: bool = False


class Action(BaseModel):
    action_type: ActionType
    crew_id: Optional[str] = None
    leak_id: Optional[str] = None
    mode: Optional[RepairMode] = None
    valve_id: Optional[str] = None
    segment_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(self) -> "Action":
        if self.action_type in {ActionType.ASSIGN_CREW, ActionType.REROUTE_CREW}:
            if not self.crew_id:
                raise ValueError("crew_id is required for assign_crew/reroute_crew")
            if not self.leak_id:
                raise ValueError("leak_id is required for assign_crew/reroute_crew")
        if self.action_type == ActionType.ASSIGN_CREW and self.mode is None:
            raise ValueError("mode is required for assign_crew")
        if self.action_type == ActionType.HOLD and not self.crew_id:
            raise ValueError("crew_id is required for hold")
        if self.action_type in {ActionType.OPEN_VALVE, ActionType.CLOSE_VALVE} and not self.valve_id:
            raise ValueError("valve_id is required for open_valve/close_valve")
        if self.action_type == ActionType.FLUSH_SEGMENT:
            if not self.segment_id:
                raise ValueError("segment_id is required for flush_segment")
            if not self.crew_id:
                raise ValueError("crew_id is required for flush_segment")
        return self


class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float]
    message: str


class ResetRequest(BaseModel):
    task_id: str = "easy_single_crew"


class ResetResponse(BaseModel):
    observation: Observation
    done: bool
    info: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GradeResult(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, float]
    passed: bool
    summary: str


class StateResponse(BaseModel):
    task_id: str
    tick: int
    max_ticks: int
    done: bool
    budget_remaining: float
    initial_budget: float
    total_water_loss: float
    sla_breaches: int
    total_leaks_generated: int
    resolved_leaks: int
    priority_leaks_total: int
    priority_resolved: int
    invalid_actions: int
    fairness_score: float
    ward_service_ratio: Dict[str, float]
    active_leak_ids: List[str]
    service_disruption_total: float
    critical_outage_ticks: int
    critical_segments_total: int
    total_demand_units: float
    contamination_events_triggered: int
    contamination_resolved_segments: int
    contamination_risk_index: float
    valve_toggles: int
    grade: Optional[GradeResult] = None


class TaskMeta(BaseModel):
    task_id: str
    difficulty: str
    max_steps: int
    description: str
    reward_range: List[float] = Field(default_factory=lambda: [0.0, 1.0])
