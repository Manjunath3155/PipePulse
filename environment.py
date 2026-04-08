from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from grader import grade_episode
from models import (
    Action,
    ActionType,
    CrewState,
    CrewStatus,
    LeakNode,
    LeakStatus,
    Observation,
    RepairMode,
    Reward,
    SegmentState,
    StateResponse,
    ValveState,
    ZoneType,
)
from tasks import CRITICAL_ZONES, TaskConfig, get_task_config


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class PipePulseEnv:
    def __init__(self) -> None:
        self.current_task_id = "easy_single_crew"
        self._task: TaskConfig = get_task_config(self.current_task_id)

        self._segments: Dict[str, Dict[str, Any]] = {}
        self._valves: Dict[str, Dict[str, Any]] = {}
        self._leaks: Dict[str, Dict[str, Any]] = {}
        self._crews: Dict[str, Dict[str, Any]] = {}

        self._burst_schedule: Dict[int, List[Dict[str, Any]]] = {}
        self._contamination_schedule: Dict[int, List[Dict[str, Any]]] = {}

        self._tick = 0
        self._done = False
        self._budget_remaining = 0.0
        self._initial_budget = 0.0
        self._total_water_loss = 0.0
        self._sla_breaches = 0
        self._resolved_leaks = 0
        self._priority_total = 0
        self._priority_resolved = 0
        self._invalid_actions = 0
        self._total_leaks_generated = 0
        self._last_grade = None
        self._history: List[Dict[str, Any]] = []
        self._ward_targets: Dict[str, int] = {}
        self._ward_resolved: Dict[str, int] = {}

        self._service_outage_units_tick = 0.0
        self._critical_outages_tick = 0
        self._service_disruption_total = 0.0
        self._critical_outage_ticks = 0
        self._contamination_events_triggered = 0
        self._contamination_resolved_segments = 0
        self._valve_toggles = 0

        self.reset(self.current_task_id)

    def reset(self, task_id: str) -> Observation:
        self.current_task_id = task_id
        self._task = get_task_config(task_id)

        self._tick = 0
        self._done = False
        self._budget_remaining = self._task.initial_budget
        self._initial_budget = self._task.initial_budget
        self._total_water_loss = 0.0
        self._sla_breaches = 0
        self._resolved_leaks = 0
        self._priority_total = 0
        self._priority_resolved = 0
        self._invalid_actions = 0
        self._total_leaks_generated = 0
        self._last_grade = None
        self._history = []
        self._ward_targets = {}
        self._ward_resolved = {}
        self._segments = {}
        self._valves = {}
        self._leaks = {}
        self._crews = {}

        self._service_outage_units_tick = 0.0
        self._critical_outages_tick = 0
        self._service_disruption_total = 0.0
        self._critical_outage_ticks = 0
        self._contamination_events_triggered = 0
        self._contamination_resolved_segments = 0
        self._valve_toggles = 0

        self._burst_schedule = deepcopy(self._task.burst_events)
        self._contamination_schedule = deepcopy(self._task.contamination_events)

        for segment in self._task.segments:
            self._register_segment(segment)
        for valve in self._task.valves:
            self._register_valve(valve)
        for leak in self._task.leaks:
            self._register_leak(leak)

        source_location = self._segment_location(self._task.source_segments[0])
        for crew_idx in range(self._task.crew_count):
            crew_id = f"crew_{crew_idx + 1}"
            self._crews[crew_id] = {
                "crew_id": crew_id,
                "status": CrewStatus.AVAILABLE.value,
                "location_index": source_location + crew_idx,
                "current_leak_id": None,
                "current_task_segment_id": None,
                "travel_ticks_remaining": 0,
                "repair_ticks_remaining": 0,
                "mode": None,
            }

        self._update_network_state()
        return self._build_observation(done=False)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            obs = self._build_observation(done=True)
            reward = Reward(total=0.0, components={"terminal": 0.0}, message="episode_done")
            info = {"reason": "episode_already_done", "grade": self._last_grade.model_dump() if self._last_grade else None}
            return obs, reward, True, info

        action_outcome = self._apply_action(action)
        transition = self._advance_tick()
        self._done = self._check_done()
        reward = self._compute_reward(action_outcome, transition)

        if self._done:
            self._last_grade = grade_episode(self.current_task_id, self.state().model_dump())

        observation = self._build_observation(done=self._done)
        info = {
            "tick": self._tick,
            "resolved_this_tick": transition["resolved_count"],
            "water_loss_this_tick": round(transition["water_loss"], 4),
            "service_outage_units": round(transition["service_outage_units"], 4),
            "critical_facility_outages": transition["critical_outages"],
            "contamination_risk_index": round(transition["contamination_risk"], 4),
            "invalid_action": action_outcome["invalid"],
            "action_message": action_outcome["message"],
            "grade": self._last_grade.model_dump() if self._last_grade else None,
        }

        self._history.append(
            {
                "tick": self._tick,
                "action": action.model_dump(),
                "reward": reward.total,
                "done": self._done,
                "transition": transition,
                "info": info,
            }
        )
        return observation, reward, self._done, info

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self.current_task_id,
            tick=self._tick,
            max_ticks=self._task.max_ticks,
            done=self._done,
            budget_remaining=round(self._budget_remaining, 4),
            initial_budget=self._initial_budget,
            total_water_loss=round(self._total_water_loss, 4),
            sla_breaches=self._sla_breaches,
            total_leaks_generated=self._total_leaks_generated,
            resolved_leaks=self._resolved_leaks,
            priority_leaks_total=self._priority_total,
            priority_resolved=self._priority_resolved,
            invalid_actions=self._invalid_actions,
            fairness_score=round(self._fairness_score(), 4),
            ward_service_ratio=self._ward_service_ratio(),
            active_leak_ids=sorted(leak_id for leak_id, leak in self._leaks.items() if leak["status"] == LeakStatus.ACTIVE.value),
            service_disruption_total=round(self._service_disruption_total, 4),
            critical_outage_ticks=self._critical_outage_ticks,
            critical_segments_total=self._critical_segments_total(),
            total_demand_units=round(self._total_demand_units(), 4),
            contamination_events_triggered=self._contamination_events_triggered,
            contamination_resolved_segments=self._contamination_resolved_segments,
            contamination_risk_index=round(self._contamination_risk_index(), 4),
            valve_toggles=self._valve_toggles,
            grade=self._last_grade,
        )

    def history(self) -> List[Dict[str, Any]]:
        return deepcopy(self._history)

    def _register_segment(self, segment_fixture: Dict[str, Any]) -> None:
        segment = deepcopy(segment_fixture)
        segment["isolated"] = False
        segment["contamination_level"] = 0.0
        segment["outage_units"] = 0.0
        self._segments[segment["id"]] = segment

    def _register_valve(self, valve_fixture: Dict[str, Any]) -> None:
        valve = deepcopy(valve_fixture)
        valve["is_closed"] = bool(valve.get("initially_closed", False))
        self._valves[valve["id"]] = valve

    def _register_leak(self, leak_fixture: Dict[str, Any]) -> None:
        leak = deepcopy(leak_fixture)
        leak["status"] = LeakStatus.ACTIVE.value
        leak["age"] = 0
        leak["sla_breached"] = False
        leak["assigned_crew_id"] = None
        leak["resolved_tick"] = None
        leak["flow_multiplier"] = 1.0
        leak["zone_type"] = ZoneType(leak["zone_type"]).value

        leak_id = leak["id"]
        self._leaks[leak_id] = leak
        self._total_leaks_generated += 1

        ward = leak["ward"]
        self._ward_targets[ward] = self._ward_targets.get(ward, 0) + 1
        self._ward_resolved.setdefault(ward, 0)

        if leak["zone_type"] in CRITICAL_ZONES:
            self._priority_total += 1

    def _apply_action(self, action: Action) -> Dict[str, Any]:
        outcome = {"invalid": False, "message": "ok", "idle_hold": False, "containment_action": False}

        if action.action_type == ActionType.HOLD:
            crew = self._crews.get(action.crew_id or "")
            if crew is None:
                return self._mark_invalid("unknown_crew_for_hold")
            outcome["idle_hold"] = crew["status"] == CrewStatus.AVAILABLE.value
            outcome["message"] = "hold_applied"
            return outcome

        if action.action_type == ActionType.OPEN_VALVE:
            return self._toggle_valve(action.valve_id or "", close=False)
        if action.action_type == ActionType.CLOSE_VALVE:
            return self._toggle_valve(action.valve_id or "", close=True)
        if action.action_type == ActionType.FLUSH_SEGMENT:
            flush_outcome = self._flush_segment(crew_id=action.crew_id or "", segment_id=action.segment_id or "")
            flush_outcome["containment_action"] = True
            return flush_outcome

        crew = self._crews.get(action.crew_id or "")
        if crew is None:
            return self._mark_invalid("unknown_crew")
        leak = self._leaks.get(action.leak_id or "")
        if leak is None:
            return self._mark_invalid("unknown_leak")
        if leak["status"] != LeakStatus.ACTIVE.value:
            return self._mark_invalid("inactive_leak")

        if action.action_type == ActionType.ASSIGN_CREW:
            if crew["status"] != CrewStatus.AVAILABLE.value:
                return self._mark_invalid("crew_not_available_for_assign")
            if leak["assigned_crew_id"] and leak["assigned_crew_id"] != crew["crew_id"]:
                return self._mark_invalid("leak_already_assigned")
            mode = action.mode or RepairMode.PATCH
            self._attach_crew_to_leak(crew=crew, leak=leak, mode=mode, reroute=False)
            return outcome

        if action.action_type == ActionType.REROUTE_CREW:
            if crew["status"] == CrewStatus.AVAILABLE.value:
                return self._mark_invalid("crew_already_available_use_assign")
            mode = action.mode or RepairMode.PATCH
            old_leak_id = crew["current_leak_id"]
            if old_leak_id and old_leak_id in self._leaks:
                old_leak = self._leaks[old_leak_id]
                if old_leak["status"] == LeakStatus.ACTIVE.value:
                    old_leak["assigned_crew_id"] = None
            self._attach_crew_to_leak(crew=crew, leak=leak, mode=mode, reroute=True)
            outcome["message"] = "reroute_applied"
            return outcome

        return self._mark_invalid("unsupported_action_type")

    def _toggle_valve(self, valve_id: str, close: bool) -> Dict[str, Any]:
        valve = self._valves.get(valve_id)
        if valve is None:
            return self._mark_invalid("unknown_valve")
        if close and valve["is_closed"]:
            return self._mark_invalid("valve_already_closed")
        if (not close) and (not valve["is_closed"]):
            return self._mark_invalid("valve_already_open")

        valve["is_closed"] = close
        self._budget_remaining -= 22.0
        self._valve_toggles += 1
        self._update_network_state()
        return {
            "invalid": False,
            "message": "valve_closed" if close else "valve_opened",
            "idle_hold": False,
            "containment_action": False,
        }

    def _flush_segment(self, crew_id: str, segment_id: str) -> Dict[str, Any]:
        crew = self._crews.get(crew_id)
        if crew is None:
            return self._mark_invalid("unknown_crew_for_flush")
        if crew["status"] != CrewStatus.AVAILABLE.value:
            return self._mark_invalid("crew_not_available_for_flush")
        segment = self._segments.get(segment_id)
        if segment is None:
            return self._mark_invalid("unknown_segment_for_flush")

        before = float(segment["contamination_level"])
        if before <= 0.05:
            return self._mark_invalid("segment_not_contaminated")

        after = max(0.0, before - 0.65)
        segment["contamination_level"] = after
        if before > 0.25 and after <= 0.05:
            self._contamination_resolved_segments += 1

        self._budget_remaining -= 95.0
        crew["status"] = CrewStatus.IN_REPAIR.value
        crew["current_leak_id"] = None
        crew["current_task_segment_id"] = segment_id
        crew["travel_ticks_remaining"] = 0
        crew["repair_ticks_remaining"] = 1
        crew["mode"] = None

        return {
            "invalid": False,
            "message": "segment_flushed",
            "idle_hold": False,
            "containment_action": True,
        }

    def _attach_crew_to_leak(self, crew: Dict[str, Any], leak: Dict[str, Any], mode: RepairMode, reroute: bool) -> None:
        distance = abs(int(crew["location_index"]) - int(leak["location_index"]))
        travel_ticks = 1 + (distance // 4)

        difficulty = int(leak["repair_difficulty"])
        if mode == RepairMode.FULL_REPAIR:
            repair_ticks = difficulty + 1
            mode_cost = 130.0
        elif mode == RepairMode.ISOLATE_LINE:
            repair_ticks = max(1, difficulty - 1)
            mode_cost = 108.0
        else:
            repair_ticks = difficulty
            mode_cost = 88.0

        dispatch_cost = mode_cost + (travel_ticks * 12.0) + (26.0 if reroute else 0.0)
        self._budget_remaining -= dispatch_cost

        crew["status"] = CrewStatus.EN_ROUTE.value
        crew["current_leak_id"] = leak["id"]
        crew["current_task_segment_id"] = leak["segment_id"]
        crew["travel_ticks_remaining"] = travel_ticks
        crew["repair_ticks_remaining"] = repair_ticks
        crew["mode"] = mode.value
        leak["assigned_crew_id"] = crew["crew_id"]

    def _mark_invalid(self, message: str) -> Dict[str, Any]:
        self._invalid_actions += 1
        return {"invalid": True, "message": message, "idle_hold": False, "containment_action": False}

    def _advance_tick(self) -> Dict[str, Any]:
        self._tick += 1

        pre_risk = self._contamination_risk_index()
        self._apply_contamination_events(self._tick)
        self._apply_burst_events(self._tick)
        self._update_network_state()

        resolved_ids: List[str] = []
        water_loss_this_tick = 0.0
        priority_resolved = 0
        sla_saved = 0

        for leak in self._leaks.values():
            if leak["status"] != LeakStatus.ACTIVE.value:
                continue

            leak["age"] += 1
            if leak["age"] % 4 == 0:
                leak["severity"] = min(5, int(leak["severity"]) + 1)

            segment = self._segments.get(leak["segment_id"], {})
            isolated = bool(segment.get("isolated", False))
            flow_multiplier = 0.35 if isolated else 1.0
            leak["flow_multiplier"] = flow_multiplier

            tick_loss = float(leak["flow_rate"]) * (1.0 + (0.22 * float(leak["severity"]))) * flow_multiplier
            water_loss_this_tick += tick_loss
            self._total_water_loss += tick_loss

            if not leak["sla_breached"] and leak["age"] > leak["sla_ticks"]:
                leak["sla_breached"] = True
                self._sla_breaches += 1

            if leak["zone_type"] == ZoneType.INDUSTRIAL.value and leak["severity"] >= 4:
                seg = self._segments.get(leak["segment_id"])
                if seg:
                    seg["contamination_level"] = _clip01(seg["contamination_level"] + 0.06)

        for crew in self._crews.values():
            if crew["status"] == CrewStatus.EN_ROUTE.value:
                crew["travel_ticks_remaining"] -= 1
                if crew["travel_ticks_remaining"] <= 0:
                    crew["status"] = CrewStatus.IN_REPAIR.value
            elif crew["status"] == CrewStatus.IN_REPAIR.value:
                crew["repair_ticks_remaining"] -= 1
                if crew["repair_ticks_remaining"] <= 0:
                    leak_id = crew["current_leak_id"]
                    if leak_id:
                        leak = self._leaks.get(leak_id)
                        if leak and leak["status"] == LeakStatus.ACTIVE.value:
                            leak["status"] = LeakStatus.RESOLVED.value
                            leak["resolved_tick"] = self._tick
                            leak["assigned_crew_id"] = None
                            resolved_ids.append(leak_id)
                            self._resolved_leaks += 1
                            self._ward_resolved[leak["ward"]] = self._ward_resolved.get(leak["ward"], 0) + 1
                            if leak["zone_type"] in CRITICAL_ZONES:
                                self._priority_resolved += 1
                                priority_resolved += 1
                            if not leak["sla_breached"]:
                                sla_saved += 1
                            crew["location_index"] = leak["location_index"]
                    else:
                        segment_id = crew.get("current_task_segment_id")
                        if segment_id and segment_id in self._segments:
                            crew["location_index"] = int(self._segments[segment_id]["location_index"])

                    crew["status"] = CrewStatus.AVAILABLE.value
                    crew["current_leak_id"] = None
                    crew["current_task_segment_id"] = None
                    crew["travel_ticks_remaining"] = 0
                    crew["repair_ticks_remaining"] = 0
                    crew["mode"] = None

        self._spread_contamination()
        self._update_network_state()

        self._service_disruption_total += self._service_outage_units_tick
        self._critical_outage_ticks += self._critical_outages_tick
        post_risk = self._contamination_risk_index()

        return {
            "resolved_ids": resolved_ids,
            "resolved_count": len(resolved_ids),
            "water_loss": water_loss_this_tick,
            "priority_resolved": priority_resolved,
            "sla_saved": sla_saved,
            "service_outage_units": self._service_outage_units_tick,
            "critical_outages": self._critical_outages_tick,
            "contamination_risk": post_risk,
            "containment_improved": post_risk + 0.02 < pre_risk,
        }

    def _apply_burst_events(self, tick: int) -> None:
        for burst in self._burst_schedule.get(tick, []):
            if burst["id"] not in self._leaks:
                self._register_leak(burst)

    def _apply_contamination_events(self, tick: int) -> None:
        for event in self._contamination_schedule.get(tick, []):
            segment = self._segments.get(event["segment_id"])
            if segment is None:
                continue
            segment["contamination_level"] = max(float(segment["contamination_level"]), float(event["level"]))
            self._contamination_events_triggered += 1

    def _spread_contamination(self) -> None:
        additions: Dict[str, float] = {segment_id: 0.0 for segment_id in self._segments}

        for valve in self._valves.values():
            if valve["is_closed"]:
                continue
            a_id = valve["from_segment"]
            b_id = valve["to_segment"]
            a_level = float(self._segments[a_id]["contamination_level"])
            b_level = float(self._segments[b_id]["contamination_level"])

            if a_level > 0.25:
                additions[b_id] = max(additions[b_id], a_level * 0.22)
            if b_level > 0.25:
                additions[a_id] = max(additions[a_id], b_level * 0.22)

        for segment in self._segments.values():
            segment["contamination_level"] = max(0.0, float(segment["contamination_level"]) - 0.03)

        for segment_id, add_level in additions.items():
            if add_level <= 0.0:
                continue
            segment = self._segments[segment_id]
            segment["contamination_level"] = _clip01(float(segment["contamination_level"]) + add_level)

    def _update_network_state(self) -> None:
        open_graph: Dict[str, Set[str]] = {segment_id: set() for segment_id in self._segments}
        for valve in self._valves.values():
            if valve["is_closed"]:
                continue
            a = valve["from_segment"]
            b = valve["to_segment"]
            if a in open_graph and b in open_graph:
                open_graph[a].add(b)
                open_graph[b].add(a)

        reachable: Set[str] = set()
        stack = list(self._task.source_segments)
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(open_graph.get(node, []))

        outage_units = 0.0
        critical_outages = 0
        for segment_id, segment in self._segments.items():
            is_isolated = segment_id not in reachable and float(segment["demand_units"]) > 0.0
            segment["isolated"] = is_isolated
            segment["outage_units"] = float(segment["demand_units"]) if is_isolated else 0.0
            outage_units += segment["outage_units"]
            if is_isolated and bool(segment["critical_facility"]):
                critical_outages += 1

        self._service_outage_units_tick = outage_units
        self._critical_outages_tick = critical_outages

    def _check_done(self) -> bool:
        if self._tick >= self._task.max_ticks:
            return True
        active_leak_exists = any(leak["status"] == LeakStatus.ACTIVE.value for leak in self._leaks.values())
        future_bursts = any(tick > self._tick for tick in self._burst_schedule)
        future_contamination = any(tick > self._tick for tick in self._contamination_schedule)
        contamination_active = any(float(segment["contamination_level"]) > 0.10 for segment in self._segments.values())
        return (not active_leak_exists) and (not future_bursts) and (not future_contamination) and (not contamination_active)

    def _compute_reward(self, action_outcome: Dict[str, Any], transition: Dict[str, Any]) -> Reward:
        resolved_count = transition["resolved_count"]
        progress = resolved_count / max(1, self._total_leaks_generated)
        sla_saves_rate = (transition["sla_saved"] / resolved_count) if resolved_count > 0 else 0.0
        priority_rate = (transition["priority_resolved"] / resolved_count) if resolved_count > 0 else 0.0
        water_efficiency = _clip01(1.0 - (transition["water_loss"] / self._task.water_loss_norm))
        fairness = self._fairness_score()

        outage_control = _clip01(1.0 - (transition["service_outage_units"] / max(1.0, self._total_demand_units())))
        critical_uptime = _clip01(1.0 - (transition["critical_outages"] / max(1, self._critical_segments_total())))
        contamination_control = _clip01(1.0 - transition["contamination_risk"])

        invalid_penalty = 0.32 if action_outcome["invalid"] else 0.0
        idle_penalty = 0.12 if action_outcome["idle_hold"] else 0.0
        no_progress_penalty = (
            0.06
            if (resolved_count == 0 and not transition["containment_improved"] and not action_outcome["containment_action"])
            else 0.0
        )
        budget_penalty = min(0.20, max(0.0, -self._budget_remaining) / max(1.0, self._initial_budget))

        fairness_weight = 0.08 if self._task.difficulty in {"hard", "hard_plus"} else 0.04
        raw_total = (
            0.08
            + (0.24 * progress)
            + (0.10 * sla_saves_rate)
            + (0.10 * priority_rate)
            + (0.10 * water_efficiency)
            + (0.12 * outage_control)
            + (0.10 * critical_uptime)
            + (0.12 * contamination_control)
            + (fairness_weight * fairness)
            - invalid_penalty
            - idle_penalty
            - no_progress_penalty
            - budget_penalty
        )

        if self._done and self._resolved_leaks == self._total_leaks_generated and transition["contamination_risk"] < 0.12:
            raw_total += 0.12

        total = _clip01(raw_total)
        components = {
            "progress": round(progress, 4),
            "sla_saves_rate": round(sla_saves_rate, 4),
            "priority_rate": round(priority_rate, 4),
            "water_efficiency": round(water_efficiency, 4),
            "outage_control": round(outage_control, 4),
            "critical_uptime": round(critical_uptime, 4),
            "contamination_control": round(contamination_control, 4),
            "fairness": round(fairness, 4),
            "invalid_penalty": round(invalid_penalty, 4),
            "idle_penalty": round(idle_penalty, 4),
            "no_progress_penalty": round(no_progress_penalty, 4),
            "budget_penalty": round(budget_penalty, 4),
        }
        return Reward(total=total, components=components, message=action_outcome["message"])

    def _build_observation(self, done: bool) -> Observation:
        active_leaks = [
            self._to_leak_model(leak)
            for leak in self._leaks.values()
            if leak["status"] == LeakStatus.ACTIVE.value
        ]
        active_leaks.sort(key=lambda leak: leak.urgency_score, reverse=True)

        crews = [self._to_crew_model(crew) for crew in self._crews.values()]
        crews.sort(key=lambda crew: crew.crew_id)

        segments = [self._to_segment_model(segment) for segment in self._segments.values()]
        segments.sort(key=lambda segment: segment.segment_id)

        valves = [self._to_valve_model(valve) for valve in self._valves.values()]
        valves.sort(key=lambda valve: valve.valve_id)

        contamination_risk = self._contamination_risk_index()
        contaminated_segments = sum(1 for seg in self._segments.values() if float(seg["contamination_level"]) > 0.2)

        return Observation(
            task_id=self.current_task_id,
            tick=self._tick,
            max_ticks=self._task.max_ticks,
            budget_remaining=round(self._budget_remaining, 4),
            total_water_loss=round(self._total_water_loss, 4),
            sla_breaches=self._sla_breaches,
            active_leaks=active_leaks,
            crews=crews,
            segments=segments,
            valves=valves,
            next_critical_deadline=self._next_critical_deadline(),
            ward_fairness_gap=round(self._ward_fairness_gap(), 4),
            pressure_risk_index=round(self._pressure_risk_index(), 4),
            service_outage_units=round(self._service_outage_units_tick, 4),
            critical_facility_outages=self._critical_outages_tick,
            contamination_risk_index=round(contamination_risk, 4),
            contaminated_segments=contaminated_segments,
            done=done,
        )

    def _to_leak_model(self, leak: Dict[str, Any]) -> LeakNode:
        remaining_sla = max(0, leak["sla_ticks"] - leak["age"])
        urgency = 0.5 * (leak["severity"] / 5.0) + 0.5 * (1.0 - (remaining_sla / max(1, leak["sla_ticks"])))
        if leak["zone_type"] in CRITICAL_ZONES:
            urgency += 0.1
        urgency = _clip01(urgency)

        return LeakNode(
            leak_id=leak["id"],
            ward=leak["ward"],
            segment_id=leak["segment_id"],
            zone_type=ZoneType(leak["zone_type"]),
            severity=leak["severity"],
            flow_rate=leak["flow_rate"],
            age=leak["age"],
            sla_ticks=leak["sla_ticks"],
            sla_breached=leak["sla_breached"],
            repair_difficulty=leak["repair_difficulty"],
            status=LeakStatus(leak["status"]),
            location_index=leak["location_index"],
            urgency_score=urgency,
            assigned_crew_id=leak["assigned_crew_id"],
            flow_multiplier=leak["flow_multiplier"],
        )

    def _to_crew_model(self, crew: Dict[str, Any]) -> CrewState:
        if crew["status"] == CrewStatus.EN_ROUTE.value:
            ticks_to_available = crew["travel_ticks_remaining"] + crew["repair_ticks_remaining"]
        elif crew["status"] == CrewStatus.IN_REPAIR.value:
            ticks_to_available = crew["repair_ticks_remaining"]
        else:
            ticks_to_available = 0

        mode = RepairMode(crew["mode"]) if crew["mode"] else None
        return CrewState(
            crew_id=crew["crew_id"],
            status=CrewStatus(crew["status"]),
            location_index=crew["location_index"],
            current_leak_id=crew["current_leak_id"],
            current_task_segment_id=crew["current_task_segment_id"],
            ticks_to_available=max(0, ticks_to_available),
            mode=mode,
        )

    def _to_segment_model(self, segment: Dict[str, Any]) -> SegmentState:
        return SegmentState(
            segment_id=segment["id"],
            ward=segment["ward"],
            demand_units=segment["demand_units"],
            critical_facility=segment["critical_facility"],
            isolated=segment["isolated"],
            contamination_level=round(float(segment["contamination_level"]), 4),
            outage_units=round(float(segment["outage_units"]), 4),
        )

    def _to_valve_model(self, valve: Dict[str, Any]) -> ValveState:
        return ValveState(
            valve_id=valve["id"],
            from_segment=valve["from_segment"],
            to_segment=valve["to_segment"],
            is_closed=bool(valve["is_closed"]),
        )

    def _segment_location(self, segment_id: str) -> int:
        segment = self._segments.get(segment_id)
        if segment is None:
            return 0
        return int(segment["location_index"])

    def _next_critical_deadline(self) -> Optional[int]:
        deadlines: List[int] = []
        for leak in self._leaks.values():
            if leak["status"] != LeakStatus.ACTIVE.value:
                continue
            if leak["zone_type"] not in CRITICAL_ZONES:
                continue
            remaining = max(0, leak["sla_ticks"] - leak["age"])
            deadlines.append(remaining)
        return min(deadlines) if deadlines else None

    def _pressure_risk_index(self) -> float:
        active = [leak for leak in self._leaks.values() if leak["status"] == LeakStatus.ACTIVE.value]
        if not active:
            return 0.0
        risk = sum(float(leak["flow_rate"]) * float(leak["severity"]) * float(leak["flow_multiplier"]) for leak in active)
        norm = len(active) * 250.0
        return _clip01(risk / norm)

    def _contamination_risk_index(self) -> float:
        total_demand = max(1.0, self._total_demand_units())
        risk = 0.0
        for segment in self._segments.values():
            weight = 1.6 if segment["critical_facility"] else 1.0
            risk += float(segment["contamination_level"]) * float(segment["demand_units"]) * weight
        return _clip01(risk / (total_demand * 1.6))

    def _total_demand_units(self) -> float:
        return sum(float(segment["demand_units"]) for segment in self._segments.values())

    def _critical_segments_total(self) -> int:
        return sum(1 for segment in self._segments.values() if segment["critical_facility"])

    def _ward_service_ratio(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for ward, target in self._ward_targets.items():
            resolved = self._ward_resolved.get(ward, 0)
            ratios[ward] = round(_clip01(resolved / max(1, target)), 4)
        return ratios

    def _ward_fairness_gap(self) -> float:
        ratios = list(self._ward_service_ratio().values())
        if len(ratios) <= 1:
            return 0.0
        return _clip01(max(ratios) - min(ratios))

    def _fairness_score(self) -> float:
        return _clip01(1.0 - self._ward_fairness_gap())
