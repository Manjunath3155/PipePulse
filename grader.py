from __future__ import annotations

from typing import Dict

from models import GradeResult


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def grade_episode(task_id: str, state: Dict) -> GradeResult:
    total_leaks = max(1, int(state.get("total_leaks_generated", 0)))
    resolved = int(state.get("resolved_leaks", 0))
    sla_breaches = int(state.get("sla_breaches", 0))
    priority_total = max(1, int(state.get("priority_leaks_total", 0)))
    priority_resolved = int(state.get("priority_resolved", 0))
    total_water_loss = float(state.get("total_water_loss", 0.0))
    fairness_score = float(state.get("fairness_score", 1.0))
    invalid_actions = int(state.get("invalid_actions", 0))
    max_ticks = max(1, int(state.get("max_ticks", 1)))
    initial_budget = max(1.0, float(state.get("initial_budget", 1.0)))
    budget_remaining = float(state.get("budget_remaining", 0.0))

    total_demand_units = max(1.0, float(state.get("total_demand_units", 1.0)))
    service_disruption_total = float(state.get("service_disruption_total", 0.0))
    critical_outage_ticks = int(state.get("critical_outage_ticks", 0))
    critical_segments_total = max(1, int(state.get("critical_segments_total", 1)))
    contamination_risk_index = float(state.get("contamination_risk_index", 0.0))
    contamination_events_triggered = max(1, int(state.get("contamination_events_triggered", 0)))
    contamination_resolved_segments = int(state.get("contamination_resolved_segments", 0))

    completion = _clip01(resolved / total_leaks)
    sla_compliance = _clip01(1.0 - (sla_breaches / total_leaks))
    priority_coverage = _clip01(priority_resolved / priority_total)
    loss_efficiency = _clip01(1.0 - (total_water_loss / (total_leaks * 850.0)))
    budget_discipline = _clip01(1.0 - (max(0.0, -budget_remaining) / initial_budget))
    validity = _clip01(1.0 - (invalid_actions / max_ticks))

    outage_control = _clip01(1.0 - (service_disruption_total / (max_ticks * total_demand_units)))
    critical_uptime = _clip01(1.0 - (critical_outage_ticks / (max_ticks * critical_segments_total)))
    contamination_control = _clip01(1.0 - contamination_risk_index)
    containment_effectiveness = _clip01(contamination_resolved_segments / contamination_events_triggered)

    if task_id == "easy_single_crew":
        base_score = (
            0.38 * completion
            + 0.28 * sla_compliance
            + 0.22 * loss_efficiency
            + 0.12 * budget_discipline
        )
    elif task_id == "medium_valve_tradeoff":
        base_score = (
            0.28 * completion
            + 0.22 * sla_compliance
            + 0.15 * priority_coverage
            + 0.14 * outage_control
            + 0.13 * critical_uptime
            + 0.08 * loss_efficiency
        )
    elif task_id == "hard_plus_contamination_containment":
        base_score = (
            0.18 * completion
            + 0.14 * sla_compliance
            + 0.12 * priority_coverage
            + 0.12 * fairness_score
            + 0.08 * budget_discipline
            + 0.12 * outage_control
            + 0.10 * critical_uptime
            + 0.10 * contamination_control
            + 0.04 * containment_effectiveness
        )
    else:
        base_score = (
            0.22 * completion
            + 0.18 * sla_compliance
            + 0.15 * priority_coverage
            + 0.13 * fairness_score
            + 0.10 * budget_discipline
            + 0.12 * outage_control
            + 0.10 * critical_uptime
        )

    score = _clip01(base_score * validity)
    passed = score >= 0.68
    summary = (
        f"score={score:.3f} completion={completion:.3f} "
        f"sla={sla_compliance:.3f} outage={outage_control:.3f} contamination={contamination_control:.3f}"
    )

    return GradeResult(
        task_id=task_id,
        score=score,
        passed=passed,
        summary=summary,
        metrics={
            "completion": completion,
            "sla_compliance": sla_compliance,
            "priority_coverage": priority_coverage,
            "loss_efficiency": loss_efficiency,
            "fairness_score": _clip01(fairness_score),
            "budget_discipline": budget_discipline,
            "outage_control": outage_control,
            "critical_uptime": critical_uptime,
            "contamination_control": contamination_control,
            "containment_effectiveness": containment_effectiveness,
            "validity": validity,
        },
    )
