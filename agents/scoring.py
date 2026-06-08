"""Deterministic scoring engine — no LLM calls, pure math.

Converts all 6 DE agent outputs + Z2O triage into two composite scores (A=internal, B=external)
and a full CompositeScore breakdown. The qualitative fields (weakest_dimension, next_action)
are computed from the scores themselves rather than another LLM call, keeping this fast and
auditable.
"""
from __future__ import annotations

from typing import Optional

from config import config
from models import (
    AcquisitionOutput,
    CompositeScore,
    CustomerOutput,
    DimensionScore,
    EconomicsOutput,
    FeasibilityOutput,
    ScaleOutput,
    TriageOutput,
    ValuePropOutput,
)

# ── Dimension weights (must sum to 1.0) ──────────────────────────────────────
WEIGHTS = {
    "Problem Severity & ROI": 0.25,
    "Build Speed & Feasibility": 0.20,
    "AI Moat & Differentiation": 0.15,
    "Customer Clarity": 0.15,
    "Adoption & Distribution": 0.15,
    "Strategic Leverage": 0.10,
}

# ── Z2O triage score → 0-100 helper ──────────────────────────────────────────
def _t(score: int) -> int:
    """Convert 0-4 triage dimension score to 0-100."""
    return score * 25


# ── Decision band helper ──────────────────────────────────────────────────────
def _decision(score: int) -> str:
    if score >= config.build_now_min:
        return "build_now"
    if score >= config.prototype_min:
        return "prototype"
    if score >= config.park_min:
        return "park"
    return "reject"


# ── Next action lookup ────────────────────────────────────────────────────────
_NEXT_ACTIONS = {
    "Problem Severity & ROI": (
        "Run 5 user interviews to validate pain severity and gather quantified cost evidence "
        "before committing engineering time."
    ),
    "Build Speed & Feasibility": (
        "Spend one day on a technical spike: confirm data access, test the key API integration, "
        "and validate the core LLM prompt before scoping the full MVP."
    ),
    "AI Moat & Differentiation": (
        "Identify the proprietary data or workflow knowledge that competitors can't replicate — "
        "if none exists, reconsider whether to build or buy."
    ),
    "Customer Clarity": (
        "Sharpen the beachhead: name five specific people at specific companies who have this pain "
        "today and would adopt within 30 days."
    ),
    "Adoption & Distribution": (
        "Map the full approval chain and identify the internal champion who will drive adoption — "
        "without a champion, even great tools stall."
    ),
    "Strategic Leverage": (
        "Define the reuse playbook: what other Xavor workflows or external beachheads does this "
        "unlock after the first win?"
    ),
}


def compute(
    triage: TriageOutput,
    customer: CustomerOutput,
    value_prop: ValuePropOutput,
    acquisition: AcquisitionOutput,
    economics: EconomicsOutput,
    feasibility: FeasibilityOutput,
    scale: ScaleOutput,
    idea_track: str = "both",
) -> CompositeScore:

    # ── Dimension 1: Problem Severity & ROI (25%) ────────────────────────────
    # Z2O: 10x test (60%) + secret test (40%) → how non-obvious and impactful is the problem
    # DE:  value prop score (50%) + economics score (50%)
    z2o_problem = _t(triage.ten_x.score) * 0.6 + _t(triage.secret.score) * 0.4
    de_problem = value_prop.value_prop_score * 0.5 + economics.economics_score * 0.5
    problem_score = int(z2o_problem * 0.35 + de_problem * 0.65)

    # ── Dimension 2: Build Speed & Feasibility (20%) ─────────────────────────
    # Z2O: lazybones fit (25%) — team capability is a function of their specific skills
    # DE:  feasibility score (60%) + data readiness (40%)
    z2o_feasibility = _t(triage.lazybones_fit.score)
    de_feasibility = feasibility.feasibility_score * 0.6 + feasibility.data_readiness_score * 0.4
    # Cap feasibility dimension if data readiness is below floor
    if feasibility.data_readiness_score < config.data_readiness_floor:
        de_feasibility = min(de_feasibility, 40)
    feasibility_dim_score = int(z2o_feasibility * 0.25 + de_feasibility * 0.75)

    # ── Dimension 3: AI Moat & Differentiation (15%) ─────────────────────────
    # Z2O: secret (50%) + timing (50%) — timing creates defensible windows
    # DE:  moat strength score
    z2o_moat = _t(triage.secret.score) * 0.5 + _t(triage.timing.score) * 0.5
    de_moat = scale.moat_strength_score
    moat_score = int(z2o_moat * 0.40 + de_moat * 0.60)

    # ── Dimension 4: Customer Clarity (15%) ──────────────────────────────────
    # Z2O: monopoly test (30%) — specificity of beachhead
    # DE:  customer score (70%)
    z2o_customer = _t(triage.monopoly.score)
    de_customer = customer.customer_score
    customer_clarity_score = int(z2o_customer * 0.30 + de_customer * 0.70)

    # ── Dimension 5: Adoption & Distribution (15%) ───────────────────────────
    # Z2O: timing (25%) — timing signals adoption readiness
    # DE:  acquisition score (75%)
    z2o_adoption = _t(triage.timing.score)
    de_adoption = acquisition.acquisition_score
    adoption_score = int(z2o_adoption * 0.25 + de_adoption * 0.75)

    # ── Dimension 6: Strategic Leverage (10%) ────────────────────────────────
    # Z2O: lazybones fit (50%) + timing (25%) — team fit + moment fit = leverage
    # DE:  scale score (60%) + beachhead reuse bonus (40%)
    z2o_strategic = _t(triage.lazybones_fit.score) * 0.5 + _t(triage.timing.score) * 0.25
    # Bonus for multiple next beachheads (reuse potential)
    beachhead_bonus = min(len(scale.next_beachheads_external) * 5, 15)
    de_strategic = scale.scale_score * 0.6 + beachhead_bonus * 0.4
    # Blend: 40% Z2O signal + 60% DE analysis
    strategic_leverage_score = int(z2o_strategic * 0.40 + de_strategic * 0.60)

    # ── Clamp all dimensions to 0-100 ────────────────────────────────────────
    def clamp(v: int) -> int:
        return max(0, min(100, v))

    problem_score = clamp(problem_score)
    feasibility_dim_score = clamp(feasibility_dim_score)
    moat_score = clamp(moat_score)
    customer_clarity_score = clamp(customer_clarity_score)
    adoption_score = clamp(adoption_score)
    strategic_leverage_score = clamp(strategic_leverage_score)

    # ── Base composite (same formula for A and B before gate adjustments) ────
    base = (
        problem_score * WEIGHTS["Problem Severity & ROI"]
        + feasibility_dim_score * WEIGHTS["Build Speed & Feasibility"]
        + moat_score * WEIGHTS["AI Moat & Differentiation"]
        + customer_clarity_score * WEIGHTS["Customer Clarity"]
        + adoption_score * WEIGHTS["Adoption & Distribution"]
        + strategic_leverage_score * WEIGHTS["Strategic Leverage"]
    )
    base_int = int(base)

    # ── Score A: Internal Xavor ───────────────────────────────────────────────
    # No economics gate on internal — ROI replaces LTV/COCA
    composite_a = base_int
    if idea_track == "external":
        composite_a = 0  # Not applicable

    # ── Score B: External Product ─────────────────────────────────────────────
    composite_b = base_int
    if idea_track == "internal":
        composite_b = 0  # Not applicable
    elif not economics.economics_gate_pass:
        composite_b = min(composite_b, 55)  # Hard cap — weak unit economics
    if scale.external_product_potential == "none":
        composite_b = min(composite_b, 40)

    # ── Hard gate flags ───────────────────────────────────────────────────────
    economics_gate_pass = economics.economics_gate_pass
    data_readiness_gate_pass = feasibility.data_readiness_score >= config.data_readiness_floor
    triage_gate_pass = triage.total_score >= config.triage_gate_min

    # ── Special flags ─────────────────────────────────────────────────────────
    quick_win_flag = (
        max(composite_a, composite_b) >= config.prototype_min
        and feasibility.mvp_timeline_days <= config.quick_win_mvp_days
        and data_readiness_gate_pass
    )
    strategic_bet_flag = (
        max(composite_a, composite_b) >= config.strategic_bet_min_score
        and triage.secret.score == 4
        and triage.timing.score >= config.strategic_bet_timing_min
    )

    # ── Weakest / Strongest dimension ────────────────────────────────────────
    dim_map = {
        "Problem Severity & ROI": problem_score,
        "Build Speed & Feasibility": feasibility_dim_score,
        "AI Moat & Differentiation": moat_score,
        "Customer Clarity": customer_clarity_score,
        "Adoption & Distribution": adoption_score,
        "Strategic Leverage": strategic_leverage_score,
    }
    weakest_dim = min(dim_map, key=dim_map.__getitem__)
    strongest_dim = max(dim_map, key=dim_map.__getitem__)

    recommended_next_action = _NEXT_ACTIONS[weakest_dim]

    # ── Dimension breakdown ───────────────────────────────────────────────────
    breakdown = []
    for name, weight in WEIGHTS.items():
        score = dim_map[name]
        # Approximate Z2O vs DE split for display
        z2o_map = {
            "Problem Severity & ROI": int(_t(triage.ten_x.score) * 0.6 + _t(triage.secret.score) * 0.4),
            "Build Speed & Feasibility": _t(triage.lazybones_fit.score),
            "AI Moat & Differentiation": int(_t(triage.secret.score) * 0.5 + _t(triage.timing.score) * 0.5),
            "Customer Clarity": _t(triage.monopoly.score),
            "Adoption & Distribution": _t(triage.timing.score),
            "Strategic Leverage": int(_t(triage.lazybones_fit.score) * 0.5 + _t(triage.timing.score) * 0.25),
        }
        de_map = {
            "Problem Severity & ROI": int(value_prop.value_prop_score * 0.5 + economics.economics_score * 0.5),
            "Build Speed & Feasibility": int(feasibility.feasibility_score * 0.6 + feasibility.data_readiness_score * 0.4),
            "AI Moat & Differentiation": scale.moat_strength_score,
            "Customer Clarity": customer.customer_score,
            "Adoption & Distribution": acquisition.acquisition_score,
            "Strategic Leverage": scale.scale_score,
        }
        breakdown.append(
            DimensionScore(
                name=name,
                score=score,
                z2o_component=clamp(z2o_map[name]),
                de_component=clamp(de_map[name]),
                weight=weight,
                weighted_contribution=round(score * weight, 1),
            )
        )

    return CompositeScore(
        problem_severity_score=problem_score,
        feasibility_dim_score=feasibility_dim_score,
        moat_score=moat_score,
        customer_clarity_score=customer_clarity_score,
        adoption_score=adoption_score,
        strategic_leverage_score=strategic_leverage_score,
        composite_score_a=composite_a,
        composite_score_b=composite_b,
        economics_gate_pass=economics_gate_pass,
        data_readiness_gate_pass=data_readiness_gate_pass,
        triage_gate_pass=triage_gate_pass,
        decision_a=_decision(composite_a),
        decision_b=_decision(composite_b),
        quick_win_flag=quick_win_flag,
        strategic_bet_flag=strategic_bet_flag,
        weakest_dimension=weakest_dim,
        strongest_dimension=strongest_dim,
        recommended_next_action=recommended_next_action,
        dimension_breakdown=breakdown,
    )
