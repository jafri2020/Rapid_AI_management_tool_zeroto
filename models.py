from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── LLM output coercion ─────────────────────────────────────────────────────
# LLMs (Kimi/Qwen/DeepSeek especially) often return dicts or lists where the
# schema asks for a string field. We coerce them into readable prose rather
# than failing validation.

def _flatten_to_string(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                parts.append("; ".join(f"{k}: {_flatten_to_string(val)}" for k, val in item.items()))
            else:
                parts.append(_flatten_to_string(item))
        return " | ".join(p for p in parts if p)
    if isinstance(v, dict):
        return " | ".join(f"{k}: {_flatten_to_string(val)}" for k, val in v.items())
    return str(v)


def _coerce_to_list_of_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [_flatten_to_string(item) for item in v if item not in (None, "")]
    if isinstance(v, dict):
        # Treat dict values as the list items
        return [_flatten_to_string(val) for val in v.values()]
    # Single value → wrap in list
    s = _flatten_to_string(v)
    return [s] if s else []


class LLMOutput(BaseModel):
    """Base for any model that is populated from LLM JSON output.

    Coerces structured values (dicts, lists) into the expected primitive type
    when the LLM gets too creative with the schema.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            v = data[field_name]
            ann = field_info.annotation

            # Plain str field
            if ann is str and not isinstance(v, str):
                data[field_name] = _flatten_to_string(v)
                continue

            # List[str] field
            origin = get_origin(ann)
            if origin in (list, List):
                args = get_args(ann)
                if args and args[0] is str and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                    data[field_name] = _coerce_to_list_of_str(v)
                    continue

            # Numeric fields receiving strings (e.g. "9.0" instead of 9.0)
            if ann is float and isinstance(v, str):
                try:
                    data[field_name] = float(v.strip().replace(",", "").rstrip("%"))
                except (ValueError, AttributeError):
                    pass
            elif ann is int and isinstance(v, str):
                try:
                    data[field_name] = int(float(v.strip().replace(",", "").rstrip("%")))
                except (ValueError, AttributeError):
                    pass

            # Bool field receiving string ("true"/"false"/"yes"/"no")
            if ann is bool and isinstance(v, str):
                low = v.strip().lower()
                if low in ("true", "yes", "1", "pass", "passes"):
                    data[field_name] = True
                elif low in ("false", "no", "0", "fail", "fails"):
                    data[field_name] = False

            # Literal[str, ...] field — normalise to lowercase so 'Hybrid' matches 'hybrid'
            if get_origin(ann) is Literal and isinstance(v, str):
                allowed = get_args(ann)
                if all(isinstance(a, str) for a in allowed):
                    low = v.strip().lower()
                    if low in allowed:
                        data[field_name] = low
                    else:
                        # Models often emit prose for constrained fields
                        # (e.g. "Vitamin disguised as a painkiller..." for vitamin|painkiller|both).
                        # Recover the intended option as the allowed value appearing earliest
                        # in the text. If none appears, leave v untouched so validation fails loudly.
                        positions = [(low.find(a), a) for a in allowed if a in low]
                        if positions:
                            data[field_name] = min(positions)[1]
        return data


class IdeaTrack(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    BOTH = "both"


class IdeaInput(BaseModel):
    title: str
    description: str
    submitter: str = "unknown"
    source_context: str = ""
    track: IdeaTrack = IdeaTrack.BOTH
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Z2O Triage ────────────────────────────────────────────────────────────────

class TriageDimension(BaseModel):
    score: int = Field(ge=0, le=4)
    rationale: str


class TriageOutput(LLMOutput):
    secret: TriageDimension
    ten_x: TriageDimension
    timing: TriageDimension
    monopoly: TriageDimension
    lazybones_fit: TriageDimension
    total_score: int = Field(ge=0, le=20)
    triage_recommendation: Literal["fast_track", "proceed", "park", "reject"]
    triage_summary: str

    @model_validator(mode="after")
    def recompute_total(self) -> TriageOutput:
        computed = (
            self.secret.score
            + self.ten_x.score
            + self.timing.score
            + self.monopoly.score
            + self.lazybones_fit.score
        )
        self.total_score = computed
        if computed >= 14:
            self.triage_recommendation = "fast_track"
        elif computed >= 8:
            self.triage_recommendation = "proceed"
        elif computed >= 5:
            self.triage_recommendation = "park"
        else:
            self.triage_recommendation = "reject"
        return self


# ── DE Research Agents ────────────────────────────────────────────────────────

class CustomerOutput(LLMOutput):
    beachhead_definition: str
    end_user_persona: str
    buyer_persona: str
    workflow_before: str
    trigger_event: str
    lifecycle_use_case: str
    beachhead_size_estimate: str
    internal_deployment_scope: str
    customer_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)


class ValuePropOutput(LLMOutput):
    pain_score: int = Field(ge=1, le=10)
    pain_evidence: str
    current_workarounds: str
    quantified_value_statement: str
    before_after_statement: str
    top_competitor: str
    differentiation: str
    vitamin_or_painkiller: Literal["vitamin", "painkiller", "both"]
    value_prop_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)


class AcquisitionOutput(LLMOutput):
    dmu_map: str
    adoption_process: str
    internal_approval_path: str
    channel_recommendation: str
    coca_estimate: str
    comparable_tools_playbook: str
    acquisition_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)


class EconomicsOutput(LLMOutput):
    business_model: str
    pricing_estimate: str
    ltv_estimate: str
    coca_estimate: str
    ltv_coca_ratio: float
    economics_gate_pass: bool

    @field_validator("ltv_coca_ratio", mode="before")
    @classmethod
    def coerce_ratio(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for key in ("external", "internal"):
                val = v.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
            raise ValueError(f"ltv_coca_ratio dict has no numeric value: {v}")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot coerce ltv_coca_ratio to float: {v!r}")
    roi_months_internal: str = ""
    manual_process_cost_estimate: str = ""
    economics_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_gate(self) -> EconomicsOutput:
        from config import config
        self.economics_gate_pass = self.ltv_coca_ratio >= config.ltv_coca_min_ratio
        return self


class FeasibilityOutput(LLMOutput):
    mvp_scope: str
    data_requirements: str
    data_readiness_score: int = Field(ge=0, le=100)
    integration_requirements: str
    build_vs_buy_recommendation: Literal["build", "buy", "integrate", "hybrid"]
    build_vs_buy_rationale: str
    mvp_timeline_estimate: str
    mvp_timeline_days: int
    top_technical_risks: List[str]
    feasibility_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)


class ScaleOutput(LLMOutput):
    moat_type: str
    moat_strength_score: int = Field(ge=0, le=100)
    durability_assessment: str
    reuse_potential_internal: str
    next_beachheads_external: List[str]
    external_product_potential: Literal["high", "medium", "low", "none"]
    external_product_rationale: str
    strategic_leverage_description: str
    scale_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    confidence_rationale: str
    weakest_aspect: str
    search_queries_used: List[str] = Field(default_factory=list)


# ── Composite Scoring ─────────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    name: str
    score: int
    z2o_component: int
    de_component: int
    weight: float
    weighted_contribution: float


class CompositeScore(BaseModel):
    # Six dimension scores (0-100 each)
    problem_severity_score: int
    feasibility_dim_score: int
    moat_score: int
    customer_clarity_score: int
    adoption_score: int
    strategic_leverage_score: int

    # Final composite scores
    composite_score_a: int   # internal Xavor adoption
    composite_score_b: int   # external product potential

    # Hard gates
    economics_gate_pass: bool
    data_readiness_gate_pass: bool
    triage_gate_pass: bool

    # Decisions
    decision_a: Literal["build_now", "prototype", "park", "reject"]
    decision_b: Literal["build_now", "prototype", "park", "reject"]

    # Special flags
    quick_win_flag: bool
    strategic_bet_flag: bool

    # Qualitative
    weakest_dimension: str
    strongest_dimension: str
    recommended_next_action: str

    dimension_breakdown: List[DimensionScore]


# ── Full Validation Result ────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    id: Optional[str] = None
    idea: IdeaInput
    triage: Optional[TriageOutput] = None
    customer: Optional[CustomerOutput] = None
    value_prop: Optional[ValuePropOutput] = None
    acquisition: Optional[AcquisitionOutput] = None
    economics: Optional[EconomicsOutput] = None
    feasibility: Optional[FeasibilityOutput] = None
    scale: Optional[ScaleOutput] = None
    scores: Optional[CompositeScore] = None
    stage_reached: str = "capture"
    agent_errors: dict = Field(default_factory=dict)
    validated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
