"""Generates the full-detail validation memo from a ValidationResult."""
from __future__ import annotations

from models import ValidationResult

_DECISION_LABELS = {
    "build_now": "BUILD NOW",
    "prototype": "PROTOTYPE / VALIDATE FURTHER",
    "park": "PARK",
    "reject": "REJECT",
}

_DECISION_EMOJI = {
    "build_now": "🟢",
    "prototype": "🟡",
    "park": "🟠",
    "reject": "🔴",
}

_TRIAGE_REC_LABELS = {
    "fast_track": "FAST-TRACK (strong Z2O signal)",
    "proceed": "PROCEED (weak-to-moderate Z2O signal)",
    "park": "PARK (marginal signal)",
    "reject": "REJECT (insufficient signal)",
}


def bar(score: int, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> str:
    return f"\n── {title} " + "─" * max(0, 72 - len(title) - 4)


def _field(label: str, value: str, indent: int = 2) -> str:
    pad = " " * indent
    label_col = f"{label:<28}"
    return f"{pad}{label_col}{value}"


def _list_field(label: str, items: list, indent: int = 2) -> list[str]:
    pad = " " * indent
    lines = []
    for i, item in enumerate(items):
        prefix = f"{label:<28}" if i == 0 else " " * 28
        lines.append(f"{pad}{prefix}{item}")
    return lines


def generate(result: ValidationResult) -> str:
    idea = result.idea
    t = result.triage
    s = result.scores

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"VALIDATION MEMO — {idea.title.upper()}")
    lines.append("=" * 72)
    lines.append(f"Submitter : {idea.submitter}")
    lines.append(f"Track     : {idea.track.value.upper()}")
    lines.append(f"Validated : {result.validated_at[:10]}")
    lines.append(f"Stage     : {result.stage_reached}")
    if idea.source_context:
        lines.append(f"Context   : {idea.source_context}")
    lines.append("")

    # ── Triage ────────────────────────────────────────────────────────────────
    if t:
        lines.append(_section("Z2O TRIAGE"))
        lines.append(f"Total score : {t.total_score}/20  →  {_TRIAGE_REC_LABELS.get(t.triage_recommendation, t.triage_recommendation)}")
        lines.append("")
        dims = [
            ("Secret        ", t.secret),
            ("10x Test      ", t.ten_x),
            ("Timing        ", t.timing),
            ("Monopoly      ", t.monopoly),
            ("Lazybones Fit ", t.lazybones_fit),
        ]
        for label, dim in dims:
            stars = "★" * dim.score + "☆" * (4 - dim.score)
            lines.append(f"  {label} {stars} ({dim.score}/4)")
            lines.append(f"    {dim.rationale}")
        lines.append("")
        lines.append(f"  Summary: {t.triage_summary}")
        lines.append("")

        if t.total_score < 8:
            lines.append("⛔ BELOW TRIAGE GATE — DE research not run.")
            lines.append("=" * 72)
            return "\n".join(lines)

    # ── Composite Scores ──────────────────────────────────────────────────────
    if s:
        lines.append(_section("COMPOSITE SCORES"))
        da = _DECISION_LABELS.get(s.decision_a, s.decision_a)
        db = _DECISION_LABELS.get(s.decision_b, s.decision_b)
        ea = _DECISION_EMOJI.get(s.decision_a, "")
        eb = _DECISION_EMOJI.get(s.decision_b, "")
        lines.append(f"  Score A (Internal Xavor) : {s.composite_score_a:>3}/100  {ea} {da}")
        lines.append(f"  Score B (External Prod)  : {s.composite_score_b:>3}/100  {eb} {db}")
        lines.append("")

        flags = []
        if s.quick_win_flag:
            flags.append("⚡ QUICK WIN")
        if s.strategic_bet_flag:
            flags.append("🎯 STRATEGIC BET")
        if not s.economics_gate_pass:
            flags.append("⚠️  ECONOMICS GATE FAILED (LTV/COCA < 3 — external score capped at 55)")
        if not s.data_readiness_gate_pass:
            flags.append("⚠️  DATA BLOCKER (readiness < 30 — feasibility dimension capped)")
        if flags:
            for f in flags:
                lines.append(f"  {f}")
            lines.append("")

        lines.append(_section("DIMENSION SCORES"))
        for dim in s.dimension_breakdown:
            pct = f"{int(dim.weight * 100)}%"
            lines.append(
                f"  {dim.name:<28} {dim.score:>3}/100  {bar(dim.score, 16)}  "
                f"(wt {pct}  Z2O:{dim.z2o_component}  DE:{dim.de_component})"
            )
        lines.append("")
        lines.append(f"  Strongest : {s.strongest_dimension}")
        lines.append(f"  Weakest   : {s.weakest_dimension}")
        lines.append("")

    # ── Customer Agent ────────────────────────────────────────────────────────
    c = result.customer
    if c:
        lines.append(_section("CUSTOMER ANALYSIS"))
        lines.append(_field("Score", f"{c.customer_score}/100  (confidence: {c.confidence}/100)"))
        lines.append("")
        lines.append(_field("Beachhead Definition", c.beachhead_definition))
        lines.append("")
        lines.append("  End-User Persona")
        lines.append(f"    {c.end_user_persona}")
        lines.append("")
        lines.append("  Buyer Persona")
        lines.append(f"    {c.buyer_persona}")
        lines.append("")
        lines.append("  Current Workflow (Before)")
        lines.append(f"    {c.workflow_before}")
        lines.append("")
        lines.append("  Trigger Event")
        lines.append(f"    {c.trigger_event}")
        lines.append("")
        lines.append("  Lifecycle Use Case")
        lines.append(f"    {c.lifecycle_use_case}")
        lines.append("")
        lines.append("  Beachhead Size Estimate")
        lines.append(f"    {c.beachhead_size_estimate}")
        lines.append("")
        if c.internal_deployment_scope:
            lines.append("  Internal Deployment Scope")
            lines.append(f"    {c.internal_deployment_scope}")
            lines.append("")
        lines.append(_field("Confidence Rationale", c.confidence_rationale))
        lines.append(_field("Weakest Aspect", c.weakest_aspect))
        if c.search_queries_used:
            lines.extend(_list_field("Search Queries Used", c.search_queries_used))
        lines.append("")

    # ── Value Prop Agent ──────────────────────────────────────────────────────
    vp = result.value_prop
    if vp:
        lines.append(_section("VALUE PROPOSITION"))
        lines.append(_field("Score", f"{vp.value_prop_score}/100  (confidence: {vp.confidence}/100)"))
        lines.append(_field("Type", vp.vitamin_or_painkiller.upper()))
        lines.append(_field("Pain Score", f"{vp.pain_score}/10"))
        lines.append("")
        lines.append("  Pain Evidence")
        lines.append(f"    {vp.pain_evidence}")
        lines.append("")
        lines.append("  Current Workarounds")
        lines.append(f"    {vp.current_workarounds}")
        lines.append("")
        lines.append("  Before → After")
        lines.append(f"    {vp.before_after_statement}")
        lines.append("")
        lines.append("  Quantified Value")
        lines.append(f"    {vp.quantified_value_statement}")
        lines.append("")
        lines.append("  Top Competitor")
        lines.append(f"    {vp.top_competitor}")
        lines.append("")
        lines.append("  Differentiation")
        lines.append(f"    {vp.differentiation}")
        lines.append("")
        lines.append(_field("Confidence Rationale", vp.confidence_rationale))
        lines.append(_field("Weakest Aspect", vp.weakest_aspect))
        if vp.search_queries_used:
            lines.extend(_list_field("Search Queries Used", vp.search_queries_used))
        lines.append("")

    # ── Acquisition Agent ─────────────────────────────────────────────────────
    acq = result.acquisition
    if acq:
        lines.append(_section("ACQUISITION & GO-TO-MARKET"))
        lines.append(_field("Score", f"{acq.acquisition_score}/100  (confidence: {acq.confidence}/100)"))
        lines.append("")
        lines.append("  Decision-Making Unit Map")
        lines.append(f"    {acq.dmu_map}")
        lines.append("")
        lines.append("  Adoption Process")
        lines.append(f"    {acq.adoption_process}")
        lines.append("")
        if acq.internal_approval_path:
            lines.append("  Internal Approval Path")
            lines.append(f"    {acq.internal_approval_path}")
            lines.append("")
        lines.append("  Channel Recommendation")
        lines.append(f"    {acq.channel_recommendation}")
        lines.append("")
        lines.append(_field("COCA Estimate", acq.coca_estimate))
        lines.append("")
        lines.append("  Comparable Tools Playbook")
        lines.append(f"    {acq.comparable_tools_playbook}")
        lines.append("")
        lines.append(_field("Confidence Rationale", acq.confidence_rationale))
        lines.append(_field("Weakest Aspect", acq.weakest_aspect))
        if acq.search_queries_used:
            lines.extend(_list_field("Search Queries Used", acq.search_queries_used))
        lines.append("")

    # ── Economics Agent ───────────────────────────────────────────────────────
    econ = result.economics
    if econ:
        gate_str = "✅ PASS" if econ.economics_gate_pass else "❌ FAIL"
        lines.append(_section("UNIT ECONOMICS"))
        lines.append(_field("Score", f"{econ.economics_score}/100  (confidence: {econ.confidence}/100)"))
        lines.append(_field("LTV/COCA Gate", f"{econ.ltv_coca_ratio:.2f}  {gate_str}  (threshold ≥ 3.0)"))
        lines.append("")
        lines.append("  Business Model")
        lines.append(f"    {econ.business_model}")
        lines.append("")
        lines.append("  Pricing Estimate")
        lines.append(f"    {econ.pricing_estimate}")
        lines.append("")
        lines.append("  LTV Calculation")
        lines.append(f"    {econ.ltv_estimate}")
        lines.append("")
        lines.append("  COCA Estimate")
        lines.append(f"    {econ.coca_estimate}")
        lines.append("")
        if econ.manual_process_cost_estimate:
            lines.append("  Manual Process Cost (Internal)")
            lines.append(f"    {econ.manual_process_cost_estimate}")
            lines.append("")
        if econ.roi_months_internal:
            lines.append(_field("Internal ROI Break-even", econ.roi_months_internal))
            lines.append("")
        lines.append(_field("Confidence Rationale", econ.confidence_rationale))
        lines.append(_field("Weakest Aspect", econ.weakest_aspect))
        if econ.search_queries_used:
            lines.extend(_list_field("Search Queries Used", econ.search_queries_used))
        lines.append("")

    # ── Feasibility Agent ─────────────────────────────────────────────────────
    feas = result.feasibility
    if feas:
        lines.append(_section("FEASIBILITY"))
        lines.append(_field("Score", f"{feas.feasibility_score}/100  (confidence: {feas.confidence}/100)"))
        lines.append(_field("Data Readiness", f"{feas.data_readiness_score}/100"))
        lines.append(_field("Build Recommendation", f"{feas.build_vs_buy_recommendation.upper()}"))
        lines.append(_field("MVP Timeline", f"{feas.mvp_timeline_estimate} (~{feas.mvp_timeline_days} days)"))
        lines.append("")
        lines.append("  MVP Scope")
        lines.append(f"    {feas.mvp_scope}")
        lines.append("")
        lines.append("  Data Requirements")
        lines.append(f"    {feas.data_requirements}")
        lines.append("")
        lines.append("  Integration Requirements")
        lines.append(f"    {feas.integration_requirements}")
        lines.append("")
        lines.append("  Build vs Buy Rationale")
        lines.append(f"    {feas.build_vs_buy_rationale}")
        lines.append("")
        if feas.top_technical_risks:
            lines.append("  Technical Risks")
            for i, risk in enumerate(feas.top_technical_risks, 1):
                lines.append(f"    {i}. {risk}")
            lines.append("")
        lines.append(_field("Confidence Rationale", feas.confidence_rationale))
        lines.append(_field("Weakest Aspect", feas.weakest_aspect))
        if feas.search_queries_used:
            lines.extend(_list_field("Search Queries Used", feas.search_queries_used))
        lines.append("")

    # ── Scale & Moat Agent ────────────────────────────────────────────────────
    sc = result.scale
    if sc:
        lines.append(_section("SCALE & MOAT"))
        lines.append(_field("Score", f"{sc.scale_score}/100  (confidence: {sc.confidence}/100)"))
        lines.append(_field("Moat Type", sc.moat_type))
        lines.append(_field("Moat Strength", f"{sc.moat_strength_score}/100"))
        lines.append(_field("External Potential", sc.external_product_potential.upper()))
        lines.append("")
        lines.append("  Durability Assessment")
        lines.append(f"    {sc.durability_assessment}")
        lines.append("")
        lines.append("  External Product Rationale")
        lines.append(f"    {sc.external_product_rationale}")
        lines.append("")
        lines.append("  Strategic Leverage")
        lines.append(f"    {sc.strategic_leverage_description}")
        lines.append("")
        if sc.reuse_potential_internal:
            lines.append("  Internal Reuse Potential")
            lines.append(f"    {sc.reuse_potential_internal}")
            lines.append("")
        if sc.next_beachheads_external:
            lines.append("  Next Beachheads (External)")
            for bh in sc.next_beachheads_external:
                lines.append(f"    • {bh}")
            lines.append("")
        lines.append(_field("Confidence Rationale", sc.confidence_rationale))
        lines.append(_field("Weakest Aspect", sc.weakest_aspect))
        if sc.search_queries_used:
            lines.extend(_list_field("Search Queries Used", sc.search_queries_used))
        lines.append("")

    # ── Recommended Next Action ───────────────────────────────────────────────
    if s:
        lines.append(_section("RECOMMENDED NEXT ACTION"))
        lines.append(f"  {s.recommended_next_action}")
        lines.append("")

    # ── Agent Errors ──────────────────────────────────────────────────────────
    if result.agent_errors:
        lines.append(_section("AGENT ERRORS"))
        for agent, err in result.agent_errors.items():
            lines.append(f"  [{agent}]")
            for err_line in err.strip().splitlines():
                lines.append(f"    {err_line}")
            lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
