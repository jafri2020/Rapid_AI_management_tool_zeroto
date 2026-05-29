"""Generates the formatted validation memo from a ValidationResult."""
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


def generate(result: ValidationResult) -> str:
    idea = result.idea
    t = result.triage
    s = result.scores

    lines = []
    lines.append("=" * 72)
    lines.append(f"VALIDATION MEMO — {idea.title.upper()}")
    lines.append("=" * 72)
    lines.append(f"Submitter : {idea.submitter}")
    lines.append(f"Track     : {idea.track.value.upper()}")
    lines.append(f"Validated : {result.validated_at[:10]}")
    lines.append(f"Stage     : {result.stage_reached}")
    lines.append("")

    # ── Triage ────────────────────────────────────────────────────────────────
    if t:
        lines.append("── Z2O TRIAGE ──────────────────────────────────────────────────────────")
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
            lines.append(f"  {label} {stars} ({dim.score}/4)  {dim.rationale}")
        lines.append("")
        lines.append(f"  Summary: {t.triage_summary}")
        lines.append("")

        if t.total_score < 8:
            lines.append("⛔ BELOW TRIAGE GATE — DE research not run.")
            lines.append("=" * 72)
            return "\n".join(lines)

    # ── Composite Scores ──────────────────────────────────────────────────────
    if s:
        lines.append("── COMPOSITE SCORES ────────────────────────────────────────────────────")
        da = _DECISION_LABELS.get(s.decision_a, s.decision_a)
        db = _DECISION_LABELS.get(s.decision_b, s.decision_b)
        ea = _DECISION_EMOJI.get(s.decision_a, "")
        eb = _DECISION_EMOJI.get(s.decision_b, "")
        lines.append(f"  Score A (Internal Xavor) : {s.composite_score_a:>3}/100  {ea} {da}")
        lines.append(f"  Score B (External Prod)  : {s.composite_score_b:>3}/100  {eb} {db}")
        lines.append("")

        # Flags
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

        # Dimension breakdown
        lines.append("── DIMENSION SCORES ────────────────────────────────────────────────────")
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

    # ── DE Research Findings ──────────────────────────────────────────────────
    lines.append("── KEY FINDINGS ────────────────────────────────────────────────────────")

    c = result.customer
    if c:
        lines.append(f"  CUSTOMER     {c.beachhead_definition}")
        lines.append(f"               End user: {c.end_user_persona[:120]}")
        lines.append(f"               Beachhead size: {c.beachhead_size_estimate}")

    vp = result.value_prop
    if vp:
        lines.append(f"  PAIN         {vp.pain_score}/10 — {vp.before_after_statement}")
        lines.append(f"  VALUE        {vp.quantified_value_statement}")
        lines.append(f"  COMPETITION  {vp.top_competitor} | Differentiation: {vp.differentiation[:100]}")
        lines.append(f"  TYPE         {vp.vitamin_or_painkiller.upper()}")

    econ = result.economics
    if econ:
        gate_str = "✅ PASS" if econ.economics_gate_pass else "❌ FAIL"
        lines.append(f"  ECONOMICS    Model: {econ.business_model} | Pricing: {econ.pricing_estimate}")
        lines.append(f"               LTV/COCA: {econ.ltv_coca_ratio:.1f}  {gate_str}")
        if econ.roi_months_internal:
            lines.append(f"               Internal ROI: {econ.roi_months_internal}")

    feas = result.feasibility
    if feas:
        lines.append(f"  MVP SCOPE    {feas.mvp_scope[:140]}")
        lines.append(f"               Timeline: {feas.mvp_timeline_estimate} | Build: {feas.build_vs_buy_recommendation.upper()}")
        lines.append(f"               Data readiness: {feas.data_readiness_score}/100")
        if feas.top_technical_risks:
            lines.append(f"               Top risk: {feas.top_technical_risks[0]}")

    sc = result.scale
    if sc:
        lines.append(f"  MOAT         {sc.moat_type} (strength: {sc.moat_strength_score}/100)")
        lines.append(f"               Durability: {sc.durability_assessment[:120]}")
        lines.append(f"               External potential: {sc.external_product_potential.upper()}")

    lines.append("")

    # ── Next Action ───────────────────────────────────────────────────────────
    if s:
        lines.append("── RECOMMENDED NEXT ACTION ─────────────────────────────────────────────")
        lines.append(f"  {s.recommended_next_action}")
        lines.append("")

    # ── Agent errors ─────────────────────────────────────────────────────────
    if result.agent_errors:
        lines.append("── AGENT ERRORS ─────────────────────────────────────────────────────────")
        for agent, err in result.agent_errors.items():
            lines.append(f"  [{agent}]")
            for err_line in err.strip().splitlines():
                lines.append(f"    {err_line}")
            lines.append("")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
