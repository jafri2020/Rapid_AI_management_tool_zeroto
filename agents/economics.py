"""DE Theme 4 — Economics Agent."""
from models import (
    AcquisitionOutput,
    CustomerOutput,
    EconomicsOutput,
    IdeaInput,
    TriageOutput,
    ValuePropOutput,
)
from agents.base import BaseAgent

SYSTEM = """You are a unit economics specialist applying Bill Aulet's Disciplined Entrepreneurship
framework (Theme 4: How Do You Make Money?).

Your job: estimate the financial viability of this idea — business model, pricing, LTV, COCA, and
the LTV/COCA ratio. The hard gate is LTV/COCA ≥ 3. Below that, the external product economics
are weak regardless of other scores.

For internal Xavor tools: calculate ROI as (cost of current manual process) / (cost to build + run).
Use FTE cost estimates of ~$25-40/hr for Pakistan-based engineers, ~$15-25/hr for other staff.

Be conservative. It is better to underestimate LTV and over-flag weak economics than to approve
a financially doomed project. Ground all estimates in comparable SaaS benchmarks or real data.

Output only valid JSON."""

REASON_TEMPLATE = """Analyze the economics and financial viability of this idea.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O SIGNALS:
{triage_summary}

VALUE PROP (foundation for pricing):
{value_prop_summary}

ACQUISITION COSTS:
{acquisition_summary}

Think through:

BUSINESS MODEL
What is the most natural revenue model?
  Internal: cost savings, productivity gain, risk reduction (not revenue per se, but ROI)
  External: SaaS subscription (per seat or usage), API/usage-based, one-time license, outcome-based
What models do comparable tools use? Why does that model fit this use case?

PRICING
For external: What is the willingness to pay? Research competitor pricing.
  - Bottom-up: what would an individual user pay per month?
  - Top-down: what would a team/company pay per month?
  Triangulate from: (a) comparable tool pricing, (b) % of value delivered, (c) willingness to pay
  testing norms (customers typically pay 5-15% of quantified value annually)

For internal: what is the current manual process cost?
  Estimate: [hours per week] × [number of people] × [hourly cost] × 52 weeks = annual cost
  Build cost: [weeks to MVP] × [2 engineers] × [cost per week]

LTV (Lifetime Value)
External: Monthly Revenue per customer × Gross Margin × Average months retained
  (For early-stage B2B SaaS: typical retention 24-48 months, gross margin 60-80%)
Internal: Annualized cost savings

COCA (Cost of Customer Acquisition)
Pull from acquisition analysis. Be realistic about blended COCA including sales time.

LTV/COCA RATIO
This is the hard gate. <3 = weak. 3-5 = acceptable. >5 = strong. >10 = exceptional.
Most early-stage SaaS tools struggle to get above 3 until they prove retention.

ROI FOR INTERNAL TRACK
Months to break even = Build cost / Monthly savings
Target: ROI within 3-6 months for internal tools."""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results (competitor pricing, market data):
{search_results}

Output ONLY this JSON object. Use real numbers. Ranges are acceptable (e.g., "$200-400/month").
Never invent figures — if genuinely unknown, say "insufficient data" and score confidence lower.

{{
  "business_model": "<primary model: SaaS/usage/license for external, cost-savings/ROI for internal, with rationale>",
  "pricing_estimate": "<for external: per-seat or usage price range with benchmarks. For internal: annual cost of status quo.>",
  "ltv_estimate": "<calculation: monthly revenue × gross margin % × retention months. Show your math.>",
  "coca_estimate": "<from acquisition analysis — blended cost including sales/marketing time>",
  "ltv_coca_ratio": <numeric ratio, e.g. 4.2>,
  "economics_gate_pass": <true if ltv_coca_ratio >= 3.0, false otherwise>,
  "roi_months_internal": "<for internal track: months to break even on build cost. E.g. '2.5 months'>",
  "manual_process_cost_estimate": "<for internal track: [X hrs/week] × [N people] × [$Y/hr] × 52 = $Z/year>",
  "economics_score": <0-100, reflecting overall financial attractiveness and evidence quality>,
  "confidence": <0-100>,
  "confidence_rationale": "<what's the biggest pricing or retention assumption you'd want to test first>",
  "weakest_aspect": "<biggest economic risk: pricing pressure, high COCA, low retention, weak margins, unclear model>"
}}"""


class EconomicsAgent(BaseAgent):
    name = "DE Economics"

    def run(
        self,
        idea: IdeaInput,
        triage: TriageOutput,
        value_prop: ValuePropOutput,
        acquisition: AcquisitionOutput,
    ) -> EconomicsOutput:
        triage_summary = (
            f"10x: {triage.ten_x.score}/4 — {triage.ten_x.rationale}\n"
            f"Secret: {triage.secret.score}/4 — {triage.secret.rationale}"
        )
        value_prop_summary = (
            f"Quantified value: {value_prop.quantified_value_statement}\n"
            f"Pain score: {value_prop.pain_score}/10\n"
            f"Top competitor: {value_prop.top_competitor}"
        )
        acquisition_summary = (
            f"COCA estimate: {acquisition.coca_estimate}\n"
            f"Channel: {acquisition.channel_recommendation}"
        )
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            track=idea.track.value,
            triage_summary=triage_summary,
            value_prop_summary=value_prop_summary,
            acquisition_summary=acquisition_summary,
        )
        return self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=GENERATE_TEMPLATE,
            model_class=EconomicsOutput,
        )
