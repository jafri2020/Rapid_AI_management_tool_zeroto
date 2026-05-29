"""DE Theme 6 — Scale & Moat Agent."""
from models import (
    CustomerOutput,
    EconomicsOutput,
    FeasibilityOutput,
    IdeaInput,
    ScaleOutput,
    TriageOutput,
    ValuePropOutput,
)
from agents.base import BaseAgent

SYSTEM = """You are a competitive strategy and scale specialist applying Bill Aulet's Disciplined
Entrepreneurship framework (Theme 6: How Do You Scale?) combined with Peter Thiel's durability
and moat thinking.

Your job: assess whether this idea can build a defensible moat, whether the advantage lasts,
what the next beachheads look like after the first, and how much internal reuse potential exists
for the Xavor team.

Context: Xavor AI Foundry — a small team that needs moats either from proprietary Xavor data and
workflows (internal advantage) or from network effects, switching costs, or deep integration
(external product advantage). Generic LLM wrappers have no moat and will be commoditized quickly.

Output only valid JSON."""

REASON_TEMPLATE = """Assess the moat, durability, and scale potential for this idea.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O SECRET AND TIMING:
{triage_summary}

CUSTOMER AND VALUE CONTEXT:
{customer_value_summary}

ECONOMICS AND FEASIBILITY:
{econ_feasibility_summary}

Think through:

MOAT TYPE AND STRENGTH
The possible moat types for AI tools:
1. Proprietary data: AI trained on unique data nobody else has (Xavor's internal docs, client
   workflow history, domain-specific labeled datasets)
2. Network effects: product gets better as more users join (data network effects, community)
3. Switching costs: deep integration into workflows makes leaving painful (data lock-in, trained
   models, customized pipelines)
4. Proprietary technology: genuinely novel AI technique, not just API calls to OpenAI
5. Brand/trust: established reputation in a specific niche
6. Process lock-in: so embedded in daily workflow that switching requires retraining + data migration

For a generic LLM wrapper: moat = near zero. Any competitor can replicate in days.
For a tool trained on Xavor's proprietary engagement data: moat = potentially strong.

DURABILITY ASSESSMENT
In 2-3 years, will this still be differentiated?
- Frontier models are getting more capable. Will GPT-6 / Claude 5 make this trivially replicable?
- Will the data advantage compound (more users → better data → better AI → more users)?
- What would it take for a better-funded competitor to copy this in 12 months?

INTERNAL REUSE POTENTIAL
For internal Xavor track: can the same core agent/pipeline be reused across multiple departments?
(e.g., a proposal agent for Sales could also work for Presales, then Engineering estimates)
High reuse = lower cost per workflow automated = stronger case for building.

NEXT BEACHHEADS (external track)
After dominating the first beachhead, what are the natural adjacent markets?
Each should be a specific role/industry/workflow, not a vague expansion.
Strong next beachheads share: same buyer type, same underlying technology, lower acquisition cost
because of existing reputation.

EXTERNAL PRODUCT POTENTIAL
Given everything above: how strong is the case for this becoming a standalone external product?
- high: clear moat, large adjacent markets, natural product-led expansion
- medium: some advantage but moat is thin, markets are niche
- low: internal tool, hard to generalize
- none: pure internal workflow, no external value"""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results (competitor moats, market expansion patterns):
{search_results}

Output ONLY this JSON object:

{{
  "moat_type": "<primary moat type with explanation: proprietary data / network effects / switching costs / tech / brand / process — or 'none' if it's a generic wrapper>",
  "moat_strength_score": <0-100, where 0=no moat, 50=some defensibility, 100=near-impenetrable advantage>,
  "durability_assessment": "<will this still be differentiated in 2-3 years? what could erode or strengthen the moat?>",
  "reuse_potential_internal": "<for internal track: what other Xavor functions could use the same core technology? estimate multiplier.>",
  "next_beachheads_external": [
    "<beachhead 2: specific role/industry/workflow>",
    "<beachhead 3: specific role/industry/workflow>",
    "<beachhead 4: specific role/industry/workflow>"
  ],
  "external_product_potential": "<high|medium|low|none>",
  "external_product_rationale": "<specific reasoning: what makes it generalizable or why it can't generalize>",
  "strategic_leverage_description": "<how does building this strengthen Xavor's capabilities, IP, or credibility in ways that compound?>",
  "scale_score": <0-100, reflecting moat strength + scale potential + strategic leverage>,
  "confidence": <0-100>,
  "confidence_rationale": "<biggest uncertainty: will the moat hold? will the market expand as expected?>",
  "weakest_aspect": "<the primary reason this might not scale: thin moat, niche market, poor reuse, low durability, etc.>"
}}"""


class ScaleAgent(BaseAgent):
    name = "DE Scale & Moat"

    def run(
        self,
        idea: IdeaInput,
        triage: TriageOutput,
        customer: CustomerOutput,
        value_prop: ValuePropOutput,
        economics: EconomicsOutput,
        feasibility: FeasibilityOutput,
    ) -> ScaleOutput:
        triage_summary = (
            f"Secret: {triage.secret.score}/4 — {triage.secret.rationale}\n"
            f"Timing: {triage.timing.score}/4 — {triage.timing.rationale}\n"
            f"Monopoly: {triage.monopoly.score}/4 — {triage.monopoly.rationale}"
        )
        customer_value_summary = (
            f"Beachhead: {customer.beachhead_definition}\n"
            f"Differentiation: {value_prop.differentiation}\n"
            f"Pain: {value_prop.pain_score}/10"
        )
        econ_feasibility_summary = (
            f"LTV/COCA: {economics.ltv_coca_ratio:.1f}\n"
            f"Business model: {economics.business_model}\n"
            f"Build recommendation: {feasibility.build_vs_buy_recommendation}"
        )
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            track=idea.track.value,
            triage_summary=triage_summary,
            customer_value_summary=customer_value_summary,
            econ_feasibility_summary=econ_feasibility_summary,
        )
        return self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=GENERATE_TEMPLATE,
            model_class=ScaleOutput,
        )
