"""DE Theme 1 — Customer Agent."""
from models import CustomerOutput, IdeaInput, TriageOutput
from agents.base import BaseAgent

SYSTEM = """You are a customer research specialist applying Bill Aulet's Disciplined Entrepreneurship
framework (Theme 1: Who Is Your Customer?).

Your job: given an AI idea, rigorously define the beachhead market and customer profile.
Vague answers ("enterprise companies", "all knowledge workers") are unacceptable. You must name
specific roles, specific company types, specific workflow moments.

Context: You are evaluating ideas for Xavor AI Foundry — a small AI team inside Xavor Corporation
(IT services, Pakistan-based, global clients). Ideas may target internal Xavor deployment OR
the external market as a standalone product, or both.

Use web search results to ground your estimates. Cite evidence where possible.
Output only valid JSON."""

REASON_TEMPLATE = """Analyze the customer profile for this idea using DE Theme 1 principles.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O TRIAGE CONTEXT:
{triage_summary}

Think through:

BEACHHEAD MARKET
The beachhead is the single most winnable, homogeneous group of customers. NOT the total addressable
market. NOT "all IT companies." The smallest market that, if fully captured, gives you a foothold.
For internal track: which specific Xavor department and workflow?
For external track: which specific industry + role + company size?

END USER vs. BUYER vs. CHAMPION
DE distinguishes:
- End user: the person who uses the tool daily. What is their day like? What frustrates them?
- Economic buyer: who signs the check / approves the purchase?
- Champion: who wants this to exist and will fight internally to adopt it?
These may be the same person or different people. Map them explicitly.

WORKFLOW BEFORE
Walk through the end user's workflow today, step by step. Where exactly does the pain occur?
What does the output of their current process look like? How long does it take?

TRIGGER EVENT
What specific moment or situation causes someone to realize they need this solution?
(e.g., "After the third proposal revision comes back with the same comment...")

LIFECYCLE USE CASE
How does the product fit into the full work lifecycle? What triggers use, what happens during use,
and what does the user do with the output?

BEACHHEAD SIZE
For the beachhead market: how many people fit this profile? What frequency do they use this?
Ground your estimate in real data if possible."""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results:
{search_results}

Output ONLY this JSON object — be specific, use real job titles, real department names, real
workflow steps. Avoid vague generalities.

{{
  "beachhead_definition": "<one crisp sentence: [role] at [company type/size] who [specific workflow context]>",
  "end_user_persona": "<job title, company context, daily workflow, what they hate about the status quo>",
  "buyer_persona": "<who approves the purchase/adoption — same as end user or different? explain>",
  "workflow_before": "<step-by-step description of the workflow today, where the pain is, how long it takes>",
  "trigger_event": "<the specific moment that makes someone realize they need this>",
  "lifecycle_use_case": "<what triggers use → what happens during use → what the output feeds into>",
  "beachhead_size_estimate": "<estimated number of people in beachhead × frequency of use, with evidence>",
  "internal_deployment_scope": "<for internal track: which Xavor department(s), estimated headcount, rollout complexity>",
  "customer_score": <0-100, reflecting clarity and specificity of customer definition>,
  "confidence": <0-100>,
  "confidence_rationale": "<why not higher — what would increase this score>",
  "weakest_aspect": "<the single biggest gap or uncertainty in the customer analysis>"
}}"""


class CustomerAgent(BaseAgent):
    name = "DE Customer"

    def run(self, idea: IdeaInput, triage: TriageOutput) -> CustomerOutput:
        triage_summary = (
            f"Secret: {triage.secret.score}/4 — {triage.secret.rationale}\n"
            f"10x: {triage.ten_x.score}/4 — {triage.ten_x.rationale}\n"
            f"Monopoly: {triage.monopoly.score}/4 — {triage.monopoly.rationale}\n"
            f"Summary: {triage.triage_summary}"
        )
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            track=idea.track.value,
            triage_summary=triage_summary,
        )
        return self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=GENERATE_TEMPLATE,
            model_class=CustomerOutput,
        )
