"""DE Theme 3 — Acquisition Agent."""
from models import AcquisitionOutput, CustomerOutput, IdeaInput, TriageOutput, ValuePropOutput
from agents.base import BaseAgent

SYSTEM = """You are a go-to-market and acquisition specialist applying Bill Aulet's Disciplined
Entrepreneurship framework (Theme 3: How Does the Customer Acquire the Product?).

Your job: map the decision-making unit and the complete acquisition path — from first awareness
to live adoption. For internal tools, this means the approval chain inside Xavor. For external
products, this means the GTM motion and estimated cost to acquire.

Context: Xavor AI Foundry — small AI team, resource-constrained, needing fast adoption wins.
They have direct access to internal Xavor stakeholders (Hussain = VP-level sponsor).

Output only valid JSON."""

REASON_TEMPLATE = """Analyze the acquisition and adoption path for this idea.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O TRIAGE — Timing & Distribution signals:
{triage_summary}

CUSTOMER CONTEXT:
{customer_summary}

VALUE PROP CONTEXT:
{value_prop_summary}

Think through:

DECISION-MAKING UNIT (DMU)
DE identifies distinct roles in a buying decision:
- Economic buyer (controls budget/approval)
- End user (uses it daily — their resistance or enthusiasm matters)
- Champion (internal advocate who wants it to exist)
- Saboteur (who might block it and why)
- Influencer (whose opinion the economic buyer trusts)

For INTERNAL Xavor adoption: Who in Xavor approves new tools? Is this a department head decision,
an IT decision, a VP decision? What is the procurement/approval cycle? Is Hussain's backing
enough to bypass normal procurement?

For EXTERNAL products: Who is the economic buyer? Is it the end user self-serve (bottom-up) or
a department head top-down sale? What is the typical sales cycle for this buyer type?

ADOPTION PROCESS
Step by step: How does someone go from "heard about this" to "using it daily"?
What friction points exist? (IT security review, data privacy, procurement, training, habit change)

CHANNEL
For internal: what's the deployment channel? (Slack bot, email, web app, API integration?)
For external: what's the GTM channel? (product-led growth, direct sales, content/SEO, partner?)
What channels are comparable tools using successfully?

COCA ESTIMATE
Cost to Acquire Customer. For internal: cost of the approval process (time × people × cycles).
For external: estimate based on channel (PLG ~ $50-500, inside sales ~ $1,000-5,000, etc.)
Research comparable tools' reported CAC if possible."""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results:
{search_results}

Output ONLY this JSON object:

{{
  "dmu_map": "<map each DMU role: economic buyer, end user, champion, likely saboteur, influencer>",
  "adoption_process": "<step-by-step: awareness → trial → approval → onboarding → habit. Flag friction points.>",
  "internal_approval_path": "<for internal track: who needs to approve, how many steps, estimated cycle time>",
  "channel_recommendation": "<primary acquisition channel with rationale, secondary channel if relevant>",
  "coca_estimate": "<estimated cost to acquire one customer/user, with benchmarks or reasoning>",
  "comparable_tools_playbook": "<how do 1-2 similar tools go to market? what can be learned from them?>",
  "acquisition_score": <0-100, reflecting ease and efficiency of the acquisition path>,
  "confidence": <0-100>,
  "confidence_rationale": "<what assumption is most uncertain here>",
  "weakest_aspect": "<biggest acquisition risk: long sales cycle, high COCA, strong gatekeepers, behavior change required, etc.>"
}}"""


class AcquisitionAgent(BaseAgent):
    name = "DE Acquisition"

    def run(
        self,
        idea: IdeaInput,
        triage: TriageOutput,
        customer: CustomerOutput,
        value_prop: ValuePropOutput,
    ) -> AcquisitionOutput:
        triage_summary = (
            f"Timing: {triage.timing.score}/4 — {triage.timing.rationale}\n"
            f"Fit: {triage.lazybones_fit.score}/4 — {triage.lazybones_fit.rationale}"
        )
        customer_summary = (
            f"Beachhead: {customer.beachhead_definition}\n"
            f"End user: {customer.end_user_persona}\n"
            f"Buyer: {customer.buyer_persona}"
        )
        value_prop_summary = (
            f"Pain score: {value_prop.pain_score}/10\n"
            f"Vitamin or painkiller: {value_prop.vitamin_or_painkiller}\n"
            f"Before/after: {value_prop.before_after_statement}"
        )
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            track=idea.track.value,
            triage_summary=triage_summary,
            customer_summary=customer_summary,
            value_prop_summary=value_prop_summary,
        )
        return self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=GENERATE_TEMPLATE,
            model_class=AcquisitionOutput,
        )
