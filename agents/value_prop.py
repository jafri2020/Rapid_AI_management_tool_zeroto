"""DE Theme 2 — Value Proposition Agent."""
from models import CustomerOutput, IdeaInput, TriageOutput, ValuePropOutput
from agents.base import BaseAgent

SYSTEM = """You are a value proposition specialist applying Bill Aulet's Disciplined Entrepreneurship
framework (Theme 2: What Can You Do For Your Customer?).

Your job: rigorously quantify the value this idea delivers. Vague claims ("saves time", "improves
quality") are unacceptable. You need to get to numbers: how many hours saved, what error rate
reduced from X% to Y%, what cost cut from $N to $M.

Context: Xavor AI Foundry — evaluating ideas for internal Xavor deployment and/or external
product. Use web search to find benchmark data on current tool costs, market pain evidence,
and competitor pricing.

Output only valid JSON."""

REASON_TEMPLATE = """Analyze the value proposition for this idea using DE Theme 2 principles.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O TRIAGE CONTEXT:
{triage_summary}

CUSTOMER PROFILE:
{customer_summary}

Think through:

PAIN SEVERITY (1-10)
How painful is the current situation? 10 = people lose jobs over this / company-ending risk.
7-9 = significant daily friction, measurable cost. 4-6 = annoying but manageable. 1-3 = minor.
Find evidence: job postings mentioning this pain, forum complaints, industry reports, competitor
existence (if competitors exist and are funded, the pain is real).

CURRENT WORKAROUNDS
What do people do TODAY to solve this problem? What tools do they use? Manual processes?
Workarounds reveal the true cost of the problem. A painful, expensive workaround = strong signal.

QUANTIFIED VALUE
This is the heart of DE Theme 2. Produce a specific "from X to Y" statement:
  "Currently this task takes [X hours] per [week/month]. This tool reduces it to [Y hours]."
  "Error rate goes from [X%] to [Y%]."
  "Cost per [unit] drops from [$X] to [$Y]."
Search for benchmark data to support these estimates.

COMPETITIVE LANDSCAPE
What existing tools address this space? What do they charge? What are they missing?
The differentiation must be specific — not "we use AI" but "we do X that they don't because Y."

VITAMIN vs. PAINKILLER
Vitamins are nice-to-have. Painkillers solve acute pain. Painkillers get bought, vitamins get
evaluated and shelved. Be honest about which this is."""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results (use these to ground your estimates):
{search_results}

Output ONLY this JSON object — use real numbers, real competitor names, real evidence.
Do not fabricate figures. If uncertain, give a range and note it.

{{
  "pain_score": <1-10, with 10 being company-critical pain>,
  "pain_evidence": "<specific evidence: complaints, cost data, workaround existence, funded competitors>",
  "current_workarounds": "<what people do today: tools used, manual steps, estimated time/cost>",
  "quantified_value_statement": "<specific from-to statement with numbers: X hours → Y hours, $X → $Y, etc.>",
  "before_after_statement": "<one sentence: Before [this], [person] had to [painful thing]. After, they [specific improvement].>",
  "top_competitor": "<name the strongest existing solution and what it costs>",
  "differentiation": "<specific, falsifiable statement of what makes this 10x better than the top competitor>",
  "vitamin_or_painkiller": "<vitamin|painkiller|both>",
  "value_prop_score": <0-100, reflecting strength and specificity of the value proposition>,
  "confidence": <0-100>,
  "confidence_rationale": "<what data is missing that would increase this score>",
  "weakest_aspect": "<the single most important gap: missing evidence, weak differentiation, vitamin concern, etc.>"
}}"""


class ValuePropAgent(BaseAgent):
    name = "DE Value Prop"

    def run(
        self, idea: IdeaInput, triage: TriageOutput, customer: CustomerOutput
    ) -> ValuePropOutput:
        triage_summary = (
            f"10x signal: {triage.ten_x.score}/4 — {triage.ten_x.rationale}\n"
            f"Secret: {triage.secret.score}/4 — {triage.secret.rationale}"
        )
        customer_summary = (
            f"Beachhead: {customer.beachhead_definition}\n"
            f"End user: {customer.end_user_persona}\n"
            f"Workflow before: {customer.workflow_before}"
        )
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            track=idea.track.value,
            triage_summary=triage_summary,
            customer_summary=customer_summary,
        )
        return self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=GENERATE_TEMPLATE,
            model_class=ValuePropOutput,
        )
