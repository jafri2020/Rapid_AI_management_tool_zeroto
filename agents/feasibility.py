"""DE Theme 5 — Feasibility Agent."""
from models import (
    CustomerOutput,
    FeasibilityOutput,
    IdeaInput,
    TriageOutput,
    ValuePropOutput,
)
from agents.base import BaseAgent

SYSTEM = """You are a technical feasibility specialist applying Bill Aulet's Disciplined
Entrepreneurship framework (Theme 5: How Do You Design and Build Your Product?).

Your job: define the minimal viable product (MVP), assess what data and integrations are needed,
recommend build vs. buy, estimate the build timeline for a 2-person AI engineering team, and
surface the top technical risks.

Context: The Xavor AI Foundry team (Taha and Hasnain) are experienced AI engineers. They work in
Python, build agents and LLM pipelines, use Claude/OpenAI APIs, can build Streamlit/FastAPI apps,
and integrate with enterprise tools. They are NOT a large engineering team — they ship in days and
weeks, not quarters.

Technical stack assumptions: Python (primary), Anthropic/OpenAI APIs, Streamlit or Next.js for
UI, FastAPI for backends, SQLite/PostgreSQL for storage, Fibery and Discord for integrations.

Output only valid JSON."""

REASON_TEMPLATE = """Assess the technical feasibility of building this AI idea.

IDEA: {title}
DESCRIPTION: {description}
TRACK: {track}

Z2O ENGINEERING SIGNAL:
{triage_summary}

TARGET USER AND WORKFLOW:
{customer_summary}

VALUE TO DELIVER:
{value_prop_summary}

Think through:

MVP SCOPE
What is the absolute minimum version of this that would prove the core hypothesis?
"Minimum" means: one workflow, one integration, one user type, no admin panel, no multi-tenancy,
no edge cases. What can be cut without breaking the core value test?
The MVP must be demonstrable to a real user within the timeline estimate.

DATA REQUIREMENTS
What data does this AI system need to function?
  - Training data? (fine-tuning, RAG corpus, examples)
  - Runtime data? (what the LLM needs to see to do its job)
  - User data? (what the user inputs)
What is the data readiness score (0-100)?
  - 0-30: Data doesn't exist, inaccessible, or requires months of collection
  - 31-60: Data exists but needs significant cleaning, structuring, or access negotiation
  - 61-80: Data mostly ready, minor gaps
  - 81-100: Data clean, accessible, well-structured, ready to use today

INTEGRATION REQUIREMENTS
What systems does this need to connect to?
  - Internal Xavor systems? (project management, HR systems, client portals)
  - External APIs? (email, Slack, calendar, CRM, ERP)
  - Complexity: each integration adds risk and time — be realistic

BUILD vs. BUY vs. INTEGRATE
Could an existing tool (with configuration or light customization) solve 80% of this?
  build: no existing tool, unique AI capability required
  buy: existing SaaS tool covers it — focus on adoption, not building
  integrate: build thin layer on top of existing tool/API
  hybrid: buy the infrastructure, build the domain-specific AI layer

MVP TIMELINE
For a 2-person AI engineering team (Taha + Hasnain), working focused on this:
  - What are the discrete build steps?
  - What is the realistic calendar time? (be honest about complexity)
  - Provide a numeric estimate in days (for Quick Win flag: ≤7 days)

TOP 3 TECHNICAL RISKS
What could cause the MVP to fail or take 3x longer than expected?"""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search results (relevant tools, APIs, technical approaches):
{search_results}

Output ONLY this JSON object:

{{
  "mvp_scope": "<precise description of the minimum version: one user type, one workflow, specific inputs and outputs, what is explicitly out of scope>",
  "data_requirements": "<what data the system needs: source, format, volume estimate, current accessibility>",
  "data_readiness_score": <0-100, see rubric in your analysis>,
  "integration_requirements": "<list of systems/APIs needed with complexity assessment per integration>",
  "build_vs_buy_recommendation": "<build|buy|integrate|hybrid>",
  "build_vs_buy_rationale": "<specific tools considered, why build/buy/integrate wins, what exists in the market>",
  "mvp_timeline_estimate": "<e.g. '3-5 days', '2 weeks', '4-6 weeks' — be honest>",
  "mvp_timeline_days": <numeric estimate, best case, e.g. 5>,
  "top_technical_risks": [
    "<risk 1: description and why it could blow up the timeline>",
    "<risk 2: ...>",
    "<risk 3: ...>"
  ],
  "feasibility_score": <0-100, reflecting overall build feasibility for this specific team>,
  "confidence": <0-100>,
  "confidence_rationale": "<biggest unknown: data access? integration complexity? LLM reliability? user adoption of new workflow?>",
  "weakest_aspect": "<the single most likely reason this MVP would stall: data gap, integration complexity, user behavior change, LLM reliability, etc.>"
}}"""


class FeasibilityAgent(BaseAgent):
    name = "DE Feasibility"

    def run(
        self,
        idea: IdeaInput,
        triage: TriageOutput,
        customer: CustomerOutput,
        value_prop: ValuePropOutput,
    ) -> FeasibilityOutput:
        triage_summary = (
            f"Lazybones fit: {triage.lazybones_fit.score}/4 — {triage.lazybones_fit.rationale}\n"
            f"Timing: {triage.timing.score}/4 — {triage.timing.rationale}"
        )
        customer_summary = (
            f"End user: {customer.end_user_persona}\n"
            f"Workflow before: {customer.workflow_before}\n"
            f"Internal scope: {customer.internal_deployment_scope}"
        )
        value_prop_summary = (
            f"MVP must deliver: {value_prop.quantified_value_statement}\n"
            f"Current workarounds: {value_prop.current_workarounds}"
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
            model_class=FeasibilityOutput,
        )
