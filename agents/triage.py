"""Z2O Triage Agent — rapid strategic filter before DE deep research."""
from models import IdeaInput, TriageOutput
from agents.base import BaseAgent

SYSTEM = """You are a Zero to One strategic analyst evaluating AI/software ideas for a small 3-person AI
engineering team (called lazybones) inside Xavor Corporation — an IT services, cloud, and product
engineering company based in Pakistan with global clients.

The team's mission: rapidly adopt AI across Xavor's internal functions AND package successful
workflows into standalone external products to sell.

You score ideas on 5 Zero to One dimensions, each 0-4:
  0 = No signal (anyone could do this, nothing special here)
  1 = Weak signal (maybe, but doubtful)
  2 = Clear signal (real evidence of this quality)
  3 = Strong signal (compelling, hard to argue against)
  4 = Exceptional (rare — reserve for genuinely non-obvious insights)

Be ruthlessly honest. A score of 4 should be rare. Most ideas score 1-2 on most dimensions.
Inflate nothing — a false positive wastes the team's most limited resource: time.

Output only valid JSON. No markdown, no commentary outside the JSON."""

REASON_TEMPLATE = """Evaluate this idea carefully before scoring.

IDEA: {title}
DESCRIPTION: {description}
SOURCE CONTEXT: {source_context}
TRACK: {track}

Think through each of the 5 dimensions in detail:

1. THE SECRET TEST
   A secret is something true that most people either don't know or actively disagree with.
   Xavor's potential secrets: deep knowledge of IT services delivery pain points, real enterprise
   client relationships, insider view of which AI promises fail in practice, ability to move fast
   inside enterprise constraints.
   Ask: Does this idea rest on such an insight? Or is it "AI for [obvious thing]" that any vendor
   could build with no special knowledge?

2. THE 10X TEST
   Not "10% faster." Not "a bit more accurate." 10x means: eliminates a category of manual work,
   unlocks something previously impossible, or creates an order-of-magnitude shift in the economics
   of a workflow.
   Ask: What exactly is 10x better? Can you state it precisely? Is there evidence the current
   approach is that painful?

3. THE TIMING TEST
   Why now? What changed in the last 12-18 months — a new model capability, a new API, a new
   internal pain point, a regulatory shift, a competitor's failure — that makes this the right
   moment? An idea without a timing story is a feature idea, not a product idea.

4. THE MONOPOLY TEST
   Zero to One means dominating a small market completely before expanding. What is the narrowest
   possible beachhead — a specific workflow, a specific team, a specific industry + role — that
   this could own 100% of first? Broad ideas score low here.

5. THE LAZYBONES FIT TEST
   Specific to this team: Hussain (VP-level air cover + Xavor client relationships), Taha and
   Hasnain (AI engineering + rapid build capability), and the unique position inside an IT
   services company with real enterprise client workflows. Does this team have an edge that would
   make it hard for a random startup to compete? Or could any three engineers build this?

After thinking through each, produce your scored output."""

GENERATE_TEMPLATE = """Based on your analysis:
---
{reasoning}
---

Search context (if any):
{search_results}

Now produce the final JSON for this idea. Scores must reflect your actual analysis above — do NOT
round up. Output ONLY this JSON object:

{{
  "secret":        {{"score": <0-4>, "rationale": "<1-2 sentences — be specific>"}},
  "ten_x":         {{"score": <0-4>, "rationale": "<1-2 sentences — what exactly is 10x>"}},
  "timing":        {{"score": <0-4>, "rationale": "<1-2 sentences — what changed recently>"}},
  "monopoly":      {{"score": <0-4>, "rationale": "<1-2 sentences — name the specific beachhead>"}},
  "lazybones_fit": {{"score": <0-4>, "rationale": "<1-2 sentences — what specific edge does this team have>"}},
  "total_score":   <sum of all 5 scores>,
  "triage_recommendation": "<fast_track|proceed|park|reject>",
  "triage_summary": "<2-3 sentences: overall signal, biggest strength, biggest concern>"
}}"""


class TriageAgent(BaseAgent):
    name = "Z2O Triage"

    def run(self, idea: IdeaInput) -> TriageOutput:
        reason_prompt = REASON_TEMPLATE.format(
            title=idea.title,
            description=idea.description,
            source_context=idea.source_context or "Not specified",
            track=idea.track.value,
        )
        gen_template = GENERATE_TEMPLATE

        # Triage doesn't benefit much from web search — it's a strategic judgement.
        # Still run the pipeline for the critique step.
        result = self.run_pipeline(
            system=SYSTEM,
            reason_prompt=reason_prompt,
            generate_prompt_template=gen_template,
            model_class=TriageOutput,
        )
        return result
