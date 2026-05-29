"""Main pipeline orchestrator.

Runs:  IdeaInput → Triage → [gate] → 6 DE agents → Scoring → ValidationResult
Each agent error is caught and stored; partial results are still scored and saved.
"""
from __future__ import annotations

import traceback
from typing import Callable, Optional

from agents.acquisition import AcquisitionAgent
from agents.customer import CustomerAgent
from agents.economics import EconomicsAgent
from agents.feasibility import FeasibilityAgent
from agents.scale import ScaleAgent
from agents.scoring import compute as compute_scores
from agents.triage import TriageAgent
from agents.value_prop import ValuePropAgent
from config import config
from llm import LLMClient
from models import IdeaInput, ValidationResult
from search import SearchClient
from storage import Storage


class Pipeline:
    def __init__(self, storage: Optional[Storage] = None, on_progress: Optional[Callable[[str], None]] = None):
        self._llm = LLMClient()
        self._search = SearchClient()
        self._storage = storage or Storage(config.db_path)
        self._on_progress = on_progress or (lambda msg: print(f"[pipeline] {msg}"))

        self._triage_agent = TriageAgent(self._llm, self._search)
        self._customer_agent = CustomerAgent(self._llm, self._search)
        self._value_prop_agent = ValuePropAgent(self._llm, self._search)
        self._acquisition_agent = AcquisitionAgent(self._llm, self._search)
        self._economics_agent = EconomicsAgent(self._llm, self._search)
        self._feasibility_agent = FeasibilityAgent(self._llm, self._search)
        self._scale_agent = ScaleAgent(self._llm, self._search)

    def _progress(self, msg: str):
        self._on_progress(msg)

    def run(self, idea: IdeaInput) -> ValidationResult:
        result = ValidationResult(idea=idea)

        # ── Stage 1: Z2O Triage ───────────────────────────────────────────────
        self._progress("Running Z2O triage...")
        try:
            result.triage = self._triage_agent.run(idea)
            result.stage_reached = "triage"
        except Exception:
            result.agent_errors["triage"] = traceback.format_exc()
            self._progress(f"Triage failed: {result.agent_errors['triage'][:120]}")
            self._storage.save(result)
            return result

        triage = result.triage
        self._progress(
            f"Triage complete: {triage.total_score}/20 → {triage.triage_recommendation}"
        )

        # ── Triage gate ───────────────────────────────────────────────────────
        if triage.total_score < config.triage_gate_min:
            self._progress(
                f"Below triage gate ({triage.total_score} < {config.triage_gate_min}). "
                "Skipping DE research."
            )
            result.stage_reached = "triage_rejected"
            self._storage.save(result)
            return result

        # ── Stage 2: DE Research (sequential — each agent feeds the next) ─────
        result.stage_reached = "research"

        # Customer
        self._progress("Running Customer agent...")
        try:
            result.customer = self._customer_agent.run(idea, triage)
        except Exception:
            result.agent_errors["customer"] = traceback.format_exc()
            self._progress("Customer agent failed — continuing with partial results.")

        customer = result.customer

        # Value Prop (needs customer context)
        self._progress("Running Value Prop agent...")
        try:
            if customer:
                result.value_prop = self._value_prop_agent.run(idea, triage, customer)
            else:
                result.agent_errors["value_prop"] = "Skipped: customer agent failed."
        except Exception:
            result.agent_errors["value_prop"] = traceback.format_exc()
            self._progress("Value Prop agent failed — continuing.")

        value_prop = result.value_prop

        # Acquisition (needs customer + value prop)
        self._progress("Running Acquisition agent...")
        try:
            if customer and value_prop:
                result.acquisition = self._acquisition_agent.run(idea, triage, customer, value_prop)
            else:
                result.agent_errors["acquisition"] = "Skipped: upstream agent failed."
        except Exception:
            result.agent_errors["acquisition"] = traceback.format_exc()
            self._progress("Acquisition agent failed — continuing.")

        acquisition = result.acquisition

        # Economics (needs value prop + acquisition)
        self._progress("Running Economics agent...")
        try:
            if value_prop and acquisition:
                result.economics = self._economics_agent.run(idea, triage, value_prop, acquisition)
            else:
                result.agent_errors["economics"] = "Skipped: upstream agent failed."
        except Exception:
            result.agent_errors["economics"] = traceback.format_exc()
            self._progress("Economics agent failed — continuing.")

        economics = result.economics

        # Feasibility (needs customer + value prop)
        self._progress("Running Feasibility agent...")
        try:
            if customer and value_prop:
                result.feasibility = self._feasibility_agent.run(idea, triage, customer, value_prop)
            else:
                result.agent_errors["feasibility"] = "Skipped: upstream agent failed."
        except Exception:
            result.agent_errors["feasibility"] = traceback.format_exc()
            self._progress("Feasibility agent failed — continuing.")

        feasibility = result.feasibility

        # Scale (needs all previous)
        self._progress("Running Scale & Moat agent...")
        try:
            if customer and value_prop and economics and feasibility:
                result.scale = self._scale_agent.run(
                    idea, triage, customer, value_prop, economics, feasibility
                )
            else:
                result.agent_errors["scale"] = "Skipped: upstream agent failed."
        except Exception:
            result.agent_errors["scale"] = traceback.format_exc()
            self._progress("Scale agent failed — continuing.")

        # ── Stage 3: Scoring ──────────────────────────────────────────────────
        self._progress("Computing composite scores...")
        if (
            result.triage
            and result.customer
            and result.value_prop
            and result.acquisition
            and result.economics
            and result.feasibility
            and result.scale
        ):
            try:
                result.scores = compute_scores(
                    triage=result.triage,
                    customer=result.customer,
                    value_prop=result.value_prop,
                    acquisition=result.acquisition,
                    economics=result.economics,
                    feasibility=result.feasibility,
                    scale=result.scale,
                    idea_track=idea.track.value,
                )
                result.stage_reached = "scored"
                self._progress(
                    f"Score A (internal): {result.scores.composite_score_a} → {result.scores.decision_a} | "
                    f"Score B (external): {result.scores.composite_score_b} → {result.scores.decision_b}"
                )
            except Exception:
                result.agent_errors["scoring"] = traceback.format_exc()
                self._progress("Scoring failed.")
        else:
            self._progress(
                "Scoring skipped — too many agent failures. "
                f"Errors: {list(result.agent_errors.keys())}"
            )

        self._storage.save(result)
        return result
