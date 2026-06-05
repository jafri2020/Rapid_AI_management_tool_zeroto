"""Streamlit UI for the Lazybones Idea Validation Engine."""
from __future__ import annotations

import os
import threading
import time
from queue import Queue, Empty

import streamlit as st

from config import config
from models import IdeaInput, IdeaTrack, ValidationResult
from pipeline import Pipeline
from report import generate as generate_report
from storage import Storage

st.set_page_config(
    page_title="Lazybones Idea Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared storage ────────────────────────────────────────────────────────────
@st.cache_resource
def get_storage() -> Storage:
    return Storage(config.db_path)


# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    provider = st.selectbox("LLM Provider", ["kimi", "ollama"], index=0)
    config.llm_provider = provider

    if provider == "kimi":
        api_key = st.text_input(
            "Kimi API Key",
            value=os.getenv("KIMI_API_KEY", ""),
            type="password",
            help="Get yours at platform.moonshot.cn. Or set KIMI_API_KEY env var.",
        )
        if api_key:
            config.kimi_api_key = api_key
        config.model = st.selectbox(
            "Model",
            ["kimi-k2.6", "kimi-k2.5"],
            index=0,
            help="32k is the best balance. Use 128k for very long idea descriptions.",
        )
    else:  # ollama
        config.ollama_base_url = st.text_input("Ollama URL", value=config.ollama_base_url)
        config.ollama_model = st.text_input(
            "Ollama Model",
            value=config.ollama_model,
            help="e.g. llama3.1:8b, qwen2.5:14b, mistral:7b",
        )

    st.divider()
    config.enable_search = st.toggle("Web Search", value=config.enable_search)
    config.enable_critique = st.toggle("Critique Step", value=config.enable_critique)
    st.divider()
    st.caption(f"DB: `{config.db_path}`")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_submit, tab_ideas, tab_detail, tab_compare = st.tabs(
    ["🆕 Submit Idea", "📋 All Ideas", "🔍 Idea Detail", "📊 Compare"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Submit Idea
# ─────────────────────────────────────────────────────────────────────────────
with tab_submit:
    st.header("Submit an Idea for Validation")
    st.caption(
        "The engine runs Z2O triage first, then 6 DE research agents, then scores. "
        "Takes ~5-10 minutes depending on search and model speed."
    )

    with st.form("idea_form"):
        title = st.text_input("Idea Title *", placeholder="AI Proposal Assistant for Sales")
        description = st.text_area(
            "Description *",
            height=120,
            placeholder=(
                "Describe the idea, the problem it solves, who has the problem, "
                "and any context you have. The more detail, the better the research."
            ),
        )
        col1, col2 = st.columns(2)
        with col1:
            submitter = st.text_input("Submitter", placeholder="Hussain / Taha / Hasnain")
        with col2:
            track = st.selectbox(
                "Track",
                options=[t.value for t in IdeaTrack],
                format_func=lambda v: {"internal": "Internal Xavor Only", "external": "External Product Only", "both": "Both Tracks"}.get(v, v),
            )
        source_context = st.text_input(
            "Source Context",
            placeholder="Where did this come from? (leadership ask, client complaint, team observation...)",
        )
        submitted = st.form_submit_button("🚀 Run Validation", type="primary", use_container_width=True)

    if submitted:
        if not title or not description:
            st.error("Title and description are required.")
        elif config.llm_provider == "kimi" and not config.kimi_api_key:
            st.error("Kimi API key is required. Add it in the sidebar or set KIMI_API_KEY in your .env file.")
        else:
            idea = IdeaInput(
                title=title,
                description=description,
                submitter=submitter or "unknown",
                track=IdeaTrack(track),
                source_context=source_context,
            )

            # Progress display
            log_container = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            log_lines: list[str] = []
            progress_queue: Queue = Queue()

            def on_progress(msg: str):
                progress_queue.put(msg)

            def run_pipeline():
                storage = get_storage()
                p = Pipeline(storage=storage, on_progress=on_progress)
                result = p.run(idea)
                progress_queue.put(("__done__", result))

            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()

            stage_progress = {
                "triage": 0.15,
                "customer": 0.30,
                "value prop": 0.45,
                "acquisition": 0.55,
                "economics": 0.65,
                "feasibility": 0.75,
                "scale": 0.85,
                "scoring": 0.95,
            }

            result: ValidationResult | None = None
            while thread.is_alive() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.2)
                    if isinstance(msg, tuple) and msg[0] == "__done__":
                        result = msg[1]
                        progress_bar.progress(1.0)
                        break
                    log_lines.append(f"• {msg}")
                    log_container.code("\n".join(log_lines[-12:]))
                    for keyword, pct in stage_progress.items():
                        if keyword in msg.lower():
                            progress_bar.progress(pct)
                            status_text.caption(f"Running: {msg}")
                            break
                except Empty:
                    time.sleep(0.1)

            if result:
                if result.agent_errors:
                    st.error(f"Pipeline finished with {len(result.agent_errors)} agent error(s). Full tracebacks below.")
                    for agent_name, err in result.agent_errors.items():
                        with st.expander(f"❌ {agent_name} — click for full traceback", expanded=True):
                            st.code(err, language="python")
                else:
                    st.success("Validation complete!")

                memo = generate_report(result)
                st.code(memo, language=None)

                if result.scores:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Score A (Internal)", f"{result.scores.composite_score_a}/100")
                    col2.metric("Score B (External)", f"{result.scores.composite_score_b}/100")
                    col3.metric("Decision A", result.scores.decision_a.replace("_", " ").title())
                    col4.metric("Decision B", result.scores.decision_b.replace("_", " ").title())

                    if result.scores.quick_win_flag:
                        st.success("⚡ Quick Win flagged — MVP ≤ 7 days and score ≥ 65")
                    if result.scores.strategic_bet_flag:
                        st.info("🎯 Strategic Bet flagged — exceptional secret + timing")

                st.download_button(
                    "⬇️ Download Memo",
                    data=memo,
                    file_name=f"memo_{result.idea.title[:30].replace(' ', '_')}.txt",
                    mime="text/plain",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — All Ideas Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_ideas:
    st.header("Ideas Leaderboard")

    storage = get_storage()
    rows = storage.list_all()

    if not rows:
        st.info("No ideas validated yet. Submit one in the first tab.")
    else:
        col_filter, col_track, col_sort = st.columns(3)
        with col_filter:
            stage_filter = st.multiselect(
                "Stage",
                options=["scored", "triage", "triage_rejected", "research", "capture"],
                default=["scored", "triage_rejected"],
            )
        with col_track:
            track_filter = st.multiselect(
                "Track",
                options=["both", "internal", "external"],
                default=["both", "internal", "external"],
            )
        with col_sort:
            sort_by = st.selectbox("Sort by", ["Score A", "Score B", "Triage Score", "Date"])

        filtered = [
            r for r in rows
            if (not stage_filter or r["stage"] in stage_filter)
            and (not track_filter or r["track"] in track_filter)
        ]

        sort_key = {
            "Score A": lambda r: r["score_a"] or 0,
            "Score B": lambda r: r["score_b"] or 0,
            "Triage Score": lambda r: r["triage_score"] or 0,
            "Date": lambda r: r["created_at"] or "",
        }[sort_by]
        filtered.sort(key=sort_key, reverse=True)

        for r in filtered:
            decision_a = r.get("decision_a") or "—"
            decision_b = r.get("decision_b") or "—"
            flags = []
            if r.get("quick_win"):
                flags.append("⚡")
            if r.get("strategic_bet"):
                flags.append("🎯")
            flag_str = " ".join(flags)

            score_a_str = f"{r['score_a']}" if r["score_a"] is not None else "—"
            score_b_str = f"{r['score_b']}" if r["score_b"] is not None else "—"

            with st.expander(
                f"{flag_str} **{r['title']}** | A:{score_a_str} B:{score_b_str} | "
                f"Triage:{r['triage_score'] or '—'}/20 | {r['track'].upper()} | {r['stage']}"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Score A", score_a_str)
                col2.metric("Score B", score_b_str)
                col3.metric("Triage", f"{r['triage_score'] or '—'}/20")

                col4, col5 = st.columns(2)
                col4.caption(f"Decision A: **{decision_a}**")
                col5.caption(f"Decision B: **{decision_b}**")

                if st.button("View Full Memo", key=f"view_{r['id']}"):
                    st.session_state["selected_idea_id"] = r["id"]
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Idea Detail
# ─────────────────────────────────────────────────────────────────────────────
with tab_detail:
    st.header("Idea Detail")

    storage = get_storage()
    rows = storage.list_all()

    if not rows:
        st.info("No ideas yet.")
    else:
        options = {r["id"]: f"{r['title']} (A:{r['score_a'] or '—'} B:{r['score_b'] or '—'})" for r in rows}
        default_id = st.session_state.get("selected_idea_id", list(options.keys())[0])
        selected_id = st.selectbox(
            "Select Idea",
            options=list(options.keys()),
            format_func=lambda k: options[k],
            index=list(options.keys()).index(default_id) if default_id in options else 0,
        )

        if selected_id:
            result = storage.load(selected_id)
            if result:
                memo = generate_report(result)
                st.code(memo, language=None)

                # Dimension chart
                if result.scores and result.scores.dimension_breakdown:
                    st.subheader("Dimension Scores")
                    import pandas as pd
                    df = pd.DataFrame([
                        {
                            "Dimension": d.name,
                            "Score": d.score,
                            "Z2O Component": d.z2o_component,
                            "DE Component": d.de_component,
                            "Weight": f"{int(d.weight * 100)}%",
                        }
                        for d in result.scores.dimension_breakdown
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.bar_chart(df.set_index("Dimension")["Score"])

                # Raw JSON download
                st.download_button(
                    "⬇️ Download Full JSON",
                    data=result.model_dump_json(indent=2),
                    file_name=f"idea_{selected_id}.json",
                    mime="application/json",
                )

                # Delete
                if st.button("🗑️ Delete this idea", type="secondary"):
                    storage.delete(selected_id)
                    st.success("Deleted.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Compare
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.header("Compare Ideas")

    storage = get_storage()
    rows = storage.list_all()
    scored = [r for r in rows if r["score_a"] is not None or r["score_b"] is not None]

    if len(scored) < 2:
        st.info("Need at least 2 scored ideas to compare.")
    else:
        import pandas as pd

        df = pd.DataFrame(scored)
        df["score_a"] = df["score_a"].fillna(0)
        df["score_b"] = df["score_b"].fillna(0)
        df["triage_score"] = df["triage_score"].fillna(0)
        df["quick_win"] = df["quick_win"].astype(bool)
        df["strategic_bet"] = df["strategic_bet"].astype(bool)

        display_cols = ["title", "track", "triage_score", "score_a", "decision_a", "score_b", "decision_b", "quick_win", "strategic_bet"]
        st.dataframe(
            df[display_cols].sort_values("score_a", ascending=False).rename(columns={
                "title": "Idea",
                "track": "Track",
                "triage_score": "Triage /20",
                "score_a": "Score A",
                "decision_a": "Decision A",
                "score_b": "Score B",
                "decision_b": "Decision B",
                "quick_win": "⚡ Quick Win",
                "strategic_bet": "🎯 Strat Bet",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Score A vs Score B")
        chart_df = df[["title", "score_a", "score_b"]].set_index("title")
        st.bar_chart(chart_df)
