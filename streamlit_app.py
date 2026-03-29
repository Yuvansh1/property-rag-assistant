import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Property RAG Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme base */
    .stApp {background-color: #0e1117; color: #e0e0e0;}

    /* Answer card */
    .answer-card {
        background: #1e2130;
        border: 1px solid #2d3249;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
        font-size: 1rem;
        line-height: 1.6;
        color: #e0e0e0;
    }

    /* Metric card */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3249;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }
    .badge-grounded    {background: #064e3b; color: #6ee7b7;}
    .badge-partial     {background: #78350f; color: #fcd34d;}
    .badge-ungrounded  {background: #7f1d1d; color: #fca5a5;}
    .badge-flagged     {background: #7c3aed; color: #ede9fe;}

    /* Confidence label */
    .conf-green  {color: #34d399; font-weight: 700;}
    .conf-yellow {color: #fbbf24; font-weight: 700;}
    .conf-red    {color: #f87171; font-weight: 700;}

    /* Reasoning box */
    .reasoning-box {
        background: #161b2e;
        border-left: 3px solid #4b5563;
        padding: 0.6rem 1rem;
        margin-top: 0.6rem;
        border-radius: 0 6px 6px 0;
        font-style: italic;
        color: #9ca3af;
        font-size: 0.88rem;
    }

    /* Recommendation box */
    .recommendation-box {
        background: #1e2a45;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #93c5fd;
        font-size: 0.95rem;
    }

    /* Flagged query item */
    .flagged-item {
        background: #1e2130;
        border: 1px solid #2d3249;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #d1d5db;
    }

    /* Progress bar track */
    .bar-track {
        background: #2d3249;
        border-radius: 999px;
        height: 10px;
        margin-top: 4px;
        overflow: hidden;
    }
    .bar-fill {
        height: 10px;
        border-radius: 999px;
        background: #7c3aed;
    }

    /* Sidebar buttons */
    div[data-testid="stSidebar"] button {
        background: #1e2130;
        color: #c4b5fd;
        border: 1px solid #3b3f5c;
        border-radius: 8px;
        text-align: left;
        width: 100%;
        margin-bottom: 4px;
        font-size: 0.82rem;
    }
    div[data-testid="stSidebar"] button:hover {
        background: #2d3249;
        border-color: #7c3aed;
    }

    /* Feedback message */
    .feedback-thanks {
        color: #34d399;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* P95 highlight card */
    .p95-card {
        background: #1e2130;
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .p95-card .metric-value {color: #c4b5fd; font-size: 2rem; font-weight: 700;}
    .p95-card .metric-label {font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem;}

    /* Feedback summary card */
    .feedback-card {
        background: #1e2130;
        border: 1px solid #2d3249;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "What are closing costs?",
    "How does a fixed-rate mortgage work?",
    "What is a home appraisal?",
    "What does escrow mean in real estate?",
    "What is title insurance?",
    "How do I calculate property taxes?",
    "What is a home inspection?",
    "What are HOA fees?",
]

# Sidebar
with st.sidebar:
    st.markdown("### Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}"):
            st.session_state["ask_input"] = q

# Tabs
tab_ask, tab_monitor, tab_metrics = st.tabs(["ASK", "MONITOR", "METRICS"])

# ── ASK TAB ──────────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown("## Property Knowledge Assistant")

    default_val = st.session_state.get("ask_input", "")
    question = st.text_input(
        "Ask a real estate question",
        value=default_val,
        placeholder="e.g. What are closing costs?",
        key="ask_input",
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving answer..."):
            try:
                resp = requests.get(f"{API_BASE}/ask", params={"q": question}, timeout=30)
                resp.raise_for_status()
                st.session_state["last_answer"] = resp.json()
                st.session_state["feedback_given"] = False
            except Exception as e:
                st.error(f"API error: {e}")
                st.session_state.pop("last_answer", None)

    if st.session_state.get("last_answer"):
        data = st.session_state["last_answer"]
        answer = data.get("response", "")
        meta = data.get("meta", {})
        score = meta.get("confidence_score") or 0.0
        grounding = meta.get("grounding_status", "unknown")
        flagged = meta.get("flagged", False)
        reasoning = meta.get("critic_reasoning", "")
        query_id = meta.get("query_id")

        # Confidence color
        if score >= 0.75:
            conf_class = "conf-green"
        elif score >= 0.5:
            conf_class = "conf-yellow"
        else:
            conf_class = "conf-red"

        # Grounding badge
        grounding_class = {
            "grounded": "badge-grounded",
            "partially_grounded": "badge-partial",
            "ungrounded": "badge-ungrounded",
        }.get(grounding, "badge-partial")
        grounding_label = grounding.replace("_", " ").title()

        flagged_html = (
            '<span class="badge badge-flagged">Flagged</span>'
            if flagged else ""
        )

        st.markdown(f"""
<div class="answer-card">{answer}</div>
<div style="margin-top:0.75rem; display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
    <span style="color:#9ca3af; font-size:0.85rem;">Confidence:</span>
    <span class="{conf_class}">{score * 100:.0f}%</span>
    <span class="badge {grounding_class}">{grounding_label}</span>
    {flagged_html}
</div>
""", unsafe_allow_html=True)

        if reasoning:
            st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

        # Feedback buttons
        st.markdown("<div style='margin-top:0.9rem; color:#9ca3af; font-size:0.85rem;'>Was this answer helpful?</div>",
                    unsafe_allow_html=True)

        if not st.session_state.get("feedback_given", False):
            col_up, col_down, _ = st.columns([1, 1, 8])
            with col_up:
                if st.button("Thumbs Up", key="thumb_up") and query_id is not None:
                    try:
                        requests.post(
                            f"{API_BASE}/feedback",
                            json={"query_id": query_id, "rating": "up"},
                            timeout=10,
                        )
                    except Exception:
                        pass
                    st.session_state["feedback_given"] = True
                    st.rerun()
            with col_down:
                if st.button("Thumbs Down", key="thumb_down") and query_id is not None:
                    try:
                        requests.post(
                            f"{API_BASE}/feedback",
                            json={"query_id": query_id, "rating": "down"},
                            timeout=10,
                        )
                    except Exception:
                        pass
                    st.session_state["feedback_given"] = True
                    st.rerun()
        else:
            st.markdown('<div class="feedback-thanks">Thanks for your feedback!</div>', unsafe_allow_html=True)


# ── MONITOR TAB ───────────────────────────────────────────────────────────────
with tab_monitor:
    st.markdown("## Proactive Monitoring Report")

    if st.button("Refresh", key="monitor_refresh"):
        st.session_state.pop("monitor_data", None)

    if "monitor_data" not in st.session_state:
        with st.spinner("Loading monitor report..."):
            try:
                resp = requests.get(f"{API_BASE}/monitor", timeout=15)
                resp.raise_for_status()
                st.session_state["monitor_data"] = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.session_state["monitor_data"] = None

    monitor = st.session_state.get("monitor_data")

    if monitor:
        summary = monitor.get("summary", {})
        total = summary.get("total_queries", 0)
        flagged_count = summary.get("flagged_queries", 0)
        flag_rate = summary.get("flag_rate", 0.0)
        avg_conf = summary.get("average_confidence_score", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in [
            (c1, "Total Queries", str(total)),
            (c2, "Flagged Queries", str(flagged_count)),
            (c3, "Flag Rate", f"{flag_rate * 100:.1f}%"),
            (c4, "Avg Confidence", f"{avg_conf * 100:.1f}%"),
        ]:
            with col:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        recommendation = monitor.get("recommendation", "")
        if recommendation:
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        grounding_dist = monitor.get("grounding_distribution", {})
        if grounding_dist:
            st.markdown("#### Grounding Distribution")
            total_g = sum(grounding_dist.values()) or 1
            for status, count in grounding_dist.items():
                pct = count / total_g * 100
                label = status.replace("_", " ").title()
                st.markdown(f"""
<div style="margin-bottom:0.6rem;">
    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#d1d5db;">
        <span>{label}</span><span>{count} ({pct:.0f}%)</span>
    </div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        flagged_queries = monitor.get("recent_flagged_queries", [])
        if flagged_queries:
            st.markdown("#### Recent Flagged Queries")
            for item in flagged_queries:
                q_text = item.get("query", "")
                q_score = item.get("confidence_score", 0.0)
                q_time = item.get("timestamp", "")
                st.markdown(f"""
<div class="flagged-item">
    <strong>{q_text}</strong>
    <span style="color:#fca5a5; margin-left:0.5rem;">score: {q_score:.2f}</span>
    <span style="color:#6b7280; margin-left:0.5rem; font-size:0.78rem;">{q_time}</span>
</div>
""", unsafe_allow_html=True)
        elif total == 0:
            st.info("No queries recorded yet. Use the ASK tab to get started.")


# ── METRICS TAB ───────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("## Latency and Performance Metrics")

    if st.button("Refresh", key="metrics_refresh"):
        st.session_state.pop("metrics_data", None)
        st.session_state.pop("feedback_summary_data", None)

    if "metrics_data" not in st.session_state:
        with st.spinner("Loading metrics..."):
            try:
                resp = requests.get(f"{API_BASE}/metrics", timeout=15)
                resp.raise_for_status()
                st.session_state["metrics_data"] = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.session_state["metrics_data"] = None

    if "feedback_summary_data" not in st.session_state:
        try:
            resp = requests.get(f"{API_BASE}/feedback/summary", timeout=15)
            resp.raise_for_status()
            st.session_state["feedback_summary_data"] = resp.json()
        except Exception:
            st.session_state["feedback_summary_data"] = None

    metrics = st.session_state.get("metrics_data")
    feedback_summary = st.session_state.get("feedback_summary_data")

    if metrics:
        st.markdown("#### Per-Step Latency (avg ms)")
        c1, c2, c3, c4 = st.columns(4)
        latency_cards = [
            (c1, "Embed", metrics.get("avg_embed_latency_ms", 0.0)),
            (c2, "Retrieve", metrics.get("avg_retrieve_latency_ms", 0.0)),
            (c3, "Generate", metrics.get("avg_generate_latency_ms", 0.0)),
            (c4, "Critic", metrics.get("avg_critic_latency_ms", 0.0)),
        ]
        for col, label, val in latency_cards:
            with col:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{val:.0f}</div>
    <div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # P95 and overall stats
        col_p95, col_total, col_flag, col_conf = st.columns(4)
        with col_p95:
            p95 = metrics.get("p95_total_latency_ms", 0.0)
            st.markdown(f"""
<div class="p95-card">
    <div class="metric-value">{p95:.0f}</div>
    <div class="metric-label">P95 Total Latency (ms)</div>
</div>
""", unsafe_allow_html=True)
        with col_total:
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{metrics.get("total_queries", 0)}</div>
    <div class="metric-label">Total Queries</div>
</div>
""", unsafe_allow_html=True)
        with col_flag:
            flag_pct = metrics.get("flag_rate", 0.0) * 100
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{flag_pct:.1f}%</div>
    <div class="metric-label">Flag Rate</div>
</div>
""", unsafe_allow_html=True)
        with col_conf:
            conf_pct = metrics.get("avg_confidence_score", 0.0) * 100
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{conf_pct:.1f}%</div>
    <div class="metric-label">Avg Confidence</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if feedback_summary:
        st.markdown("#### Human Feedback Summary")
        total_fb = feedback_summary.get("total_feedback", 0)
        thumbs_up = feedback_summary.get("thumbs_up", 0)
        thumbs_down = feedback_summary.get("thumbs_down", 0)
        agreement = feedback_summary.get("agreement_rate", 0.0)
        disagreement = feedback_summary.get("disagreement_rate", 0.0)

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        for col, label, val in [
            (fc1, "Total Feedback", str(total_fb)),
            (fc2, "Thumbs Up", str(thumbs_up)),
            (fc3, "Thumbs Down", str(thumbs_down)),
            (fc4, "Agreement Rate", f"{agreement * 100:.1f}%"),
            (fc5, "Disagreement Rate", f"{disagreement * 100:.1f}%"),
        ]:
            with col:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{val}</div>
    <div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)

    if not metrics and not feedback_summary:
        st.info("No data yet. Use the ASK tab to generate queries, then refresh.")
