"""Dabba v3 — Streamlit dashboard entry point.

4-page app with warm food-tech design system, Plotly charts,
styled components, and LLM-powered features with graceful fallback.

Uses custom radio navigation (not Streamlit multi-page auto-discovery)
for consistent theming and layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from dabba.config import get_config

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Dabba — Restaurant Intelligence Platform",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load theme CSS ──────────────────────────────────────────────────

css_path = Path(__file__).parent / "assets" / "theme.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
            <span style="font-size:2.2rem;">🍛</span>
            <span style="font-size:1.5rem;font-weight:700;color:#2d2d2d;">Dabba</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="font-size:0.85rem;color:#6b7280;margin-bottom:1.5rem;">
        Restaurant ranking, recommendation, and delivery-reliability platform.
        Powered by rigorous ML + LLM.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🍽️ Discover", "🚀 Ops Monitor", "📊 Model Performance", "💬 Food Concierge"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    config = get_config()
    llm_status = "🟢 LLM Active" if config.llm_enabled and config.anthropic_api_key else "🟡 Rules-based"
    st.markdown(
        f'<p style="font-size:0.75rem;color:#9ca3af;">{llm_status} mode</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="font-size:0.75rem;color:#9ca3af;margin-top:0.5rem;">
        <b>Author:</b> <a href="https://linkedin.com/in/yourname" style="color:#ff8c42;">Your Name</a><br>
        <b>GitHub:</b> <a href="https://github.com/yourname/dabba" style="color:#ff8c42;">github.com/yourname/dabba</a>
        </p>
        """,
        unsafe_allow_html=True,
    )

# ─── Page routing ────────────────────────────────────────────────────

from pages import page_discover as discover  # noqa: E402
from pages import page_ops as ops  # noqa: E402
from pages import page_model_performance as model_perf  # noqa: E402
from pages import page_concierge as concierge  # noqa: E402

if "🍽️" in page:
    discover.show()
elif "🚀" in page:
    ops.show()
elif "📊" in page:
    model_perf.show()
elif "💬" in page:
    concierge.show()
else:
    st.title("🍛 Dabba — Restaurant Intelligence Platform")
    st.markdown(
        """
        Welcome to **Dabba**, an India-focused restaurant ranking and
        delivery-reliability platform built with rigorous ML experimentation.

        ### Navigate to a page:
        - 🍽️ **Discover** — Get personalized restaurant recommendations
        - 🚀 **Ops Monitor** — Monitor delivery SLAs and run simulations
        - 📊 **Model Performance** — See which algorithms won and why
        - 💬 **Food Concierge** — Chat with our AI copilot

        ### The Reliability Score
        ```
        reliability_score = 0.4 × norm(rating) + 0.3 × norm(sentiment)
                              - 0.3 × norm(delay_risk)
        ```
        A composite score combining restaurant quality, customer sentiment,
        and delivery reliability — powered by the winning ML models.
        """
    )
    st.info("👈 Use the sidebar to navigate to a specific page.")
