"""Streamlit dashboard — main entry point.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(
    page_title="Dabba — Restaurant Intelligence Platform",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/curry.png", width=80)
    st.title("🍛 Dabba")
    st.markdown(
        """
        **India-focused restaurant ranking, recommendation,
        and delivery-reliability platform.**

        Built for ML portfolio / job-application standard.

        ---
        **Author:** [Your Name](https://linkedin.com/in/yourname)
        **GitHub:** [github.com/yourname/dabba](https://github.com/yourname/dabba)
        """
    )

# --- Main page ---
st.title("🍛 Dabba — Restaurant Intelligence Platform")

st.markdown(
    """
    Welcome to **Dabba**, an India-focused restaurant ranking and
    delivery-reliability platform built with rigorous ML experimentation.

    ### Navigate to a page:
    - 🍽️ **Customer View** — Get personalized restaurant recommendations
    - 🚀 **Operations View** — Monitor delivery SLAs and run simulations
    - 📊 **Model Info** — See which algorithms won and why

    ### The Reliability Score
    ```
    reliability_score = 0.4 × norm(rating) + 0.3 × norm(sentiment) - 0.3 × norm(delay_risk)
    ```
    A composite score combining restaurant quality, customer sentiment,
    and delivery reliability — powered by the winning ML models.
    """
)

st.info("👈 Use the sidebar to navigate to a specific page.")
