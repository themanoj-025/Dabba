"""Page 1: Discover — restaurant recommendations with styled cards,
LLM narration, "Find Similar" button, and prioritize toggle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.components.restaurant_card import render_restaurant_card
from dabba.config import get_config
from dabba.llm.recommendation_narrator import narrate_recommendation
from dabba.llm.rag_similar_restaurants import (
    build_restaurant_embeddings,
    find_similar_restaurants,
)
from dabba.models.hybrid_recommender import HybridRecommender

PAGE_NAME = "discover"
config = get_config()


def show() -> None:
    """Render the Discover page."""
    st.title("🍽️ Discover")
    st.markdown("Find your next great meal. Tell us what you're craving.")

    # ─── Sidebar filters ─────────────────────────────────────────
    with st.sidebar:
        st.header("Your Preferences")

        cuisine_options = [
            "North Indian", "Chinese", "South Indian", "Mughlai", "Cafe",
            "Bakery", "Italian", "Fast Food", "Continental", "Desserts",
            "Biryani", "Street Food", "Ice Cream", "Andhra", "Thai",
        ]
        selected_cuisines = st.multiselect(
            "Cuisine", cuisine_options, key=f"{PAGE_NAME}_cuisine"
        )

        budget = st.slider(
            "Budget (₹ for two)", 100, 5000, 1000, 100,
            key=f"{PAGE_NAME}_budget",
        )

        area_options = [
            "All", "Koramangala", "Indiranagar", "HSR Layout", "Whitefield",
            "Electronic City", "JP Nagar", "BTM Layout", "Marathahalli",
            "Jayanagar", "MG Road", "Sarjapur Road",
        ]
        selected_area = st.selectbox(
            "Area", area_options, key=f"{PAGE_NAME}_area"
        )

        prioritize = st.select_slider(
            "Prioritize",
            options=["speed", "balanced", "quality"],
            value="balanced",
            key=f"{PAGE_NAME}_prioritize",
            help="Speed = delivery reliability first. Quality = rating/sentiment first.",
        )

        top_n = st.slider(
            "Results", 3, 20, 5, key=f"{PAGE_NAME}_topn"
        )

        st.markdown("---")
        use_llm = st.checkbox(
            "💬 LLM Explanations",
            value=True,
            help="Generate natural-language explanations (requires API key)",
            key=f"{PAGE_NAME}_llm",
        )

    # ─── Load data ────────────────────────────────────────────────
    df = _load_data()
    if df is None:
        st.warning("⚠️ Processed data not found. Run `make train` first.")
        return

    # ─── Build recommender ────────────────────────────────────────
    feature_cols = [c for c in df.columns if c.startswith("cuisine_")]
    feature_cols += [c for c in [
        "votes_log", "cost_for_two", "online_order_binary",
        "book_table_binary", "cuisine_count", "avg_sentiment",
    ] if c in df.columns]

    recommender = HybridRecommender(
        df, feature_cols, collaborative_model=None, config=config
    )

    embeddings = build_restaurant_embeddings(df, feature_cols, config=config)

    # ─── Get recommendations ──────────────────────────────────────
    cuisine_str = "|".join(selected_cuisines) if selected_cuisines else None
    area_str = selected_area if selected_area and selected_area != "All" else None

    with st.spinner("Finding the best restaurants for you..."):
        results = recommender.recommend(
            cuisine=cuisine_str,
            budget=budget,
            area=area_str,
            prioritize=prioritize,
            top_n=top_n,
        )

    if results.empty:
        st.info("No restaurants match your filters. Try adjusting your preferences.")
        return

    # ─── Handle similar-restaurant lookup via session state ────────
    state_key = f"{PAGE_NAME}_similar_result"
    name_key = f"{PAGE_NAME}_similar_name"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
        st.session_state[name_key] = None

    # Clear similar results when filters change (cuisine/budget/area change)
    # by checking if this is a rerun after a filter change
    filter_key = f"{PAGE_NAME}_filter_hash"
    current_hash = hash((tuple(selected_cuisines or []), budget, selected_area, prioritize))
    if st.session_state.get(filter_key) != current_hash:
        st.session_state[state_key] = None
        st.session_state[name_key] = None
        st.session_state[filter_key] = current_hash

    def _on_find_similar(restaurant_name: str):
        matches = df[df["name"].str.contains(restaurant_name, case=False, na=False)]
        if not matches.empty:
            idx = df.index.get_loc(matches.index[0])
            sim = find_similar_restaurants(idx, df, embeddings, top_k=5, config=config)
            st.session_state[state_key] = sim
            st.session_state[name_key] = restaurant_name
        else:
            st.session_state[name_key] = None

    # ─── Display results ──────────────────────────────────────────
    st.subheader(f"Top {len(results)} Recommendations")

    for i, (_, row) in enumerate(results.iterrows()):
        rest_dict = row.to_dict()

        explanation = None
        if use_llm:
            rs = rest_dict.get("reliability_score_display", 0.5)
            sentiment = rest_dict.get("avg_sentiment", 0.0)
            explanation = narrate_recommendation(
                rest_dict, rs, sentiment, config=config,
            )

        # Render card with similar button
        render_restaurant_card(
            rest_dict,
            explanation=explanation,
            show_similar_button=True,
            similar_callback=_on_find_similar,
            key_prefix=f"{PAGE_NAME}_{i}",
        )

    # ─── Similar restaurants section ──────────────────────────────
    similar_df = st.session_state.get(f"{PAGE_NAME}_similar_result")
    similar_name = st.session_state.get(f"{PAGE_NAME}_similar_name")

    if similar_df is not None and not similar_df.empty and similar_name:
        st.markdown("---")
        st.subheader(f"🔍 Similar to **{similar_name}**")
        for j, (_, srow) in enumerate(similar_df.iterrows()):
            render_restaurant_card(
                srow.to_dict(),
                key_prefix=f"sim_{j}",
            )

    # Priority explanation
    if prioritize != "balanced":
        st.markdown("---")
        st.caption(
            f"📊 Showing **{prioritize}-first** rankings. "
            "Change the priority slider above to see different weight profiles."
        )


@st.cache_data
def _load_data() -> Optional[pd.DataFrame]:
    """Load processed restaurant data with caching."""
    data_path = Path("data/processed/restaurants_processed.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None
