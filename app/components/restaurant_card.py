"""Styled restaurant card component for the Discover page.

Renders a restaurant with name, cuisine tags, rating badge, sentiment
badge, ETA chip, reliability score bar, and LLM explanation caption.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def render_restaurant_card(
    restaurant: Dict[str, Any],
    explanation: Optional[str] = None,
    show_similar_button: bool = False,
    similar_callback=None,
    key_prefix: str = "",
) -> None:
    """Render a styled restaurant card.

    Args:
        restaurant: Dict with keys: name, rate, cost_for_two, location,
            cuisines, reliability_score_display, etc.
        explanation: Optional natural-language explanation (LLM or rules-based).
        show_similar_button: Whether to show "Find similar" button.
        similar_callback: Callback for similar button (takes restaurant index).
        key_prefix: Unique key prefix for Streamlit components.
    """
    name = restaurant.get("name", "Unknown")
    rating = restaurant.get("rate", "N/A")
    cost = restaurant.get("cost_for_two", "N/A")
    location = restaurant.get("location", "N/A")
    cuisines = restaurant.get("cuisines", "N/A")
    reliability = restaurant.get(
        "reliability_score_display", restaurant.get("reliability_score", 0.5)
    )
    combined_score = restaurant.get("combined_score", None)
    cf_score = restaurant.get("cf_score", None)

    with st.container():
        st.markdown('<div class="restaurant-card">', unsafe_allow_html=True)

        # Row 1: Name + Scores
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(f'<p class="name">{name}</p>', unsafe_allow_html=True)

        with cols[1]:
            try:
                r = float(rating) if rating != "N/A" else 0
                st.markdown(
                    f'<span class="badge badge-rating">⭐ {r:.1f}/5</span>',
                    unsafe_allow_html=True,
                )
            except (ValueError, TypeError):
                st.markdown(
                    '<span class="badge badge-rating">⭐ N/A</span>',
                    unsafe_allow_html=True,
                )

        with cols[2]:
            try:
                rs = float(reliability)
                emoji = "🟢" if rs >= 0.7 else "🟡" if rs >= 0.4 else "🔴"
                st.markdown(
                    f'<span class="badge badge-reliability">{emoji} {rs:.2f}</span>',
                    unsafe_allow_html=True,
                )
            except (ValueError, TypeError):
                st.markdown(
                    '<span class="badge badge-reliability">⚪ N/A</span>',
                    unsafe_allow_html=True,
                )

        # Row 2: Details
        st.markdown(
            f'<p class="details">📍 {location} | 🍳 {cuisines[:60]} | '
            f'<span class="badge badge-cost">₹{int(cost) if cost != "N/A" else "?"}</span>'
            f'{" | CF: " + f"{cf_score:.2f}" if cf_score else ""}'
            f'{" | Score: " + f"{combined_score:.3f}" if combined_score else ""}'
            f"</p>",
            unsafe_allow_html=True,
        )

        # Row 3: Explanation
        if explanation:
            st.markdown(
                f'<p class="explanation">💬 {explanation}</p>', unsafe_allow_html=True
            )

        # Row 4: Similar button
        if show_similar_button and similar_callback:
            if st.button(
                "🔍 Find Similar", key=f"{key_prefix}sim_{name}", type="secondary"
            ):
                similar_callback(restaurant)

        st.markdown("</div>", unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    variant: str = "default",
    delta: Optional[str] = None,
) -> None:
    """Render a styled metric card for the Ops Monitor page.

    Args:
        label: Metric label (e.g., "On-Time Rate").
        value: Metric value string.
        variant: 'default', 'danger', or 'success'.
        delta: Optional delta string.
    """
    variant_class = f" {variant}" if variant != "default" else ""
    st.markdown(
        f"""
        <div class="metric-card{variant_class}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {f'<div style="font-size:0.75rem;color:#6b7280;margin-top:0.25rem;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
