"""Page 4: Food Concierge — chat interface with styled bubbles,
tool-use integration, and example prompt chips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from dabba.config import get_config
from dabba.llm.food_concierge import ConciergeTools, get_concierge_response

PAGE_NAME = "concierge"
config = get_config()


def show() -> None:
    """Render the Food Concierge page."""
    st.title("💬 Food Concierge")
    st.markdown(
        "Your AI-powered restaurant discovery assistant. "
        "Ask for recommendations, ETAs, or reliability scores in plain English."
    )

    # Load restaurant data
    df = _load_data()

    # Initialize concierge tools
    tools = ConciergeTools(
        restaurants_df=df if df is not None else pd.DataFrame(),
        eta_model=None,
        config=config,
    )

    # Initialize chat history
    if f"{PAGE_NAME}_messages" not in st.session_state:
        st.session_state[f"{PAGE_NAME}_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Namaste!** I'm your Dabba Food Concierge. "
                    "I can help you find restaurants, check delivery ETAs, "
                    "or look up reliability scores.\n\n"
                    "Try asking me something like:\n"
                    "- \"Find me something spicy under ₹400 near Koramangala\"\n"
                    "- \"What's the most reliable restaurant in Indiranagar?\"\n"
                    "- \"How long does delivery from Meghana Foods take?\""
                ),
            }
        ]

    # ─── Example prompt chips ─────────────────────────────────────
    examples = [
        "Find North Indian food under ₹500 near Koramangala",
        "What's the most reliable restaurant in Indiranagar?",
        "How long does delivery from Truffles take?",
        "Find cheap restaurants under ₹300",
    ]

    # Only show chips at the start
    if len(st.session_state[f"{PAGE_NAME}_messages"]) <= 1:
        st.markdown("##### Quick examples:")
        chip_cols = st.columns(len(examples))
        for col, example in zip(chip_cols, examples):
            with col:
                if st.button(f"💡 {example}", key=f"{PAGE_NAME}_chip_{example[:20]}",
                             use_container_width=True, type="secondary"):
                    st.session_state[f"{PAGE_NAME}_input"] = example
                    st.rerun()

    # ─── Chat display ─────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state[f"{PAGE_NAME}_messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-assistant">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # ─── Chat input ───────────────────────────────────────────────
    # Use text from example chips if set
    input_key = f"{PAGE_NAME}_input"
    default_value = st.session_state.get(input_key, "")

    user_input = st.chat_input(
        "Ask about restaurants, ETAs, or reliability...",
        key=f"{PAGE_NAME}_chat_input",
    )

    # Clear the stored input after using it
    if input_key in st.session_state:
        del st.session_state[input_key]

    if user_input:
        # Add user message
        st.session_state[f"{PAGE_NAME}_messages"].append(
            {"role": "user", "content": user_input}
        )

        # Generate response
        with st.spinner("Thinking..."):
            response = get_concierge_response(
                user_input,
                st.session_state[f"{PAGE_NAME}_messages"][:-1],
                tools,
                config=config,
            )

        st.session_state[f"{PAGE_NAME}_messages"].append(
            {"role": "assistant", "content": response}
        )

        st.rerun()


@st.cache_data
def _load_data() -> pd.DataFrame:
    """Load processed restaurant data."""
    data_path = Path("data/processed/restaurants_processed.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()
