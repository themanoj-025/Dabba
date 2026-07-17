"""Customer View — personalized restaurant recommendations."""

from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Customer View — Dabba", page_icon="🍽️", layout="wide")

st.title("🍽️ Restaurant Recommendations")
st.markdown("Tell us what you're craving, and we'll find the best restaurants for you.")

# --- Sidebar filters ---
st.sidebar.header("Your Preferences")

cuisine_options = [
    "North Indian", "Chinese", "South Indian", "Mughlai", "Cafe",
    "Bakery", "Italian", "Fast Food", "Continental", "Desserts",
    "Biryani", "Street Food", "Ice Cream", "Andhra", "Thai",
]
selected_cuisines = st.sidebar.multiselect("Cuisine", cuisine_options)

budget = st.sidebar.slider(
    "Budget (₹ for two)",
    min_value=100, max_value=5000, value=1000, step=100,
)

area_options = [
    "Koramangala", "Indiranagar", "HSR Layout", "Whitefield",
    "Electronic City", "JP Nagar", "BTM Layout", "Marathahalli",
    "Jayanagar", "MG Road", "Sarjapur Road",
]
selected_area = st.sidebar.selectbox("Area", ["All"] + area_options)

top_n = st.sidebar.slider("Number of results", 3, 20, 5)

# --- Load data and model ---
@st.cache_data
def load_data():
    """Load processed restaurant data."""
    data_path = Path("data/processed/restaurants_processed.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is None:
    st.warning(
        "⚠️ Processed data not found. Please run `make train` first to "
        "download and process the datasets."
    )
    st.stop()

# --- Apply filters ---
filtered = df.copy()

if selected_cuisines:
    cuisine_mask = filtered["cuisines"].str.contains(
        "|".join(selected_cuisines), case=False, na=False
    )
    filtered = filtered[cuisine_mask]

if selected_area and selected_area != "All":
    filtered = filtered[
        filtered["location"].str.contains(selected_area, case=False, na=False)
    ]

if "cost_for_two" in filtered.columns:
    filtered = filtered[filtered["cost_for_two"] <= budget]

# --- Display results ---
st.subheader(f"Top {min(top_n, len(filtered))} Restaurants")

if filtered.empty:
    st.info("No restaurants match your filters. Try adjusting your preferences.")
else:
    display_cols = ["name", "rate", "cost_for_two", "location", "cuisines"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    results = filtered.head(top_n)

    for _, row in results.iterrows():
        with st.container():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.markdown(f"**{row.get('name', 'Unknown')}**")
                st.caption(f"📍 {row.get('location', 'N/A')} | 🍳 {row.get('cuisines', 'N/A')[:50]}")
            with cols[1]:
                rating = row.get("rate", "N/A")
                st.metric("Rating", f"{rating}/5" if rating != "N/A" else "N/A")
            with cols[2]:
                cost = row.get("cost_for_two", "N/A")
                st.metric("Cost (₹2)", f"₹{int(cost)}" if cost != "N/A" else "N/A")
            st.divider()
