"""Model Info — shows which algorithms won and their comparison charts.

This page reinforces that the model selection was rigorous and data-driven,
not a black-box choice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Info — Dabba", page_icon="📊", layout="wide")

st.title("📊 Model Comparison & Selection")
st.markdown(
    "This page shows which algorithms were compared and why the winning model "
    "was selected. The choice was **data-driven** — lowest MAE on held-out data "
    "with k-fold cross-validation."
)

# --- Config paths ---
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
RATING_CSV = REPORTS_DIR / "model_comparison_rating.csv"
ETA_CSV = REPORTS_DIR / "model_comparison_eta.csv"


# --- Load comparison data ---
@st.cache_data
def load_rating_comparison() -> pd.DataFrame | None:
    """Load rating model comparison CSV."""
    if RATING_CSV.exists():
        return pd.read_csv(RATING_CSV)
    return None


@st.cache_data
def load_eta_comparison() -> pd.DataFrame | None:
    """Load ETA model comparison CSV."""
    if ETA_CSV.exists():
        return pd.read_csv(ETA_CSV)
    return None


rating_df = load_rating_comparison()
eta_df = load_eta_comparison()


# ================================================================
# RATING MODEL COMPARISON
# ================================================================
st.header("🍽️ Rating Prediction Model")

if rating_df is not None and not rating_df.empty:
    # Validate CSV columns
    required_cols = {"model", "mae", "rmse", "r2"}
    if not required_cols.issubset(rating_df.columns):
        st.error("Rating comparison CSV has unexpected format. Expected columns: model, mae, rmse, r2")
    else:
        # Best model callout
        best_rating = rating_df.iloc[0]
        st.success(
        f"🏆 **Winner: {best_rating['model']}** — "
        f"MAE: {best_rating['mae']:.4f} | "
        f"RMSE: {best_rating['rmse']:.4f} | "
        f"R²: {best_rating['r2']:.4f}"
    )

    # Comparison table
    st.subheader("Full Comparison Table")
    st.dataframe(
        rating_df.style.highlight_min(
            subset=["mae", "rmse"], color="#d4edda"
        ).highlight_max(
            subset=["r2"], color="#d4edda"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    st.subheader("MAE & RMSE Comparison")
    rating_chart_path = FIGURES_DIR / "rating_model_comparison.png"
    if rating_chart_path.exists():
        st.image(str(rating_chart_path), use_container_width=True)
    else:
        st.info("Chart not available yet. Run `make train` to generate it.")

    # R² chart
    rating_r2_path = FIGURES_DIR / "rating_r2_comparison.png"
    if rating_r2_path.exists():
        st.subheader("R² Score Comparison")
        st.image(str(rating_r2_path), use_container_width=True)

    # Residual plots
    rating_resid_path = FIGURES_DIR / "rating_residuals.png"
    if rating_resid_path.exists():
        st.subheader("Residual Plots (Top 3 Models)")
        st.image(str(rating_resid_path), use_container_width=True)

        # Interpretation
        st.markdown(
            f"""
            **Why {best_rating['model']} won:**
            This model achieved the lowest MAE of {best_rating['mae']:.4f} on rating prediction,
            meaning its predictions are closest to actual restaurant ratings on average.
            The comparison was done with 5-fold cross-validation using identical features
            for all models, ensuring a fair comparison.
            """
        )
else:
    st.warning(
        "⚠️ Rating model comparison not found. Run `make train` to generate "
        "the comparison CSV and charts."
    )

st.divider()


# ================================================================
# ETA MODEL COMPARISON
# ================================================================
st.header("🚀 Delivery ETA Prediction Model")

if eta_df is not None and not eta_df.empty:
    # Validate CSV columns
    required_cols = {"model", "mae", "rmse", "r2"}
    if not required_cols.issubset(eta_df.columns):
        st.error("ETA comparison CSV has unexpected format. Expected columns: model, mae, rmse, r2")
    else:
        # Best model callout
        best_eta = eta_df.iloc[0]
        st.success(
        f"🏆 **Winner: {best_eta['model']}** — "
        f"MAE: {best_eta['mae']:.2f} min | "
        f"RMSE: {best_eta['rmse']:.2f} min | "
        f"R²: {best_eta['r2']:.4f}"
    )

    # Comparison table
    st.subheader("Full Comparison Table")
    st.dataframe(
        eta_df.style.highlight_min(
            subset=["mae", "rmse"], color="#d4edda"
        ).highlight_max(
            subset=["r2"], color="#d4edda"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    st.subheader("MAE & RMSE Comparison")
    eta_chart_path = FIGURES_DIR / "eta_model_comparison.png"
    if eta_chart_path.exists():
        st.image(str(eta_chart_path), use_container_width=True)
    else:
        st.info("Chart not available yet. Run `make train` to generate it.")

    # R² chart
    eta_r2_path = FIGURES_DIR / "eta_r2_comparison.png"
    if eta_r2_path.exists():
        st.subheader("R² Score Comparison")
        st.image(str(eta_r2_path), use_container_width=True)

    # Residual plots
    eta_resid_path = FIGURES_DIR / "eta_residuals.png"
    if eta_resid_path.exists():
        st.subheader("Residual Plots (Top 3 Models)")
        st.image(str(eta_resid_path), use_container_width=True)

        # Interpretation
        st.markdown(
            f"""
            **Why {best_eta['model']} won:**
            This model achieved the lowest MAE of {best_eta['mae']:.2f} minutes on delivery time
            prediction, meaning its ETA predictions are closest to actual delivery times.
            Gradient boosting methods typically excel here because they capture non-linear
            interactions between traffic density, distance, and time-of-day that linear
            models miss.

            The comparison used the same features and 5-fold cross-validation for all
            candidates, ensuring the selection is statistically fair.
            """
        )
else:
    st.warning(
        "⚠️ ETA model comparison not found. Run `make train` to generate "
        "the comparison CSV and charts."
    )

st.divider()


# ================================================================
# METHODOLOGY NOTE
# ================================================================
st.header("📋 Methodology")
st.markdown(
    """
    **How the comparison works:**

    1. **Same features** — All models train on identical feature sets
    2. **Same split** — K-fold cross-validation (k=5) with the same random seed
    3. **Same preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals
    4. **Selection rule** — Lowest MAE by default (configurable via `config.eta_metric` and `config.rating_metric`)
    5. **Only the winner** is wired into the dashboard, API, and SLA logic

    This ensures the comparison is fair and the winning model genuinely outperforms
    the others — not just cherry-picked or lucky with a particular train/test split.
    """
)
