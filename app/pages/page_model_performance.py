"""Page 3: Model Performance — interactive Plotly charts, comparison
tables, SHAP plots, and delivery-optimizer visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

PAGE_NAME = "model"


def show() -> None:
    """Render the Model Performance page."""
    st.title("📊 Model Performance")
    st.markdown(
        "Interactive comparison of all candidate models. "
        "Charts update automatically when models are retrained."
    )

    # ─── Paths ────────────────────────────────────────────────────
    reports = Path("reports")

    # ─── Rating Model Comparison ──────────────────────────────────
    st.header("🍽️ Rating Prediction")

    rating_csv = reports / "model_comparison_rating.csv"
    if rating_csv.exists():
        rating_df = pd.read_csv(rating_csv)
        _show_model_section(rating_df, "rating")
    else:
        st.warning("Rating comparison not found. Run `make train` first.")

    st.divider()

    # ─── ETA Model Comparison ─────────────────────────────────────
    st.header("🚀 Delivery ETA Prediction")

    eta_csv = reports / "model_comparison_eta.csv"
    if eta_csv.exists():
        eta_df = pd.read_csv(eta_csv)
        _show_model_section(eta_df, "eta")
    else:
        st.warning("ETA comparison not found. Run `make train` first.")

    st.divider()

    # ─── A/B Scenario Results ────────────────────────────────────
    st.header("🔄 Reliability Score A/B Scenarios")
    ab_path = reports / "ab_scenarios.json"
    if ab_path.exists():
        _show_ab_scenarios(ab_path)
    else:
        st.info("A/B scenario results not found. Run `make train` to generate them.")

    st.divider()

    # ─── Methodology ───────────────────────────────────────────────
    st.header("📋 Methodology")
    st.markdown("""
        **How the comparison works:**

        1. **Same features** — All models train on identical feature sets
        2. **Same split** — K-fold cross-validation (k=5) with the same random seed
        3. **Same preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals
        4. **Selection rule** — Lowest MAE by default (configurable)
        5. **Only the winner** is wired into the dashboard, API, and SLA logic

        Models compared: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest,
        GradientBoosting, XGBoost, LightGBM, **CatBoost** (+ KNN for ETA).

        All experiments logged via **MLflow** — view the tracking UI at `localhost:5000`.
        """)


def _show_model_section(df: pd.DataFrame, task: str) -> None:
    """Display winner callout, comparison table, and interactive charts."""
    required_cols = {"model", "mae", "rmse", "r2"}
    if not required_cols.issubset(df.columns):
        st.error(f"{task.title()} CSV has unexpected format.")
        return

    # Winner callout
    best = df.sort_values("mae").iloc[0]
    st.success(
        f"🏆 **Winner: {best['model']}** — "
        f"MAE: {best['mae']:.4f} | "
        f"RMSE: {best['rmse']:.4f} | "
        f"R²: {best['r2']:.4f}"
    )

    # Comparison table with highlighting
    st.subheader("Full Comparison Table")
    styled = df.style.highlight_min(subset=["mae", "rmse"], color="#d4edda")
    if "r2" in df.columns:
        styled = styled.highlight_max(subset=["r2"], color="#d4edda")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Interactive Plotly chart: MAE & RMSE
    st.subheader("Interactive Model Comparison")
    chart_path = Path("reports/figures") / f"{task}_model_comparison.json"
    if chart_path.exists():
        try:
            fig = pio.read_json(chart_path)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # Fallback: plotly express
    fig = px.bar(
        df,
        x="model",
        y=["mae", "rmse"],
        title=f"{task.title()} — MAE & RMSE by Model",
        barmode="group",
        template="plotly_white",
        color_discrete_map={"mae": "#ff8c42", "rmse": "#6b7280"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # R² chart
    fig_r2 = px.bar(
        df,
        x="model",
        y="r2",
        title=f"{task.title()} — R² Score",
        template="plotly_white",
        color="r2",
        color_continuous_scale="RdYlGn",
    )
    fig_r2.update_layout(xaxis_tickangle=-45, height=350)
    st.plotly_chart(fig_r2, use_container_width=True)

    # Residual plot
    resid_json = Path("reports/figures") / f"{task}_residuals.json"
    if resid_json.exists():
        try:
            fig_res = pio.read_json(resid_json)
            st.subheader("Residual Plots (Top 3 Models)")
            st.plotly_chart(fig_res, use_container_width=True)
        except Exception:
            pass

    # SHAP summary
    shap_png = Path("reports/figures") / f"{task}_shap_summary.png"
    if shap_png.exists():
        st.subheader("🔍 SHAP Feature Importance")
        st.image(str(shap_png), use_container_width=True)

    # Winner reasoning
    st.markdown(f"""
        **Why {best['model']} won:**
        This model achieved the lowest MAE of {best['mae']:.4f}
        ({'minutes' if task == 'eta' else 'rating points'}),
        meaning its predictions are closest to the actual values on average.
        The comparison used 5-fold cross-validation with identical features
        for all models, ensuring a fair comparison.
        """)


def _show_ab_scenarios(path: Path) -> None:
    """Display A/B scenario simulation results."""
    try:
        with open(path) as f:
            data = json.load(f)

        meta = data.pop("_meta", {})
        scenarios = list(data.keys())

        cols = st.columns(len(scenarios))
        for col, (name, info) in zip(cols, data.items()):
            with col:
                mean_score = info.get("mean_score", 0)
                desc = info.get("description", "")
                st.metric(
                    label=name.replace("_", " ").title(),
                    value=f"{mean_score:.3f}",
                    help=desc,
                )

        # Show top restaurants for each scenario
        tab_names = [s.replace("_", " ").title() for s in scenarios]
        tabs = st.tabs(tab_names)

        for tab, (name, info) in zip(tabs, data.items()):
            with tab:
                desc = info.get("description", "")
                st.caption(desc)
                for i, r in enumerate(info.get("top_restaurants", [])[:5]):
                    st.markdown(
                        f"{i+1}. **{r.get('name', '?')}** — "
                        f"Score: {r.get('score', 0):.3f}"
                    )

        # Overlap analysis
        if meta:
            st.caption(
                f"🔗 Overlap: Balanced↔Quality: {meta.get('balanced_vs_quality_overlap', '?')} restaurants | "
                f"Balanced↔Speed: {meta.get('balanced_vs_speed_overlap', '?')} | "
                f"Quality↔Speed: {meta.get('quality_vs_speed_overlap', '?')}"
            )
    except Exception as e:
        st.warning(f"Could not load A/B scenarios: {e}")
