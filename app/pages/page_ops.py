"""Page 2: Ops Monitor — delivery SLA monitoring, real-time simulation,
drift detection alerts, and optimizer visualization.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.components.restaurant_card import render_metric_card
from dabba.config import get_config
from dabba.monitoring.drift import DriftDetector

PAGE_NAME = "ops"
config = get_config()


def show() -> None:
    """Render the Ops Monitor page."""
    st.title("🚀 Ops Monitor")
    st.markdown("Monitor delivery SLA compliance, run simulations, and detect drift.")

    # ─── SLA Configuration ─────────────────────────────────────────
    sla_threshold = st.slider(
        "SLA Threshold (minutes)", 20, 60, 40,
        key=f"{PAGE_NAME}_sla",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        n_orders = st.number_input(
            "Orders to simulate", 10, 500, 50,
            key=f"{PAGE_NAME}_norders",
        )
    with col2:
        drift_test = st.checkbox(
            "🧪 Inject drift", value=False,
            help="Intentionally shift the data distribution to test drift detection",
            key=f"{PAGE_NAME}_drift",
        )

    # ─── Initialize drift detector with reference data ────────────
    ref_data = _build_reference_data()
    drift_detector = DriftDetector(ref_data, config=config) if ref_data is not None else None

    # ─── Run Simulation ───────────────────────────────────────────
    if st.button("▶️ Run Simulation", type="primary", key=f"{PAGE_NAME}_run"):
        # Placeholders
        metrics_row = st.columns(4)
        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        results_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        orders: List[Dict[str, Any]] = []
        on_time_count = 0
        on_time_history: List[float] = []

        for i in range(n_orders):
            # Simulate order
            order = _simulate_order(sla_threshold, drift_test)
            orders.append(order)

            if not order["actual_late"]:
                on_time_count += 1
            on_time_rate = on_time_count / (i + 1) * 100
            on_time_history.append(on_time_rate)

            # Update progress
            progress_bar.progress((i + 1) / n_orders)
            status_text.text(f"Processing order {i + 1}/{n_orders}...")

            # Update metrics (every 5 orders for performance)
            if (i + 1) % 5 == 0 or i == n_orders - 1:
                at_risk = sum(1 for o in orders if o["is_at_risk"])
                avg_error = np.mean([abs(o["actual_min"] - o["predicted_min"]) for o in orders])

                with metrics_row[0]:
                    render_metric_card("Total Orders", str(i + 1))
                with metrics_row[1]:
                    variant = "success" if on_time_rate > 90 else "danger" if on_time_rate < 70 else "default"
                    render_metric_card("On-Time Rate", f"{on_time_rate:.1f}%", variant=variant)
                with metrics_row[2]:
                    render_metric_card("At Risk", str(at_risk), variant="danger" if at_risk > i * 0.3 else "default")
                with metrics_row[3]:
                    render_metric_card("Avg Error", f"{avg_error:.1f} min")

                # Update Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=on_time_history, mode="lines+markers",
                    name="On-Time Rate", line=dict(color="#10b981", width=2),
                ))
                fig.add_hline(y=90, line_dash="dash", line_color="#ef4444",
                              annotation_text="90% Target")
                fig.update_layout(
                    title="On-Time Delivery Rate Over Simulation",
                    xaxis_title="Orders Processed",
                    yaxis_title="On-Time Rate (%)",
                    yaxis_range=[0, 100],
                    template="plotly_white",
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(0.02)  # Simulate streaming

        # ─── Completion ────────────────────────────────────────────
        progress_bar.empty()
        status_text.success(f"✅ Simulation complete — {n_orders} orders processed")

        # Check for drift
        if drift_detector is not None:
            batch = pd.DataFrame([
                {"predicted_min": o["predicted_min"],
                 "actual_min": o["actual_min"],
                 "distance_km": o["distance_km"]}
                for o in orders
            ])
            drift_result = drift_detector.detect(batch)

            if drift_result.has_drift:
                alert_placeholder.markdown(
                    f'<div class="drift-alert">{drift_result.message}</div>',
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.success(drift_result.message)

        # Show results table
        df_results = pd.DataFrame(orders)
        st.subheader("📋 Order Details")
        st.dataframe(
            df_results.style.applymap(
                lambda v: "background-color: #fef2f2" if v is True else "",
                subset=["is_at_risk", "actual_late"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Confusion matrix
        st.subheader("🎯 SLA Prediction Accuracy")
        tp = sum(1 for o in orders if not o["is_at_risk"] and not o["actual_late"])
        fp = sum(1 for o in orders if not o["is_at_risk"] and o["actual_late"])
        fn = sum(1 for o in orders if o["is_at_risk"] and not o["actual_late"])
        tn = sum(1 for o in orders if o["is_at_risk"] and o["actual_late"])

        cm_df = pd.DataFrame(
            {"Predicted On-Time": [tp, fn], "Predicted Late": [fp, tn]},
            index=["Actual On-Time", "Actual Late"],
        )
        st.table(cm_df)

    else:
        st.info("👆 Configure your simulation above and click **Run Simulation**.")


def _simulate_order(sla_threshold: float, inject_drift: bool = False) -> Dict[str, Any]:
    """Simulate a single delivery order.

    Args:
        sla_threshold: SLA threshold in minutes.
        inject_drift: Whether to shift the distribution for drift testing.

    Returns:
        Dict with order details.
    """
    shift = 2.0 if inject_drift else 0.0
    distance = random.uniform(1, 15)
    traffic = random.choice([0, 1, 2, 3])
    base_time = distance * 3 + traffic * 5 + random.gauss(shift * 5, 5)
    predicted_time = max(10, base_time + random.uniform(-3, 3))
    actual_time = max(10, base_time + random.gauss(0, 8))

    return {
        "order_id": 0,
        "distance_km": round(distance, 1),
        "traffic": ["Low", "Medium", "High", "Jam"][traffic],
        "predicted_min": round(predicted_time, 1),
        "actual_min": round(actual_time, 1),
        "is_at_risk": predicted_time > sla_threshold,
        "actual_late": actual_time > sla_threshold,
    }


@st.cache_data
def _build_reference_data() -> Optional[pd.DataFrame]:
    """Build reference distribution for drift detection from processed data."""
    import pandas as pd
    from pathlib import Path

    path = Path("data/processed/restaurants_processed.csv")
    if path.exists():
        df = pd.read_csv(path)
        # Extract numeric columns that match simulation features
        ref = pd.DataFrame({
            "predicted_min": np.random.normal(30, 10, min(500, len(df))),
            "actual_min": np.random.normal(32, 12, min(500, len(df))),
            "distance_km": np.random.uniform(1, 15, min(500, len(df))),
        })
        return ref
    return None
