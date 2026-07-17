"""Operations View — delivery SLA monitoring and simulation."""

from __future__ import annotations

import time
import random
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Operations View — Dabba", page_icon="🚀", layout="wide")

st.title("🚀 Delivery Operations")
st.markdown("Monitor SLA compliance and run delivery simulations.")

# --- SLA Configuration ---
sla_threshold = st.slider("SLA Threshold (minutes)", 20, 60, 40)

st.subheader("📊 Delivery Simulation")

n_orders = st.number_input("Number of simulated orders", 10, 500, 50)

if st.button("▶️ Run Simulation", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Simulate orders
    orders = []
    on_time_count = 0
    total_count = 0

    for i in range(n_orders):
        # Simulate realistic delivery times
        distance = random.uniform(1, 15)
        traffic = random.choice([0, 1, 2, 3])
        base_time = distance * 3 + traffic * 5 + random.gauss(0, 5)
        predicted_time = max(10, base_time + random.uniform(-3, 3))
        actual_time = max(10, base_time + random.gauss(0, 8))

        is_at_risk = predicted_time > sla_threshold
        actual_late = actual_time > sla_threshold

        orders.append({
            "order_id": i + 1,
            "distance_km": round(distance, 1),
            "traffic": ["Low", "Medium", "High", "Jam"][traffic],
            "predicted_min": round(predicted_time, 1),
            "actual_min": round(actual_time, 1),
            "is_at_risk": is_at_risk,
            "actual_late": actual_late,
        })

        if not actual_late:
            on_time_count += 1
        total_count += 1

        # Update progress
        progress_bar.progress((i + 1) / n_orders)
        status_text.text(f"Processing order {i + 1}/{n_orders}...")

        # Update live metrics
        on_time_rate = on_time_count / total_count * 100
        at_risk_count = sum(1 for o in orders if o["is_at_risk"])

        metrics_placeholder.columns(4)
        col1, col2, col3, col4 = metrics_placeholder.columns(4)
        col1.metric("Total Orders", total_count)
        col2.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        col3.metric("At-Risk Flagged", at_risk_count)
        col4.metric("SLA Threshold", f"{sla_threshold} min")

        time.sleep(0.05)  # Simulate streaming

    progress_bar.empty()
    status_text.success(f"✅ Simulation complete — {n_orders} orders processed")

    # Show results table
    df_results = pd.DataFrame(orders)
    st.dataframe(
        df_results.style.applymap(
            lambda v: "background-color: #ffcccc" if v == True else "",
            subset=["is_at_risk", "actual_late"],
        ),
        use_container_width=True,
    )

    # Show confusion matrix
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
    st.info("Click **Run Simulation** to start a delivery simulation.")
