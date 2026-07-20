"""Full training pipeline v3 — data cleaning, feature engineering,
model comparison with CatBoost, collaborative filtering, reliability
score A/B scenarios, and interactive chart generation.

Usage: python -m dabba.pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dabba.config import get_config
from dabba.data.cleaning import clean_delivery, clean_zomato
from dabba.data.loaders import describe_dataset, load_delivery, load_zomato
from dabba.evaluation.business_cost import (
    compute_reliability_score,
    compute_sla_analysis,
    run_ab_scenario_simulation,
)
from dabba.features.delivery_features import add_delivery_features
from dabba.features.geo import compare_clustering_methods
from dabba.features.restaurant_features import add_restaurant_features
from dabba.models.collaborative_recommender import (
    generate_synthetic_interactions,
    save_collaborative_model,
    train_matrix_factorization,
)
from dabba.models.eta_model import (
    fit_best_eta_model,
    train_and_evaluate_eta_models,
)
from dabba.models.model_selection import (
    comparison_to_dataframe,
    save_comparison_csv,
    select_best_model,
)
from dabba.models.rating_model import (
    fit_best_rating_model,
    train_and_evaluate_rating_models,
)
from dabba.nlp.sentiment import add_sentiment_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dabba.pipeline")


def generate_comparison_charts(
    results: list,
    task: str = "rating",
    output_dir: Path | None = None,
) -> None:
    """Generate comparison charts — both matplotlib (saved) and
    Plotly JSON (for interactive display in UI)."""
    if output_dir is None:
        output_dir = get_config().reports_figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {"Model": r.name, "MAE": r.mae, "RMSE": r.rmse, "R2": r.r2}
        for r in results
    ])

    # Matplotlib bar chart (for README/static images)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["MAE"], width, label="MAE", color="#2196F3")
    ax.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="#FF9800")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title(f"{task.title()} Model Comparison — MAE & RMSE", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task}_model_comparison.png", dpi=150)
    plt.close()

    # Try Plotly interactive version
    try:
        import plotly.express as px
        import plotly.io as pio

        fig_px = px.bar(
            df, x="Model", y=["MAE", "RMSE"],
            title=f"{task.title()} Model Comparison — MAE & RMSE",
            barmode="group", template="plotly_white",
        )
        pio.write_json(fig_px, output_dir / f"{task}_model_comparison.json")
    except ImportError:
        pass

    # R² chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if r.r2 > 0.7 else "#FFC107" if r.r2 > 0.5 else "#F44336"
              for r in results]
    ax.barh(df["Model"], df["R2"], color=colors)
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_title(f"{task.title()} Model Comparison — R² Score", fontsize=14)
    ax.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good (0.7)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task}_r2_comparison.png", dpi=150)
    plt.close()

    logger.info("Comparison charts saved for %s task", task)


def generate_residual_plots(
    results: list,
    y_true: np.ndarray,
    task: str = "rating",
    top_n: int = 3,
    output_dir: Path | None = None,
) -> None:
    """Generate residual plots for top N models."""
    if output_dir is None:
        output_dir = get_config().reports_figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r.mae)[:top_n]
    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    if top_n == 1:
        axes = [axes]

    for ax, result in zip(axes, sorted_results):
        if result.predictions is not None:
            residuals = result.predictions - y_true
            ax.scatter(y_true, residuals, alpha=0.3, s=10)
            ax.axhline(y=0, color="red", linestyle="--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Residual")
            ax.set_title(f"{result.name}\\nMAE={result.mae:.3f}")
            ax.grid(alpha=0.3)

    plt.suptitle(f"{task.title()} — Residual Plots (Top {top_n})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task}_residuals.png", dpi=150)
    plt.close()

    # Try Plotly interactive version
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        fig = go.Figure()
        for result in sorted_results:
            if result.predictions is not None:
                fig.add_trace(go.Scatter(
                    x=y_true, y=result.predictions - y_true,
                    mode="markers", name=result.name,
                    marker=dict(size=4, opacity=0.3),
                ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title=f"{task.title()} — Residuals", template="plotly_white")
        pio.write_json(fig, output_dir / f"{task}_residuals.json")
    except ImportError:
        pass

    logger.info("Residual plots saved for %s task", task)


def compute_shap_explanations(
    model_pipeline,
    X: pd.DataFrame,
    task: str = "rating",
    output_dir: Path | None = None,
) -> None:
    """Compute and save SHAP explanations for the winning model."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping explainability")
        return

    if output_dir is None:
        output_dir = get_config().reports_figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract the model from the pipeline
        inner_model = model_pipeline.named_steps.get("model")
        if inner_model is None:
            logger.warning("Could not extract model from pipeline")
            return

        # Use TreeExplainer for tree-based models
        if hasattr(inner_model, "get_booster") or hasattr(inner_model, "feature_importances_"):
            explainer = shap.TreeExplainer(inner_model)
        else:
            explainer = shap.Explainer(inner_model, X)

        shap_values = explainer(X.sample(min(100, len(X)), random_state=42))
        shap.summary_plot(shap_values, show=False)
        plt.savefig(output_dir / f"{task}_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved for %s model", task)

        # Save shap values for UI
        np.save(output_dir / f"{task}_shap_values.npy", shap_values.values)
    except Exception as e:
        logger.warning("SHAP computation failed for %s: %s", task, e)


def main() -> None:
    """Run the full v3 training pipeline."""
    config = get_config()
    logger.info("=== Dabba v3 Training Pipeline ===")
    logger.info("Project root: %s", config.project_root)

    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.reports_figures_dir.mkdir(parents=True, exist_ok=True)
    config.data_processed_dir.mkdir(parents=True, exist_ok=True)

    # ─── Stage 2: Restaurant Intelligence ────────────────────────────
    logger.info("\n=== STAGE 2: Restaurant Intelligence ===")

    df_zomato = load_zomato(config)
    describe_dataset(df_zomato, "Zomato (raw)")
    df_zomato = clean_zomato(df_zomato, config)
    describe_dataset(df_zomato, "Zomato (cleaned)")
    df_zomato = add_restaurant_features(df_zomato, config)
    df_zomato = add_sentiment_scores(df_zomato, config=config)

    processed_path = config.data_processed_dir / "restaurants_processed.csv"
    df_zomato.to_csv(processed_path, index=False)
    logger.info("Saved processed restaurant data to %s", processed_path)

    # Rating features — deduplicated to avoid "columns are not unique" error
    seen = set()
    feature_cols = []
    for c in df_zomato.columns:
        if c.startswith("cuisine_") or c in [
            "votes_log", "cost_for_two", "online_order_binary",
            "book_table_binary", "cuisine_count", "avg_sentiment",
        ]:
            if c not in seen:
                seen.add(c)
                feature_cols.append(c)

    X_rating = df_zomato[feature_cols].fillna(0)
    y_rating = df_zomato["rate"]

    # Rating model comparison (now with CatBoost + MLflow)
    logger.info("--- Rating model comparison ---")
    rating_results, rating_best = train_and_evaluate_rating_models(X_rating, y_rating, config)

    fitted_rating_model = None
    if rating_results:
        rating_df = comparison_to_dataframe(rating_results, task="rating")
        save_comparison_csv(rating_df, config.rating_comparison_path)
        generate_comparison_charts(rating_results, task="rating")
        generate_residual_plots(rating_results, y_rating.values, task="rating")

        best_name = select_best_model(rating_results, metric=config.rating_metric, task="rating")
        logger.info("🏆 Best rating model: %s", best_name)

        if best_name:
            fitted_rating_model = fit_best_rating_model(
                best_name, X_rating, y_rating, config.best_rating_model_path, config
            )
            # SHAP explainability
            compute_shap_explanations(fitted_rating_model, X_rating, task="rating")

    # ─── Stage 3: Delivery ETA Engine ────────────────────────────────
    logger.info("\n=== STAGE 3: Delivery ETA Engine ===")

    df_delivery = load_delivery(config)
    describe_dataset(df_delivery, "Delivery (raw)")
    df_delivery = clean_delivery(df_delivery, config)
    describe_dataset(df_delivery, "Delivery (cleaned)")
    df_delivery = add_delivery_features(df_delivery, config)

    eta_feature_cols = [c for c in df_delivery.columns if c in [
        "haversine_distance_km", "traffic_ordinal", "is_festival",
        "delivery_person_age", "delivery_person_ratings", "vehicle_condition",
    ]]
    eta_feature_cols += [c for c in df_delivery.columns if c.startswith("order_hour_bucket_")]

    X_eta = df_delivery[eta_feature_cols].fillna(0)
    y_eta = df_delivery["time_taken_min"]

    logger.info("--- ETA model comparison (including CatBoost) ---")
    eta_results, eta_best = train_and_evaluate_eta_models(X_eta, y_eta, config)

    fitted_eta_model = None
    if eta_results:
        eta_df = comparison_to_dataframe(eta_results, task="eta")
        save_comparison_csv(eta_df, config.eta_comparison_path)
        generate_comparison_charts(eta_results, task="eta")
        generate_residual_plots(eta_results, y_eta.values, task="eta")

        best_eta_name = select_best_model(eta_results, metric=config.eta_metric, task="eta")
        logger.info("🏆 Best ETA model: %s", best_eta_name)

        if best_eta_name:
            fitted_eta_model = fit_best_eta_model(
                best_eta_name, X_eta, y_eta, config.best_eta_model_path, config
            )
            compute_shap_explanations(fitted_eta_model, X_eta, task="eta")

        if eta_best and eta_best.predictions is not None:
            sla_metrics = compute_sla_analysis(y_eta.values, eta_best.predictions, config=config)
            logger.info("SLA analysis: %s", sla_metrics)

    # ─── Stage 4: Reliability Score + A/B Scenarios ──────────────────
    logger.info("\n=== STAGE 4: Reliability Score & A/B Scenarios ===")

    if fitted_rating_model is not None and fitted_eta_model is not None:
        rating_scores = y_rating.values
        sentiment_scores = (
            df_zomato["avg_sentiment"].values
            if "avg_sentiment" in df_zomato.columns
            else np.zeros(len(df_zomato))
        )
        delay_risk = np.full(len(df_zomato), config.sla_threshold_minutes * 0.5)

        # Default reliability score
        reliability = compute_reliability_score(
            rating_scores, sentiment_scores, delay_risk, config=config
        )
        df_zomato["reliability_score"] = reliability
        df_zomato.to_csv(processed_path, index=False)
        logger.info("Reliability scores computed — mean=%.3f", float(np.mean(reliability)))

        # A/B scenario simulation
        scenario_df = df_zomato.copy()
        scenario_df["delay_risk"] = delay_risk
        ab_results = run_ab_scenario_simulation(scenario_df)
        ab_path = config.reports_dir / "ab_scenarios.json"
        with open(ab_path, "w") as f:
            json.dump(ab_results, f, indent=2, default=str)
        logger.info("A/B scenarios saved to %s", ab_path)

    # ─── Stage 5: Collaborative Filtering (Synthetic Data) ───────────
    logger.info("\n=== STAGE 5: Collaborative Filtering ===")

    logger.info("--- Generating synthetic user-restaurant interactions ---")
    logger.info("NOTE: This is SYNTHETIC data — not real user behavior.")
    logger.info("In production, this would use real order/rating logs.")

    interactions = generate_synthetic_interactions(
        df_zomato, n_users=3000, config=config
    )
    interactions.to_csv(config.synthetic_interactions_path, index=False)
    logger.info("Saved %d synthetic interactions to %s",
                len(interactions), config.synthetic_interactions_path)

    logger.info("--- Training matrix factorization model ---")
    n_users = int(interactions["user_id"].max()) + 1
    n_items_map = {i: i for i in range(len(df_zomato))}  # map restaurant indices

    try:
        mf_model = train_matrix_factorization(
            interactions,
            n_users=n_users,
            n_items=len(df_zomato),
            n_factors=50,
            n_epochs=20,
            device="cpu",
            config=config,
        )
        save_collaborative_model(mf_model, config.best_collaborative_model_path)
        logger.info("✅ Collaborative filtering model saved")
    except Exception as e:
        logger.error("Collaborative filtering training failed: %s", e)

    # ─── Stage 6: Geographic Clustering ──────────────────────────────
    logger.info("\n=== Stage 6: Geographic Clustering ===")

    lat_col = lon_col = None
    for c in ["restaurant_latitude", "latitude", "lat"]:
        if c in df_zomato.columns and df_zomato[c].notna().sum() > 10:
            lat_col = c
            break
    for c in ["restaurant_longitude", "longitude", "lon", "lng"]:
        if c in df_zomato.columns and df_zomato[c].notna().sum() > 10:
            lon_col = c
            break

    if lat_col and lon_col:
        coords = df_zomato[[lat_col, lon_col]].dropna().values
        if len(coords) > 20:
            cluster_results = compare_clustering_methods(coords, k_range=range(3, 11))
            for method, info in cluster_results.items():
                logger.info("  %s: silhouette=%.3f", method, info.get("silhouette_score", -1))

    logger.info("\n=== Pipeline Complete ===")
    logger.info("Best rating model: %s", config.best_rating_model_path)
    logger.info("Best ETA model: %s", config.best_eta_model_path)
    logger.info("Collaborative model: %s", config.best_collaborative_model_path)
    logger.info("Comparison CSVs: %s", config.reports_dir)
    logger.info("A/B scenarios: %s", config.reports_dir / "ab_scenarios.json")


if __name__ == "__main__":
    main()
