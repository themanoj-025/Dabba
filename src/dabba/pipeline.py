"""Full training pipeline — runs data cleaning, feature engineering,
model comparison, best-model selection, full-data retrain, and chart generation.

Usage: python -m dabba.pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dabba.config import get_config
from dabba.data.cleaning import clean_delivery, clean_zomato
from dabba.data.loaders import describe_dataset, load_delivery, load_zomato
from dabba.evaluation.business_cost import compute_reliability_score, compute_sla_analysis
from dabba.features.delivery_features import add_delivery_features
from dabba.features.geo import compare_clustering_methods
from dabba.features.restaurant_features import add_restaurant_features
from dabba.models.eta_model import fit_best_eta_model, train_and_evaluate_eta_models
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dabba.pipeline")


def generate_comparison_charts(
    results: list,
    task: str = "rating",
    output_dir: Path | None = None,
) -> None:
    """Generate bar charts comparing model performance.

    Args:
        results: List of ModelResult/ETAModelResult instances.
        task: Task name ('rating' or 'eta').
        output_dir: Directory to save charts.
    """
    if output_dir is None:
        output_dir = get_config().reports_figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {"Model": r.name, "MAE": r.mae, "RMSE": r.rmse, "R2": r.r2}
        for r in results
    ])

    # --- Grouped bar chart: MAE and RMSE ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df["MAE"], width, label="MAE", color="#2196F3")
    bars2 = ax.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="#FF9800")

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
    logger.info("Saved %s comparison chart to %s", task, output_dir / f"{task}_model_comparison.png")

    # --- R² comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if r.r2 > 0.7 else "#FFC107" if r.r2 > 0.5 else "#F44336" for r in results]
    ax.barh(df["Model"], df["R2"], color=colors)
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_title(f"{task.title()} Model Comparison — R² Score", fontsize=14)
    ax.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good threshold (0.7)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{task}_r2_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved %s R² chart to %s", task, output_dir / f"{task}_r2_comparison.png")


def generate_residual_plots(
    results: list,
    y_true: np.ndarray,
    task: str = "rating",
    top_n: int = 3,
    output_dir: Path | None = None,
) -> None:
    """Generate residual plots for top N models.

    Args:
        results: List of ModelResult/ETAModelResult instances.
        y_true: Ground truth values.
        task: Task name.
        top_n: Number of top models to plot.
        output_dir: Directory to save charts.
    """
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
            ax.set_ylabel("Residual (Predicted - Actual)")
            ax.set_title(f"{result.name}\nMAE={result.mae:.3f}")
            ax.grid(alpha=0.3)

    plt.suptitle(f"{task.title()} — Residual Plots (Top {top_n} Models)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{task}_residuals.png", dpi=150)
    plt.close()
    logger.info("Saved %s residual plots to %s", task, output_dir / f"{task}_residuals.png")


def main() -> None:
    """Run the full training pipeline."""
    config = get_config()
    logger.info("=== Dabba Training Pipeline ===")
    logger.info("Project root: %s", config.project_root)

    # Ensure output directories exist
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.reports_figures_dir.mkdir(parents=True, exist_ok=True)
    config.data_processed_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # STAGE 2: Restaurant Intelligence (Dataset A)
    # ================================================================
    logger.info("\n=== STAGE 2: Restaurant Intelligence ===")

    logger.info("--- Loading Zomato data ---")
    df_zomato = load_zomato(config)
    describe_dataset(df_zomato, "Zomato (raw)")

    logger.info("--- Cleaning Zomato data ---")
    df_zomato = clean_zomato(df_zomato, config)
    describe_dataset(df_zomato, "Zomato (cleaned)")

    logger.info("--- Feature engineering ---")
    df_zomato = add_restaurant_features(df_zomato, config)

    logger.info("--- Sentiment analysis ---")
    df_zomato = add_sentiment_scores(df_zomato, config=config)

    # Save processed restaurant data
    processed_path = config.data_processed_dir / "restaurants_processed.csv"
    df_zomato.to_csv(processed_path, index=False)
    logger.info("Saved processed restaurant data to %s", processed_path)

    # Prepare features for rating model
    feature_cols = [c for c in df_zomato.columns if c.startswith("cuisine_")]
    feature_cols += [c for c in ["votes_log", "cost_for_two", "online_order_binary",
                                  "book_table_binary", "cuisine_count", "avg_sentiment"]
                     if c in df_zomato.columns]

    X_rating = df_zomato[feature_cols].fillna(0)
    y_rating = df_zomato["rate"]

    logger.info("--- Rating model comparison ---")
    rating_results, rating_best = train_and_evaluate_rating_models(X_rating, y_rating, config)

    if rating_results:
        # Save comparison CSV
        rating_df = comparison_to_dataframe(rating_results, task="rating")
        save_comparison_csv(rating_df, config.rating_comparison_path)

        # Generate charts
        generate_comparison_charts(rating_results, task="rating", output_dir=config.reports_figures_dir)
        generate_residual_plots(rating_results, y_rating.values, task="rating",
                                output_dir=config.reports_figures_dir)

        # Select best model name
        best_name = select_best_model(rating_results, metric=config.rating_metric, task="rating")
        logger.info("🏆 Best rating model: %s", best_name)

        # Fit on full data and save to disk
        if best_name:
            fitted_rating_model = fit_best_rating_model(
                best_name, X_rating, y_rating, config.best_rating_model_path, config
            )
            logger.info("✅ Best rating model saved to %s", config.best_rating_model_path)
    else:
        logger.error("No rating models were trained successfully!")
        fitted_rating_model = None

    # ================================================================
    # STAGE 3: Delivery ETA Engine (Dataset B)
    # ================================================================
    logger.info("\n=== STAGE 3: Delivery ETA Engine ===")

    logger.info("--- Loading delivery data ---")
    df_delivery = load_delivery(config)
    describe_dataset(df_delivery, "Delivery (raw)")

    logger.info("--- Cleaning delivery data ---")
    df_delivery = clean_delivery(df_delivery, config)
    describe_dataset(df_delivery, "Delivery (cleaned)")

    logger.info("--- Delivery feature engineering ---")
    df_delivery = add_delivery_features(df_delivery, config)

    # Prepare features for ETA model
    eta_feature_cols = [c for c in df_delivery.columns if c in [
        "haversine_distance_km", "traffic_ordinal", "is_festival",
        "delivery_person_age", "delivery_person_ratings", "vehicle_condition",
    ]]
    # Add any one-hot encoded columns
    eta_feature_cols += [c for c in df_delivery.columns if c.startswith("order_hour_bucket_")]

    X_eta = df_delivery[eta_feature_cols].fillna(0)
    y_eta = df_delivery["time_taken_min"]

    logger.info("--- ETA model comparison ---")
    eta_results, eta_best = train_and_evaluate_eta_models(X_eta, y_eta, config)

    if eta_results:
        # Save comparison CSV
        eta_df = comparison_to_dataframe(eta_results, task="eta")
        save_comparison_csv(eta_df, config.eta_comparison_path)

        # Generate charts
        generate_comparison_charts(eta_results, task="eta", output_dir=config.reports_figures_dir)
        generate_residual_plots(eta_results, y_eta.values, task="eta",
                                output_dir=config.reports_figures_dir)

        # Select best model name
        best_eta_name = select_best_model(eta_results, metric=config.eta_metric, task="eta")
        logger.info("🏆 Best ETA model: %s", best_eta_name)

        # Fit on full data and save to disk
        if best_eta_name:
            fitted_eta_model = fit_best_eta_model(
                best_eta_name, X_eta, y_eta, config.best_eta_model_path, config
            )
            logger.info("✅ Best ETA model saved to %s", config.best_eta_model_path)

        # SLA analysis on CV predictions
        if eta_best and eta_best.predictions is not None:
            sla_metrics = compute_sla_analysis(y_eta.values, eta_best.predictions, config=config)
            logger.info("SLA analysis: %s", sla_metrics)
    else:
        logger.error("No ETA models were trained successfully!")
        fitted_eta_model = None

    # ================================================================
    # STAGE 4: Reliability Score
    # ================================================================
    logger.info("\n=== STAGE 4: Reliability Score ===")

    if fitted_rating_model is not None and fitted_eta_model is not None:
        # Compute reliability scores for all restaurants
        # Rating component: use actual ratings normalized
        rating_scores = y_rating.values

        # Sentiment component: use computed sentiment
        sentiment_scores = df_zomato["avg_sentiment"].values if "avg_sentiment" in df_zomato.columns else np.zeros(len(df_zomato))

        # Delay risk: use predicted ETA normalized (higher = more risk)
        # We need to predict ETA for each restaurant — use mean features as proxy
        # In production, this would use actual delivery data per restaurant
        delay_risk = np.full(len(df_zomato), config.sla_threshold_minutes * 0.5)  # placeholder

        reliability = compute_reliability_score(rating_scores, sentiment_scores, delay_risk, config)
        df_zomato["reliability_score"] = reliability

        # Save with reliability scores
        df_zomato.to_csv(processed_path, index=False)
        logger.info("Reliability scores computed and saved — mean=%.3f", float(np.mean(reliability)))
    else:
        logger.warning("Skipping reliability score — models not available")

    # ================================================================
    # STAGE 2.6: Geographic Clustering (bonus)
    # ================================================================
    logger.info("\n=== Geographic Clustering ===")

    # Use restaurant coordinates if available
    lat_col = None
    lon_col = None
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
            logger.info("Comparing clustering methods on %d restaurants...", len(coords))
            cluster_results = compare_clustering_methods(coords, k_range=range(3, 11))
            for method, info in cluster_results.items():
                logger.info("  %s: silhouette=%.3f", method, info.get("silhouette_score", -1))
    else:
        logger.info("No valid lat/long coordinates found — skipping clustering")

    logger.info("\n=== Pipeline Complete ===")
    logger.info("Best rating model saved to: %s", config.best_rating_model_path)
    logger.info("Best ETA model saved to: %s", config.best_eta_model_path)
    logger.info("Comparison CSVs saved to: %s", config.reports_dir)
    logger.info("Processed data saved to: %s", processed_path)


if __name__ == "__main__":
    main()
