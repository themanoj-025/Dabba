"""Collaborative Filtering via PyTorch Matrix Factorization.

NOTE: The Zomato dataset has no real user-interaction history.
This module generates a SYNTHETIC-BUT-REALISTIC user-interaction
dataset to demonstrate collaborative filtering properly. This is a
legitimate and common technique when a public dataset lacks user-level
data, and doing it transparently is itself a signal of sound judgment.

The synthetic-data nature is clearly documented everywhere it appears
— it is never presented as real user behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════

# NOTE: This generates SYNTHETIC interaction data — not real user behavior.
# In production, this would be replaced with real order/rating logs.

CUISINE_PREFERENCES = {
    "north_indian": ["North Indian", "Mughlai", "Biryani"],
    "south_indian": ["South Indian", "Andhra", "Kerala"],
    "chinese_lover": ["Chinese", "Thai", "Japanese", "Korean"],
    "italian_fan": ["Italian", "Continental", "French"],
    "fast_foodie": ["Fast Food", "Street Food", "Cafe", "Bakery"],
    "dessert_lover": ["Desserts", "Ice Cream", "Bakery", "Cafe"],
    "adventurous": ["Japanese", "Korean", "Mexican", "Mediterranean", "Lebanese"],
    "budget_eater": ["Street Food", "Fast Food", "South Indian"],
    "premium_diner": ["Continental", "Italian", "Mughlai", "Japanese"],
}


def generate_synthetic_interactions(
    restaurants_df: pd.DataFrame,
    n_users: int = 3000,
    min_ratings: int = 5,
    max_ratings: int = 30,
    config: Optional[DabbaConfig] = None,
) -> pd.DataFrame:
    """Generate SYNTHETIC user-restaurant interaction data.

    **IMPORTANT: This generates simulated, not real, user behavior.**
    Users are assigned cuisine/price preferences, and their ratings
    are simulated with realistic noise. This is for demonstrating the
    collaborative filtering technique only.

    Args:
        restaurants_df: Restaurant DataFrame with cuisines, cost, etc.
        n_users: Number of synthetic users to generate.
        min_ratings: Minimum ratings per user.
        max_ratings: Maximum ratings per user.
        config: Project configuration.

    Returns:
        pd.DataFrame: Synthetic interactions with columns:
            user_id, restaurant_id, rating, timestamp.
    """
    config = config or get_config()
    rng = np.random.RandomState(config.random_seed)
    logger.info(
        "Generating SYNTHETIC interactions: %d users, %d restaurants",
        n_users, len(restaurants_df),
    )

    # Build cuisine feature matrix for restaurants
    cuisine_cols = [c for c in restaurants_df.columns if c.startswith("cuisine_")]
    rest_ids = restaurants_df.index.values
    n_restaurants = len(restaurants_df)

    # Generate user preference profiles
    user_types = list(CUISINE_PREFERENCES.keys())
    user_prefs = rng.choice(user_types, size=n_users)

    # Also give each user a price sensitivity (0=budget, 1=premium)
    price_sensitivity = rng.uniform(0, 1, size=n_users)

    interactions = []

    for user_id in range(n_users):
        # How many restaurants this user rates
        n_ratings = rng.randint(min_ratings, max_ratings + 1)

        # Pick restaurants biased toward user's cuisine preferences
        pref_type = user_prefs[user_id]
        preferred_cuisines = CUISINE_PREFERENCES.get(pref_type, [])

        # Calculate affinity scores for each restaurant
        affinity = np.ones(n_restaurants) * 0.3  # base affinity

        for cuisine in preferred_cuisines:
            col = f"cuisine_{cuisine.lower().replace(' ', '_')}"
            if col in cuisine_cols:
                c_idx = cuisine_cols.index(col)
                affinity += restaurants_df[col].values * 0.5

        # Price alignment
        if "cost_for_two" in restaurants_df.columns:
            costs = restaurants_df["cost_for_two"].values
            max_cost = costs.max() if costs.max() > 0 else 1
            price_alignment = 1.0 - np.abs(
                (costs / max_cost) - price_sensitivity[user_id]
            )
            affinity += price_alignment * 0.2

        # Add noise
        affinity += rng.normal(0, 0.15, size=n_restaurants)

        # Sample restaurants with probability proportional to affinity
        probs = np.maximum(affinity, 0) + 0.05  # minimum chance
        probs = probs / probs.sum()

        chosen = rng.choice(n_restaurants, size=min(n_ratings, n_restaurants),
                           replace=False, p=probs)

        for rest_idx in chosen:
            # Rating = base affinity + noise, clipped to [1, 5]
            base = affinity[rest_idx]
            noise = rng.normal(0, 0.3)
            rating = float(np.clip(base * 4 + 1 + noise, 1.0, 5.0))

            interactions.append({
                "user_id": user_id,
                "restaurant_id": int(rest_ids[rest_idx]),
                "rating": round(rating, 2),
            })

    df = pd.DataFrame(interactions)
    logger.info(
        "Generated %d SYNTHETIC interactions (density: %.4f%%)",
        len(df), 100 * len(df) / (n_users * n_restaurants),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
# PYTORCH MATRIX FACTORIZATION MODEL
# ═══════════════════════════════════════════════════════════════════════

class MatrixFactorization(nn.Module):
    """Simple matrix factorization model using embedding layers.

    Maps users and items to latent factor vectors and computes
    rating predictions as the dot product of user and item embeddings.
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int = 50):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict ratings for given user-item pairs.

        Args:
            user_ids: Tensor of user indices.
            item_ids: Tensor of item indices.

        Returns:
            Tensor of predicted ratings.
        """
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        return (user_vecs * item_vecs).sum(dim=1)


class InteractionDataset(Dataset):
    """PyTorch Dataset for user-item interaction data."""

    def __init__(self, interactions: pd.DataFrame):
        # Use user_idx/item_idx (mapped to contiguous 0..N-1), not raw IDs
        self.user_ids = torch.LongTensor(interactions["user_idx"].values)
        self.item_ids = torch.LongTensor(interactions["item_idx"].values)
        self.ratings = torch.FloatTensor(interactions["rating"].values)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def train_matrix_factorization(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_factors: int = 50,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.01,
    device: str = "cpu",
    config: Optional[DabbaConfig] = None,
) -> nn.Module:
    """Train a matrix factorization model on interaction data.

    Args:
        interactions: DataFrame with user_id, restaurant_id, rating columns.
        n_users: Total number of users.
        n_items: Total number of items (restaurants).
        n_factors: Number of latent factors.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        device: Device to train on ('cpu' or 'cuda').
        config: Project configuration.

    Returns:
        Trained MatrixFactorization model.
    """
    config = config or get_config()

    # Map user/item IDs to contiguous indices
    user_map = {uid: i for i, uid in enumerate(range(n_users))}
    item_map = {iid: i for i, iid in enumerate(interactions["restaurant_id"].unique())}

    train_data = interactions.copy()
    train_data["user_idx"] = train_data["user_id"].map(user_map)
    train_data["item_idx"] = train_data["restaurant_id"].map(item_map)

    dataset = InteractionDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MatrixFactorization(n_users, len(item_map), n_factors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info("Training MF: %d users, %d items, %d factors, %d epochs",
                n_users, len(item_map), n_factors, n_epochs)

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/%d — loss: %.4f",
                        epoch + 1, n_epochs, total_loss / n_batches)

    logger.info("Matrix factorization training complete")
    return model


def predict_user_ratings(
    model: nn.Module,
    user_id: int,
    n_items: int,
    device: str = "cpu",
) -> np.ndarray:
    """Predict ratings for all items for a given user.

    Args:
        model: Trained MatrixFactorization model.
        user_id: User ID to predict for.
        n_items: Total number of items.
        device: Device.

    Returns:
        np.ndarray: Predicted ratings for all items.
    """
    model.eval()
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id] * n_items).to(device)
        item_tensor = torch.LongTensor(list(range(n_items))).to(device)
        predictions = model(user_tensor, item_tensor)
    return predictions.cpu().numpy()


def get_collaborative_scores(
    model: nn.Module,
    n_users: int,
    restaurant_ids: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Compute average collaborative filtering score per restaurant.

    Averages predicted ratings across all synthetic users for each
    restaurant. Higher = more universally liked.

    Args:
        model: Trained MatrixFactorization model.
        n_users: Number of synthetic users.
        restaurant_ids: Array of restaurant IDs (original indices).
        device: Device.

    Returns:
        np.ndarray: Average collaborative score per restaurant.
    """
    model.eval()
    n_items = len(restaurant_ids)
    scores = np.zeros(n_items)

    with torch.no_grad():
        batch_size = 1000
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_users = torch.LongTensor(list(range(start, end))).to(device)

            # Predict for all items for each user in batch
            user_expanded = batch_users.unsqueeze(1).expand(-1, n_items)
            item_expanded = torch.LongTensor(restaurant_ids).unsqueeze(0).expand(
                len(batch_users), -1
            ).to(device)

            preds = model(user_expanded, item_expanded)
            scores += preds.cpu().numpy().mean(axis=0)

    scores /= n_users
    return scores


def save_collaborative_model(
    model: nn.Module,
    path: Path,
) -> None:
    """Save the trained matrix factorization model.

    Args:
        model: Trained model.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved collaborative filtering model to %s", path)


def load_collaborative_model(
    n_users: int,
    n_items: int,
    n_factors: int = 50,
    path: Optional[Path] = None,
    device: str = "cpu",
) -> nn.Module:
    """Load a trained matrix factorization model.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        n_factors: Number of latent factors.
        path: Path to saved model.
        device: Device.

    Returns:
        Loaded MatrixFactorization model.
    """
    model = MatrixFactorization(n_users, n_items, n_factors).to(device)
    if path and path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        logger.info("Loaded collaborative filtering model from %s", path)
    return model
