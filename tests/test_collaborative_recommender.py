"""Tests for collaborative filtering module."""

import numpy as np
import pandas as pd
import pytest
import torch

from dabba.models.collaborative_recommender import (
    MatrixFactorization,
    InteractionDataset,
    generate_synthetic_interactions,
)


# ─── Fixtures (module-level, required by pytest) ──────────────────────

@pytest.fixture
def sample_restaurants():
    """Create a small sample restaurant DataFrame."""
    rng = np.random.RandomState(42)
    n = 20
    cols = {"name": [f"Rest_{i}" for i in range(n)]}
    cols["cost_for_two"] = rng.randint(100, 2000, n)
    cols["cuisines"] = rng.choice(["North Indian", "Chinese", "Italian", "South Indian"], n)
    for cuisine in ["north_indian", "chinese", "italian", "south_indian"]:
        cols[f"cuisine_{cuisine}"] = rng.randint(0, 2, n)
    return pd.DataFrame(cols)


# ─── Tests ────────────────────────────────────────────────────────────

class TestSyntheticDataGenerator:
    """Tests for synthetic interaction data generation."""

    def test_generates_dataframe(self, sample_restaurants):
        """Should generate a DataFrame with user_id, restaurant_id, rating."""
        df = generate_synthetic_interactions(sample_restaurants, n_users=50)
        assert isinstance(df, pd.DataFrame)
        assert "user_id" in df.columns
        assert "restaurant_id" in df.columns
        assert "rating" in df.columns
        assert len(df) > 0

    def test_ratings_in_range(self, sample_restaurants):
        """Ratings should be in [1, 5] range."""
        df = generate_synthetic_interactions(sample_restaurants, n_users=50)
        assert df["rating"].min() >= 1.0
        assert df["rating"].max() <= 5.0


class TestMatrixFactorization:
    """Tests for PyTorch matrix factorization model."""

    def test_forward_shape(self):
        """Forward pass should return correct shape."""
        model = MatrixFactorization(n_users=10, n_items=20, n_factors=8)
        user_ids = torch.LongTensor([0, 1, 2])
        item_ids = torch.LongTensor([0, 1, 2])
        preds = model(user_ids, item_ids)
        assert preds.shape == (3,)

    def test_training_loop(self):
        """Training should reduce loss."""
        model = MatrixFactorization(n_users=10, n_items=20, n_factors=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        users = torch.LongTensor([0, 1, 2, 3, 4])
        items = torch.LongTensor([0, 1, 2, 3, 4])
        targets = torch.FloatTensor([5.0, 4.0, 3.0, 2.0, 1.0])

        loss_before = criterion(model(users, items), targets).item()
        for _ in range(50):
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        loss_after = criterion(model(users, items), targets).item()
        assert loss_after < loss_before


class TestInteractionDataset:
    """Tests for the PyTorch Dataset."""

    def test_length(self):
        """Dataset length should match number of interactions."""
        df = pd.DataFrame({
            "user_idx": [0, 1, 2],
            "item_idx": [0, 1, 2],
            "rating": [4.0, 3.5, 5.0],
        })
        dataset = InteractionDataset(df)
        assert len(dataset) == 3

    def test_getitem(self):
        """Dataset should return user, item, rating tensors."""
        df = pd.DataFrame({
            "user_idx": [0],
            "item_idx": [5],
            "rating": [4.5],
        })
        dataset = InteractionDataset(df)
        user, item, rating = dataset[0]
        assert user.item() == 0
        assert item.item() == 5
        assert rating.item() == 4.5
