"""RAG Similar-Restaurant Retrieval.

Embeds restaurant feature vectors (cuisine, price tier, location,
sentiment) into a local FAISS index for fast approximate nearest
neighbor search. For any restaurant, retrieves the 3-5 most similar
alternatives.

Graceful fallback: returns cosine-similarity-based results from
scikit-learn if FAISS index is not available.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ─── FAISS index (lazy) ─────────────────────────────────────────────────

_faiss_index = None


def _build_faiss_index(
    embeddings: np.ndarray,
    config: DabbaConfig,
) -> None:
    """Build and save a FAISS index from restaurant embeddings."""
    global _faiss_index
    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(config.faiss_index_path))
        _faiss_index = index
        logger.info(
            "FAISS index built with %d embeddings (dim=%d)", len(embeddings), dim
        )
    except ImportError:
        logger.warning("FAISS not available — using sklearn fallback")


def _load_faiss_index(config: DabbaConfig):
    """Load the FAISS index from disk."""
    global _faiss_index
    if _faiss_index is not None:
        return _faiss_index
    try:
        import faiss

        index_path = config.faiss_index_path
        if index_path.exists():
            _faiss_index = faiss.read_index(str(index_path))
            logger.info("Loaded FAISS index from %s", index_path)
            return _faiss_index
    except ImportError:
        pass
    return None


def build_restaurant_embeddings(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: Optional[DabbaConfig] = None,
) -> np.ndarray:
    """Build normalized restaurant feature embeddings and save to disk.

    Args:
        df: Restaurant DataFrame with feature columns.
        feature_cols: Numeric feature columns to embed.
        config: Project configuration.

    Returns:
        np.ndarray: Normalized embedding matrix (n_restaurants x n_features).
    """
    config = config or get_config()
    existing_cols = [c for c in feature_cols if c in df.columns]
    if not existing_cols:
        logger.warning("No feature columns found for embeddings")
        return np.zeros((len(df), 1))

    embeddings = df[existing_cols].fillna(0).values.astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Save
    config.models_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(config.restaurant_embeddings_path), embeddings)
    logger.info(
        "Saved restaurant embeddings to %s (shape=%s)",
        config.restaurant_embeddings_path,
        embeddings.shape,
    )

    # Build FAISS index
    _build_faiss_index(embeddings, config)

    return embeddings


def find_similar_restaurants(
    restaurant_idx: int,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
    config: Optional[DabbaConfig] = None,
) -> pd.DataFrame:
    """Find the top-k most similar restaurants to a given restaurant.

    Args:
        restaurant_idx: Index of the query restaurant in the DataFrame.
        df: Full restaurant DataFrame.
        embeddings: Restaurant embedding matrix.
        top_k: Number of similar restaurants to return.
        config: Project configuration.

    Returns:
        pd.DataFrame: Top-K similar restaurants with similarity scores.
    """
    config = config or get_config()

    query_vec = embeddings[restaurant_idx : restaurant_idx + 1]

    # Try FAISS first
    index = _load_faiss_index(config)
    if index is not None:
        distances, indices = index.search(query_vec.astype(np.float32), top_k + 1)
        # First result is the restaurant itself
        sim_indices = indices[0][1 : top_k + 1]
        sim_scores = 1.0 / (
            1.0 + distances[0][1 : top_k + 1]
        )  # convert L2 to similarity
    else:
        # sklearn fallback
        sim_matrix = cosine_similarity(query_vec, embeddings)[0]
        sim_indices = np.argsort(sim_matrix)[::-1][1 : top_k + 1]
        sim_scores = sim_matrix[sim_indices]

    results = df.iloc[sim_indices].copy()
    results["similarity_score"] = [round(float(s), 3) for s in sim_scores]

    display_cols = [
        "name",
        "rate",
        "cost_for_two",
        "location",
        "cuisines",
        "similarity_score",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    return results[display_cols].reset_index(drop=True)
