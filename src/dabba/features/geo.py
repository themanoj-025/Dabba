"""Geospatial utility functions for the Dabba project.

Provides haversine distance calculation and geographic clustering.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Earth's mean radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula to compute distance in kilometers.

    Args:
        lat1: Latitude(s) of point 1 in decimal degrees.
        lon1: Longitude(s) of point 1 in decimal degrees.
        lat2: Latitude(s) of point 2 in decimal degrees.
        lon2: Longitude(s) of point 2 in decimal degrees.

    Returns:
        Distance(s) in kilometers. Scalar if inputs are scalar, array otherwise.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


# Bangalore neighborhood centroids (approximate)
BANGALORE_CENTROIDS: dict[str, Tuple[float, float]] = {
    "Koramangala": (12.9352, 77.6245),
    "Indiranagar": (12.9784, 77.6408),
    "HSR Layout": (12.9116, 77.6389),
    "Whitefield": (12.9698, 77.7500),
    "Electronic City": (12.8456, 77.6602),
    "JP Nagar": (12.8911, 77.5851),
    "BTM Layout": (12.9166, 77.6101),
    "Marathahalli": (12.9562, 77.7011),
    "Jayanagar": (12.9293, 77.5841),
    "MG Road": (12.9758, 77.6062),
    "HSR": (12.9116, 77.6389),
    "Bannerghatta Road": (12.8880, 77.6000),
    "Rajajinagar": (12.9878, 77.5544),
    "Vijayanagar": (12.9735, 77.5425),
    "Basavanagudi": (12.9408, 77.5722),
    "Malleshwaram": (13.0034, 77.5651),
    "Yelahanka": (13.1076, 77.5659),
    "Hebbal": (13.0358, 77.5970),
    "Sarjapur Road": (12.9010, 77.6870),
    "Old Airport Road": (12.9604, 77.6464),
}


def geocode_location(location: str) -> Tuple[float, float] | None:
    """Look up approximate centroid for a Bangalore neighborhood.

    Args:
        location: Neighborhood name.

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    # Try exact match first, then fuzzy
    loc_lower = location.strip().lower()
    for name, coords in BANGALORE_CENTROIDS.items():
        if name.lower() == loc_lower:
            return coords
    for name, coords in BANGALORE_CENTROIDS.items():
        if name.lower() in loc_lower or loc_lower in name.lower():
            return coords
    return None


def compare_clustering_methods(
    X: np.ndarray,
    k_range: range = range(2, 11),
) -> dict[str, dict]:
    """Compare KMeans, DBSCAN, and Agglomerative clustering via silhouette score.

    Args:
        X: Feature array (e.g., lat/long coordinates).
        k_range: Range of cluster counts to test for KMeans and Agglomerative.

    Returns:
        Dict mapping method name to dict with 'best_k'/'eps' and 'silhouette_score'.
    """
    results: dict[str, dict] = {}

    # KMeans
    best_score = -1
    best_k = 2
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    results["KMeans"] = {"best_k": best_k, "silhouette_score": best_score}

    # DBSCAN
    for eps in [0.01, 0.02, 0.05, 0.1]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 1:
                score = silhouette_score(X[mask], labels[mask])
                if score > best_score:
                    results["DBSCAN"] = {"eps": eps, "silhouette_score": score}
                    best_score = score

    # Agglomerative
    best_score = -1
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    results["Agglomerative"] = {"best_k": best_k, "silhouette_score": best_score}

    return results
