from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def vectorize(states: List[Dict[str, float]], nodes: List[str]) -> np.ndarray:
    """
    Convert a list of node->value dicts into a 2D array (n_samples x n_nodes),
    with columns ordered by `nodes`.
    """
    X = np.asarray([[state.get(n, 0.0) for n in nodes] for state in states], dtype=float)
    return X

def standardize(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to zero-mean/unit-variance. Returns transformed X and the fitted scaler.
    Use this when combining heterogeneous features or before PCA if scales differ.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def embed_pca(
    X: np.ndarray,
    n_components: Optional[int] = 2,
    whiten: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on X and return (Z, pca) where Z is the low-dim embedding.
    - n_components: int or None. If None, keep all components.
    - whiten: decorrelate and scale PCs to unit variance (can help some models).
    """
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca

def explained_variance_report(pca: PCA) -> Dict[str, np.ndarray]:
    """
    Convenience: return explained variance ratios and cumulative curve.
    """
    evr = np.asarray(pca.explained_variance_ratio_)
    cum = np.cumsum(evr)
    return {"explained_variance_ratio": evr, "cumulative": cum}
