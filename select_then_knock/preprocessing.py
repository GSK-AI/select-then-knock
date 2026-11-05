"""Raw-scale variance filtering and scale-only standardization for regression."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingResults:
    X_processed: np.ndarray
    y_processed: np.ndarray
    feature_mask: np.ndarray
    feature_names: np.ndarray | None
    scaler: StandardScaler
    n_features_removed: int
    removed_feature_indices: np.ndarray
    y_scale: float = 1.0


def preprocess_for_modeling(X, y, task="regression", feature_names=None, variance_threshold=0.0):
    """Keep nonconstant features, divide X and y by their standard deviations.

    No centering, outlier removal, imputation, or sample normalization is performed.
    The feature mask maps Preprocessed indices back to Original indices.
    """
    if task != "regression":
        raise ValueError("Only regression is supported")
    if feature_names is not None:
        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match the number of features")
        feature_names = np.asarray(feature_names)
    selector = VarianceThreshold(threshold=variance_threshold)
    X_proc = selector.fit_transform(X)
    feature_mask = np.asarray(selector.get_support(), dtype=bool)
    if feature_names is not None:
        feature_names = feature_names[feature_mask]
    removed_features = np.where(~feature_mask)[0]
    scaler = StandardScaler(with_mean=False)
    X_proc = scaler.fit_transform(X_proc)
    y_proc = np.asarray(y, dtype=float)
    observed = float(np.std(y_proc))
    y_scale = 1.0
    if observed > 0:
        y_scale = observed
        y_proc = y_proc / y_scale
    return PreprocessingResults(
        X_processed=X_proc,
        y_processed=y_proc,
        feature_mask=feature_mask,
        feature_names=feature_names,
        scaler=scaler,
        n_features_removed=len(removed_features),
        removed_feature_indices=removed_features,
        y_scale=y_scale,
    )


def apply_preprocessing_to_new_data(X_new, preprocessing_results):
    """Apply a screening-fitted feature mask and scaler to held-out rows."""
    return preprocessing_results.scaler.transform(X_new[:, preprocessing_results.feature_mask])
