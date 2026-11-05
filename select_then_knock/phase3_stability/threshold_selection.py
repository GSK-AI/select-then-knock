"""The evaluated stability-threshold calibration, not an aggregate guarantee."""

import numpy as np


def suggest_threshold_pfer(target_pfer=1.0, n_features=None, avg_selected=None):
    """Invert the historical calibration expression and clip to [0.51, 0.99].

    Candidate-floor augmentation and multi-split aggregation do not inherit a
    proved PFER or FDR guarantee from this expression.
    """
    if n_features is not None and avg_selected is not None and n_features > 0:
        suggested = 0.5 + (avg_selected**2) / (2 * n_features * target_pfer)
    else:
        suggested = 0.5 + 0.5 / target_pfer
    return float(np.clip(suggested, 0.51, 0.99))
