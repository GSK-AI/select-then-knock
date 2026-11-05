"""Selection-only STK implementation for the accompanying research experiments.

Cross-split aggregation has no proved FDR or PFER guarantee.
"""

from .pipeline import SelectThenKnock

__all__ = ["SelectThenKnock"]
