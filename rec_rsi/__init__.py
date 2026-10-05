"""RecRSI: recursive recommendation experiments with relational state retention."""

from .evaluation import CROSS_FAMILY, WITHIN_FAMILY, cross_generation_advantage, rank_separation
from .fusion import fuse_log_probabilities

__all__ = [
    "CROSS_FAMILY",
    "WITHIN_FAMILY",
    "cross_generation_advantage",
    "rank_separation",
    "fuse_log_probabilities",
]

__version__ = "0.1.0"
