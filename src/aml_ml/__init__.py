"""AML preprocessing and limited SCRUM-130 model-scoring candidate."""

from .api import AlertInput as LegacyAlertInput
from .api import features as legacy_features
from .serving_candidate import (
    CONTEXT_LIMIT,
    FEATURE_NAMES,
    FEATURE_VERSION,
    MODEL_VERSION,
    THRESHOLD,
    AlertInput,
    Coverage,
    ScoreResult,
    load_candidate,
    score,
)
from .serving_candidate import features as _candidate_features

__version__ = "0.1.0rc2"


def features(alerts, transactions, *, coverage=None, **legacy_options):
    """Use candidate coverage contract, or retain the existing preprocessing API."""
    if coverage is not None:
        if legacy_options:
            raise TypeError("legacy options are not valid with coverage")
        return _candidate_features(alerts, transactions, coverage=coverage)
    return legacy_features(alerts, transactions, **legacy_options)


__all__ = [
    "AlertInput",
    "LegacyAlertInput",
    "Coverage",
    "ScoreResult",
    "MODEL_VERSION",
    "FEATURE_VERSION",
    "THRESHOLD",
    "CONTEXT_LIMIT",
    "FEATURE_NAMES",
    "score",
    "features",
    "load_candidate",
]
