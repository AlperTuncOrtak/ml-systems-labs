import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [lower, upper] percentiles learned on the training data."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanpercentile(X, self.lower, axis=0)
        self.high_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def add_features(X):
    """Derived features, computed row by row (no information from other rows)."""
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X
