from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from miaudit import audit_membership


def _data(n: int = 420) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=9,
    )


def test_audit_returns_bounded_metrics_and_disjoint_target_counts() -> None:
    X, y = _data()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    result = audit_membership(X, y, model, n_shadow=3, seed=4)

    assert 0.0 <= result.attack_auc <= 1.0
    assert 0.0 <= result.tpr_at_1pct_fpr <= 1.0
    assert 0.0 <= result.tpr_at_5pct_fpr <= 1.0
    assert 0.0 <= result.max_membership_advantage <= 1.0
    assert result.target_member_count + result.target_nonmember_count < len(y)
    assert result.shadow_model_count == 3
    assert result.to_dict()["seed"] == 4


def test_seed_makes_split_and_result_reproducible() -> None:
    X, y = _data()
    model = LogisticRegression(max_iter=500)
    first = audit_membership(X, y, model, n_shadow=2, seed=21)
    second = audit_membership(X, y, model, n_shadow=2, seed=21)

    assert first == second


def test_invalid_shapes_and_shadow_count_are_rejected() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="same number of rows"):
        audit_membership(X, y[:-1], LogisticRegression())
    with pytest.raises(ValueError, match="at least 2"):
        audit_membership(X, y, LogisticRegression(), n_shadow=1)
    with pytest.raises(ValueError, match="at least 60 rows"):
        audit_membership(X[:40], y[:40], LogisticRegression())
