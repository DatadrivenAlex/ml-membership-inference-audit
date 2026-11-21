"""Train shadow models and estimate membership-inference risk on a held-out target model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AuditResult:
    """Summary statistics for one seeded, offline membership-inference experiment."""

    attack_auc: float
    tpr_at_1pct_fpr: float
    tpr_at_5pct_fpr: float
    max_membership_advantage: float
    target_member_count: int
    target_nonmember_count: int
    shadow_model_count: int
    target_accuracy: float
    member_mean_loss: float
    nonmember_mean_loss: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)


def _features(model: ClassifierMixin, X: NDArray[Any], y: NDArray[Any]) -> NDArray[np.float64]:
    """Summarize confidence, loss, entropy, margin and correctness for each labeled row."""
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("classifier returned invalid predict_proba output")
    classes = np.asarray(model.classes_)
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    try:
        true_indices = np.asarray([class_to_index[label] for label in y.tolist()], dtype=int)
    except KeyError as exc:
        raise ValueError("labels contain a class absent from the fitted classifier") from exc
    if probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba columns do not match classifier classes_")

    clipped = np.clip(probabilities, 1e-12, 1.0)
    true_probability = clipped[np.arange(len(y)), true_indices]
    sorted_probabilities = np.sort(clipped, axis=1)
    margin = sorted_probabilities[:, -1] - (
        sorted_probabilities[:, -2] if clipped.shape[1] > 1 else 0
    )
    predictions = classes[np.argmax(probabilities, axis=1)]
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return np.column_stack(
        [
            true_probability,
            np.max(clipped, axis=1),
            entropy,
            -np.log(true_probability),
            margin,
            predictions == y,
        ]
    ).astype(float)


def _stratified_split(
    X: NDArray[Any], y: NDArray[Any], test_size: float, seed: int
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError as exc:
        raise ValueError(
            "not enough examples per class for the requested stratified audit; "
            "provide more rows or reduce the number of shadow models"
        ) from exc


def audit_membership(
    X: ArrayLike,
    y: ArrayLike,
    estimator: ClassifierMixin,
    *,
    n_shadow: int = 5,
    seed: int = 17,
) -> AuditResult:
    """Estimate attack success using disjoint target and shadow-model data.

    The input is split into target training, target nonmember evaluation, and shadow pools.
    Each shadow model trains on a stratified subset of the shadow pool; its held-out rows act
    as nonmembers for training a binary attack classifier. A fresh clone of ``estimator`` is
    fitted to target members and evaluated only on the separate target holdout.

    This is an empirical educational audit, not a formal privacy guarantee or a substitute for
    a threat-model-specific assessment. Input rows should be lawful to process and preferably
    synthetic or de-identified.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2 or y_arr.ndim != 1 or len(X_arr) != len(y_arr):
        raise ValueError("X must be 2D and y must be 1D with the same number of rows")
    if len(X_arr) < max(60, n_shadow * 12) or len(np.unique(y_arr)) < 2:
        raise ValueError("provide at least 60 rows and at least two classes")
    if not isinstance(n_shadow, int) or isinstance(n_shadow, bool) or n_shadow < 2:
        raise ValueError("n_shadow must be an integer of at least 2")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("estimator must support predict_proba")

    # Keep target members, target nonmembers and shadow data disjoint to avoid leakage into
    # the learned attack model. Stratification keeps class support in each partition.
    X_target, X_rest, y_target, y_rest = _stratified_split(X_arr, y_arr, 0.25, seed)
    shadow_fraction_of_rest = 0.60
    X_shadow, X_target_test, y_shadow, y_target_test = _stratified_split(
        X_rest, y_rest, 1.0 - shadow_fraction_of_rest, seed + 1
    )
    target_model = clone(estimator).fit(X_target, y_target)

    attack_rows: list[NDArray[np.float64]] = []
    attack_labels: list[NDArray[np.int64]] = []
    for index in range(n_shadow):
        X_in, X_out, y_in, y_out = _stratified_split(X_shadow, y_shadow, 0.5, seed + 100 + index)
        shadow_model = clone(estimator).fit(X_in, y_in)
        attack_rows.extend(
            [_features(shadow_model, X_in, y_in), _features(shadow_model, X_out, y_out)]
        )
        attack_labels.extend(
            [np.ones(len(y_in), dtype=np.int64), np.zeros(len(y_out), dtype=np.int64)]
        )

    attack_X = np.vstack(attack_rows)
    attack_y = np.concatenate(attack_labels)
    attack_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    attack_model.fit(attack_X, attack_y)

    member_features = _features(target_model, X_target, y_target)
    nonmember_features = _features(target_model, X_target_test, y_target_test)
    evaluation_X = np.vstack([member_features, nonmember_features])
    evaluation_y = np.concatenate(
        [
            np.ones(len(member_features), dtype=np.int64),
            np.zeros(len(nonmember_features), dtype=np.int64),
        ]
    )
    member_scores = attack_model.predict_proba(evaluation_X)[:, 1]
    fpr, tpr, _ = roc_curve(evaluation_y, member_scores)

    def tpr_at(max_fpr: float) -> float:
        return float(np.max(tpr[fpr <= max_fpr])) if np.any(fpr <= max_fpr) else 0.0

    target_predictions = target_model.predict(X_target_test)
    return AuditResult(
        attack_auc=float(auc(fpr, tpr)),
        tpr_at_1pct_fpr=tpr_at(0.01),
        tpr_at_5pct_fpr=tpr_at(0.05),
        max_membership_advantage=float(np.max(tpr - fpr)),
        target_member_count=len(y_target),
        target_nonmember_count=len(y_target_test),
        shadow_model_count=n_shadow,
        target_accuracy=float(np.mean(target_predictions == y_target_test)),
        member_mean_loss=float(np.mean(member_features[:, 3])),
        nonmember_mean_loss=float(np.mean(nonmember_features[:, 3])),
        seed=seed,
    )
