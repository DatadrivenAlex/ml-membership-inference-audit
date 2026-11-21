"""Command-line entry point for a reproducible synthetic audit."""

from __future__ import annotations

import argparse
import json

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from .audit import audit_membership


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic shadow-model privacy audit.")
    parser.add_argument("--samples", type=int, default=1200, help="synthetic rows (minimum 60)")
    parser.add_argument(
        "--shadows", type=int, default=5, help="number of shadow models (minimum 2)"
    )
    parser.add_argument("--seed", type=int, default=17, help="reproducibility seed")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    if args.samples < 60:
        parser.error("--samples must be at least 60")
    X, y = make_classification(
        n_samples=args.samples,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        class_sep=0.8,
        random_state=args.seed,
    )
    result = audit_membership(
        X,
        y,
        RandomForestClassifier(n_estimators=80, min_samples_leaf=1, random_state=args.seed),
        n_shadow=args.shadows,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    else:
        print("Offline membership-inference audit (synthetic data)")
        print(f"Attack ROC-AUC:                 {result.attack_auc:.3f}")
        print(f"TPR at 1% FPR:                 {result.tpr_at_1pct_fpr:.3f}")
        print(f"TPR at 5% FPR:                 {result.tpr_at_5pct_fpr:.3f}")
        print(f"Maximum membership advantage:  {result.max_membership_advantage:.3f}")
        print(f"Target model accuracy:          {result.target_accuracy:.3f}")
        print(
            f"Target rows (in / out):         {result.target_member_count} / "
            f"{result.target_nonmember_count}"
        )
        print(f"Shadow models:                  {result.shadow_model_count}")
        print("Synthetic benchmark only; scores are not a privacy guarantee.")


if __name__ == "__main__":
    main()
