# ML Membership Inference Audit

An offline, reproducible privacy-risk experiment for tabular classifiers. It trains shadow models, learns a small attack classifier from model confidence/loss signals, then measures how well that attack distinguishes records used to fit a target model from separate nonmember records.

This demonstrates a standard membership-inference evaluation pattern; it is not a privacy certification, a production security product, or a guarantee that a model does (or does not) reveal training membership. Use only data you are authorized to process. The bundled demo uses synthetic data and makes no network calls.

## What it measures

- Attack ROC-AUC and true-positive rate at 1% and 5% false-positive rates.
- Maximum membership advantage (`TPR - FPR`) across the ROC curve.
- Target model accuracy and mean target-model loss for members versus nonmembers.
- Reproducible, disjoint target-member, target-nonmember, and shadow-data splits.

The attack features include true-label confidence, maximum confidence, entropy, per-row loss, prediction margin, and correctness. A balanced logistic-regression attack is trained on in/out predictions from independently fitted shadow-model clones. The result is an empirical attack under this particular split, model, dataset and attack design—not a universal estimate of privacy risk.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
miaudit --samples 1200 --shadows 5
pytest
ruff check .
```

For machine-readable output:

```bash
miaudit --samples 1200 --shadows 5 --seed 17 --json
```

## Python API

Pass a scikit-learn-style probabilistic classifier (it must implement `fit`, `predict`, `predict_proba`, and `classes_`). The function clones it for the target and shadow fits, so it does not modify the supplied estimator.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from miaudit import audit_membership

X, y = make_classification(n_samples=1200, n_features=12, random_state=17)
result = audit_membership(X, y, LogisticRegression(max_iter=1000), n_shadow=5, seed=17)
print(result.to_dict())
```

The function raises `ValueError` for malformed arrays, insufficient class support, invalid shadow counts, or classifiers without probability output. Larger datasets and more shadow models generally cost more time and memory. No raw data is persisted by this library.

## Method and limitations

The dataset is split into disjoint target training, target nonmember evaluation, and shadow-model pools. Every shadow model gets a stratified in/out split; its outputs train the attack classifier. The target model is evaluated on disjoint member and nonmember rows. The internal splits are randomized and seeded.

Results depend strongly on sample size, class balance, model capacity, attack features, shadow count and split randomness. A low attack score does not establish privacy; a high score does not establish real-world exploitability. This demo does not implement differential privacy, black-box query limits, formal privacy accounting, neural-network shadow architectures, or a production data pipeline. It supports ordinary binary or multiclass classifiers when every split has adequate examples per class.

## Design

```text
Input arrays -> stratified disjoint splits
                    | target fit -> target member/nonmember signals -> report
                    | shadow fits -> member/nonmember signals -> attack classifier
```

## License

MIT
