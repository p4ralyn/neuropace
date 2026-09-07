"""Classical baseline: gradient-boosted trees on spectral features.

A deep model's accuracy means very little without something simpler to beat.
This is that reference point: hand-built band-power features into XGBoost,
scored with the same metrics evaluate() reports so the two are comparable
side by side.

Replaces xgb.py, xgb2.py and smt.py. smt.py trained on np.random.rand and
asserted nothing; xgb.py used mean and std only; xgb2.py imported optuna and
shap without using either and hardcoded band edges that had already drifted
from the generator's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import AILMENT_CLASSES, MOOD_STATES, STRESS_LEVELS, EEGConfig
from .dataset import Recording, make_windows, split_indices, subset
from .features import FEATURE_NAMES, extract_features

HEAD_ATTRIBUTES = {
    "ailment": ("condition_id", AILMENT_CLASSES),
    "stress": ("stress_level_id", STRESS_LEVELS),
    "mood": ("mood_state_id", MOOD_STATES),
}


def featurize(
    recordings: Sequence[Recording], config: EEGConfig, head: str
) -> tuple[np.ndarray, np.ndarray]:
    """Windows -> (features, labels) for one head."""
    attribute, _ = HEAD_ATTRIBUTES[head]

    features, labels = [], []
    for recording in recordings:
        windows = make_windows(recording, config)
        if windows.shape[0] == 0:
            continue
        features.append(extract_features(windows, config.sampling_rate))
        labels.append(np.full(windows.shape[0], getattr(recording, attribute)))

    return np.concatenate(features), np.concatenate(labels)


def run_baseline(
    recordings: Sequence[Recording],
    config: EEGConfig,
    *,
    head: str = "ailment",
    seed: int = 42,
) -> dict:
    """Fit XGBoost on the training split and score the held-out split.

    Returns the same {accuracy, macro_f1, ...} shape evaluate() returns.
    """
    from xgboost import XGBClassifier

    if head not in HEAD_ATTRIBUTES:
        raise ValueError(f"unknown head {head!r}; expected {list(HEAD_ATTRIBUTES)}")

    train_idx, _, test_idx = split_indices(recordings, seed=seed)
    x_train, y_train = featurize(subset(recordings, train_idx), config, head)
    x_test, y_test = featurize(subset(recordings, test_idx), config, head)

    _, labels = HEAD_ATTRIBUTES[head]
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        objective="multi:softprob",
        num_class=len(labels),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    importances = dict(
        zip(FEATURE_NAMES, (float(v) for v in model.feature_importances_), strict=True)
    )

    return {
        "head": head,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(
            f1_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "support": int(len(y_test)),
        "n_features": len(FEATURE_NAMES),
        "feature_importance": dict(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuropace-baseline",
        description="Fit an XGBoost baseline on spectral features.",
    )
    parser.add_argument("--head", choices=tuple(HEAD_ATTRIBUTES), default="ailment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recordings-per-condition", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    from .synthetic import generate_dataset

    config = EEGConfig()
    recordings = generate_dataset(
        config,
        n_per_condition=args.recordings_per_condition,
        rng=np.random.default_rng(args.seed),
    )
    results = run_baseline(recordings, config, head=args.head, seed=args.seed)

    print(f"baseline · {results['head']}")
    print(f"  accuracy  {results['accuracy']:.4f}")
    print(f"  macro-F1  {results['macro_f1']:.4f}")
    print(f"  windows   {results['support']}")
    print("\n  top features:")
    for name, value in list(results["feature_importance"].items())[:5]:
        print(f"    {name:<20}{value:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"baseline_{args.head}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
