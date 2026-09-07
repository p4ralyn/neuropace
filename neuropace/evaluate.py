"""Evaluation: per-head metrics, confusion matrices, and a metrics manifest.

The scripts this replaces printed results[1], results[2] and results[3] off
model.evaluate positionally and called them accuracy, with no per-class
breakdown and no confusion matrix.

Macro-F1 sits next to accuracy throughout because the heads are wildly
imbalanced in difficulty: mood has ten classes drawn uniformly at random, so a
model that has learned nothing but the marginal distribution still scores a
respectable-looking accuracy. Macro-F1 does not flatter it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import AILMENT_CLASSES, MOOD_STATES, STRESS_LEVELS, EEGConfig
from .dataset import Recording

HEAD_LABELS = {
    "ailment": AILMENT_CLASSES,
    "stress": STRESS_LEVELS,
    "mood": MOOD_STATES,
}


def evaluate(
    model, recordings: Sequence[Recording], config: EEGConfig
) -> dict[str, dict]:
    """Score a model on the given recordings, per head.

    Returns {head: {accuracy, macro_f1, report, confusion_matrix, support}}.
    """
    from .data import make_dataset, windows_and_labels

    _, ailment, stress, mood = windows_and_labels(recordings, config)
    truth = {"ailment": ailment, "stress": stress, "mood": mood}

    dataset = make_dataset(recordings, config, augment_data=False, shuffle=False)
    predictions = model.predict(dataset, verbose=0)

    results: dict[str, dict] = {}
    for head, labels in HEAD_LABELS.items():
        probabilities = (
            predictions[head] if isinstance(predictions, dict) else predictions[head]
        )
        predicted = np.argmax(probabilities, axis=1)
        actual = truth[head]

        results[head] = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(
                f1_score(actual, predicted, average="macro", zero_division=0)
            ),
            "report": classification_report(
                actual,
                predicted,
                labels=list(range(len(labels))),
                target_names=list(labels),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                actual, predicted, labels=list(range(len(labels)))
            ).tolist(),
            "support": int(len(actual)),
        }

    return results


def save_confusion_matrices(
    results: dict[str, dict], config: EEGConfig, out_dir: Path | str
) -> list[Path]:
    """Write one confusion-matrix PNG per head. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")  # no display in CI
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for head, labels in HEAD_LABELS.items():
        matrix = np.array(results[head]["confusion_matrix"], dtype=float)
        # Row-normalize so a dominant class does not wash out the picture.
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, totals, out=np.zeros_like(matrix), where=totals > 0
        )

        size = max(4.0, 0.6 * len(labels))
        figure, axes = plt.subplots(figsize=(size + 2, size))
        image = axes.imshow(normalized, cmap="magma", vmin=0, vmax=1)

        axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axes.set_yticks(range(len(labels)), labels)
        axes.set_xlabel("predicted")
        axes.set_ylabel("actual")
        axes.set_title(
            f"{head}  ·  acc {results[head]['accuracy']:.3f}  ·  "
            f"macro-F1 {results[head]['macro_f1']:.3f}"
        )
        figure.colorbar(image, ax=axes, fraction=0.046)
        figure.tight_layout()

        path = out / f"confusion_{head}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written


def summarize(results: dict[str, dict]) -> str:
    """A compact table, for stdout and for pasting into a README."""
    lines = [
        f"{'head':<10}{'accuracy':>10}{'macro-F1':>10}{'classes':>9}{'windows':>9}",
        "-" * 48,
    ]
    for head, scores in results.items():
        lines.append(
            f"{head:<10}{scores['accuracy']:>10.4f}{scores['macro_f1']:>10.4f}"
            f"{len(HEAD_LABELS[head]):>9}{scores['support']:>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuropace-eval",
        description="Evaluate a trained model on a held-out synthetic split.",
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.keras"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="defaults to the seed recorded in the run manifest",
    )
    parser.add_argument("--recordings-per-condition", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.model.exists():
        parser.error(f"no model at {args.model}. Train one: python -m neuropace.train")

    import tensorflow as tf

    from .dataset import split_indices, subset
    from .seeding import seed_everything
    from .synthetic import generate_dataset

    # Reuse the training run's seed and config so the held-out split here is
    # the same one the trainer held out. Evaluating on a different split would
    # be meaningless.
    manifest_path = args.model.parent / "run.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    seed = args.seed if args.seed is not None else manifest.get("seed", 42)
    n_per_condition = args.recordings_per_condition or manifest.get(
        "n_per_condition", 50
    )
    config = (
        EEGConfig.from_dict(manifest["config"]) if "config" in manifest else EEGConfig()
    )

    seed_everything(seed)
    recordings = generate_dataset(
        config, n_per_condition=n_per_condition, rng=np.random.default_rng(seed)
    )
    _, _, test_idx = split_indices(recordings, seed=seed)
    held_out = subset(recordings, test_idx)

    model = tf.keras.models.load_model(args.model)
    results = evaluate(model, held_out, config)

    print(summarize(results))
    paths = save_confusion_matrices(results, config, args.out)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))

    print(f"\nWrote {args.out}/metrics.json")
    for path in paths:
        print(f"      {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
