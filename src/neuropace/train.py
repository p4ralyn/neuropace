"""Training entry point.

Replaces pipeline.py and pp.py, which were top-level scripts: importing either
started a 75-epoch run. They also disagreed with each other on learning rate,
epochs, and augmentation, and both recompiled a model that had already been
compiled by its builder.

Nothing here runs on import.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import EEGConfig
from .dataset import split_indices, subset
from .seeding import seed_everything


def _versions() -> dict[str, str]:
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
    }


def train(
    config: EEGConfig,
    *,
    n_per_condition: int = 50,
    epochs: int = 50,
    model_type: str = "hybrid",
    learning_rate: float = 1e-3,
    seed: int = 42,
    out_dir: Path | str = "artifacts",
) -> dict:
    """Generate data, train, and write the model beside its run manifest.

    Returns the manifest. Augmentation and shuffling apply to the training
    split only; validation and test stay untouched so their numbers mean
    something.
    """
    import tensorflow as tf

    from .data import make_dataset, windows_and_labels
    from .models import build_model

    seed_everything(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from .synthetic import generate_dataset

    recordings = generate_dataset(
        config, n_per_condition=n_per_condition, rng=np.random.default_rng(seed)
    )
    train_idx, val_idx, test_idx = split_indices(recordings, seed=seed)
    splits = {
        "train": subset(recordings, train_idx),
        "val": subset(recordings, val_idx),
        "test": subset(recordings, test_idx),
    }

    datasets = {
        "train": make_dataset(
            splits["train"], config, augment_data=True, shuffle=True, seed=seed
        ),
        "val": make_dataset(splits["val"], config),
        "test": make_dataset(splits["test"], config),
    }
    n_windows = {
        name: int(windows_and_labels(recs, config)[0].shape[0])
        for name, recs in splits.items()
    }

    model = build_model(config, model_type=model_type, learning_rate=learning_rate)

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epochs,
        # The pipeline already shuffles; Keras would ignore this anyway.
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(out / "model.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
        ],
        verbose=2,
    )

    # EarlyStopping may restore weights better than the last checkpoint wrote.
    model.save(out / "model.keras")

    manifest = {
        "seed": seed,
        "config": config.to_dict(),
        "model_type": model_type,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "n_recordings": {name: len(recs) for name, recs in splits.items()},
        "n_windows": n_windows,
        "n_per_condition": n_per_condition,
        "versions": _versions(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    (out / "history.json").write_text(
        json.dumps(
            {k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2
        )
    )
    config.save(out / "config.json")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuropace-train",
        description="Train the multi-head EEG classifier on synthetic data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--recordings-per-condition", type=int, default=50)
    parser.add_argument(
        "--model-type", choices=("cnn", "lstm", "hybrid"), default="hybrid"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = EEGConfig(
        sampling_rate=args.sampling_rate,
        channels=args.channels,
        recording_duration=args.duration,
        window_size=args.window,
        window_overlap=args.overlap,
        batch_size=args.batch_size,
    )

    manifest = train(
        config,
        n_per_condition=args.recordings_per_condition,
        epochs=args.epochs,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        seed=args.seed,
        out_dir=args.out,
    )

    print(f"\nWrote {args.out}/model.keras")
    print(f"  windows: {manifest['n_windows']}")
    print(f"  epochs run: {manifest['epochs_run']}")
    print(f"\nEvaluate it: python -m neuropace.evaluate --model {args.out}/model.keras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
