import json

import pytest

from neuropace.config import EEGConfig

pytest.importorskip("tensorflow")

from neuropace.train import train  # noqa: E402


def test_one_epoch_writes_model_and_manifest(tmp_path):
    """A real end-to-end run, sized to finish in seconds."""
    config = EEGConfig(
        sampling_rate=25,
        channels=2,
        recording_duration=2.0,
        window_size=1.0,
        window_overlap=0.5,
        batch_size=4,
    )
    manifest = train(
        config,
        n_per_condition=10,
        epochs=1,
        model_type="cnn",
        seed=0,
        out_dir=tmp_path,
    )

    assert (tmp_path / "model.keras").exists()
    assert (tmp_path / "run.json").exists()
    assert (tmp_path / "history.json").exists()

    assert manifest["seed"] == 0
    assert manifest["n_windows"]["train"] > 0
    assert manifest["n_windows"]["val"] > 0
    assert manifest["n_windows"]["test"] > 0

    on_disk = json.loads((tmp_path / "run.json").read_text())
    assert on_disk["seed"] == 0
    assert on_disk["config"]["sampling_rate"] == 25
    assert "tensorflow" in on_disk["versions"]


def test_importing_train_does_not_train():
    """pipeline.py and pp.py started a 75-epoch run on import."""
    import neuropace.train as module

    assert callable(module.train)
    assert callable(module.main)
