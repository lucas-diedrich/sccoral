import json

import anndata as ad
import numpy as np
import pytest
from lightning.pytorch.callbacks import Callback
from sccoral.model import SCCORAL
from torch.nn import BatchNorm1d


@pytest.fixture
def model():
    rng = np.random.default_rng(42)
    adata = ad.AnnData(rng.poisson(3, (40, 10)).astype(np.float32))
    adata.obs["condition"] = ["A", "B"] * 20
    SCCORAL.setup_anndata(adata, categorical_covariates="condition")
    return SCCORAL(adata, n_latent=3)


def train(model, **kwargs):
    defaults = {"accelerator": "cpu", "enable_progress_bar": False, "batch_size": 16}
    defaults.update(kwargs)
    model.train(**defaults)


def test_early_stopping_waits_for_joint_training(model, tmp_path):
    path = tmp_path / "training-status.json"
    train(
        model,
        max_epochs=10,
        pretraining_max_epochs=2,
        pretraining_early_stopping=False,
        early_stopping_patience=1,
        early_stopping_min_delta=1e9,
        training_status_path=path,
    )
    assert model.training_status_ == {
        "pretraining_enabled": True,
        "pretraining_completed": True,
        "unfreeze_epoch": 2,
        "pretraining_epochs_completed": 2,
        "joint_epochs_completed": 2,
        "epochs_completed": 4,
        "termination_reason": "early_stopping",
    }
    assert json.loads(path.read_text()) == model.training_status_


def test_pretraining_only_run_warns_and_records_status(model, tmp_path):
    path = tmp_path / "training-status.json"
    with pytest.warns(UserWarning, match="without completing a joint-training epoch"):
        train(
            model, max_epochs=2, pretraining_max_epochs=5, pretraining_early_stopping=False, training_status_path=path
        )
    assert model.training_status_["pretraining_completed"] is False
    assert model.training_status_["unfreeze_epoch"] is None
    assert model.training_status_["joint_epochs_completed"] == 0
    assert model.training_status_["pretraining_epochs_completed"] == 2
    assert model.training_status_["termination_reason"] == "max_epochs"
    assert json.loads(path.read_text()) == model.training_status_
    # A new attempt without pretraining must not inherit the frozen encoder.
    train(model, max_epochs=1, pretraining=False, early_stopping=False)
    assert model.training_status_["joint_epochs_completed"] == 1
    assert model.training_status_["pretraining_enabled"] is False
    assert all(p.requires_grad for p in model.module.z_encoder.parameters())
    assert all(m.track_running_stats for m in model.module.z_encoder.modules() if isinstance(m, BatchNorm1d))


def test_pretraining_convergence_records_actual_unfreeze_epoch(model):
    train(
        model,
        max_epochs=4,
        early_stopping=False,
        pretraining_max_epochs=50,
        pretraining_early_stopping_patience=1,
        pretraining_min_delta=1e9,
    )
    assert model.training_status_["unfreeze_epoch"] == 2
    assert model.training_status_["pretraining_completed"] is True
    assert model.training_status_["joint_epochs_completed"] == 2


def test_status_survives_model_save_and_load(model, tmp_path, monkeypatch):
    train(model, max_epochs=1, pretraining=False, early_stopping=False)
    model.save(tmp_path / "model")
    # scvi 1.1 predates torch's weights_only default; this checkpoint was just
    # created here, so allow its AnnData registry metadata to be loaded.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    restored = SCCORAL.load(tmp_path / "model", adata=model.adata)
    assert restored.training_status_ == model.training_status_


def test_failed_training_exports_failure_status(model, tmp_path):
    class FailBatch(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            raise RuntimeError("intentional test failure")

    path = tmp_path / "failed.json"
    with pytest.warns(UserWarning, match="without completing a joint-training epoch"):
        with pytest.raises(RuntimeError, match="intentional test failure"):
            train(model, max_epochs=2, callbacks=[FailBatch()], training_status_path=path)
    assert json.loads(path.read_text())["termination_reason"] == "failed"
    assert model.training_status_["epochs_completed"] == 0


def test_partial_epoch_does_not_count_as_completed(model):
    with pytest.warns(UserWarning, match="without completing a joint-training epoch"):
        train(model, max_epochs=2, max_steps=1, pretraining=False, early_stopping=False)
    assert model.training_status_["joint_epochs_completed"] == 0
    assert model.training_status_["termination_reason"] == "max_steps"
