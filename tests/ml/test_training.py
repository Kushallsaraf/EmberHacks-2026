"""All generated arrays below are test-only and never written as results."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from ml.train_form_classifier import CNN, MLP, FormDataset, evaluate, fit, make_folds, metrics, normalize_fit, probabilities
from ml.features import FEATURE_NAMES
from ml.predict import predict_form


def test_dataset_models_and_augmentation():
    x = np.ones((8, 32, 6))
    batch, label = FormDataset(x, np.arange(8) % 2, augment=True)[0]
    assert batch.shape == (6, 32) and label.ndim == 0
    assert CNN()(batch[None]).shape == (1, 2)
    assert MLP()(torch.zeros(3, 24)).shape == (3, 2)
    np.testing.assert_array_equal(x, 1)


@pytest.mark.parametrize("method", ["cnn", "mlp"])
def test_training_loss_decreases_and_is_reproducible(method):
    rng = np.random.default_rng(12)
    y = np.arange(16) % 2
    x = rng.normal(0, .03, (16, 32, 6)) + y[:, None, None] * 2
    first = fit(x, y, method, 42, epochs=30)
    second = fit(x, y, method, 42, epochs=30)
    assert first[3][-1] < first[3][0] * .8
    np.testing.assert_array_equal(first[3], second[3])
    scores = probabilities(first[0], x, first[1], first[2], method)
    assert scores.shape == (16, 2)
    np.testing.assert_allclose(scores.sum(1), 1, atol=1e-6)


def test_normalization_only_uses_supplied_training_rows():
    train = np.arange(120).reshape(5, 4, 6).astype(float)
    mean, std = normalize_fit(train, "cnn")
    np.testing.assert_allclose(mean, train.mean((0, 1)))
    assert np.isfinite(std).all()
    assert not np.allclose(mean, np.concatenate([train, train + 10000]).mean((0, 1)))


@pytest.mark.parametrize("size", [12, 60])
def test_subject_splits_have_no_leakage(size):
    y = np.arange(size) % 2
    groups = np.arange(size) // 2
    folds = make_folds(y, groups, 17)
    assert len(folds) == (size // 2 if size < 40 else 5)
    seen = []
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(size))


def test_clip_loo_and_invalid_dataset():
    assert len(make_folds(np.array([0, 0, 1, 1]), np.arange(4), 17, "clip")) == 4
    with pytest.raises(ValueError):
        make_folds(np.array([0, 1]), np.arange(2), 17)
    with pytest.raises(ValueError):
        make_folds(np.zeros(4), np.arange(4), 17)


def test_bad_class_metrics():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert result["recall_bad"] == 1
    assert result["precision_bad"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_evaluation_pools_each_clip_once_per_seed_without_writing_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y = np.arange(6) % 2
    data = {"X": np.random.default_rng(0).normal(size=(6, 8, 6)), "y": y,
            "groups": np.array(["a", "a", "b", "b", "c", "c"]),
            "clip_names": np.array([f"TEST-ONLY-{i}" for i in range(6)]), "baseline": y.copy()}
    report = evaluate(data, [17, 42], epochs=1, split="subject")
    assert len(report["folds"]) == 6
    for method in report["methods"].values():
        assert np.asarray(method["pooled_confusion_matrix"]).sum() == 12
        assert np.asarray(method["predictions"]).shape == (2, 6)
    assert report["methods"]["rules"]["accuracy"] == {"mean": 1., "std": 0.}
    assert not list(tmp_path.iterdir())


def test_predict_roundtrip(tmp_path):
    path = tmp_path / "test-only.pt"
    torch.save({"state_dict": CNN().state_dict(), "method": "cnn", "mean": torch.zeros(6),
                "std": torch.ones(6), "frames": 32, "feature_names": list(FEATURE_NAMES)}, path)
    points = np.full((8, 33, 3), .95)
    for left, right, y in ((11, 12, .2), (23, 24, .4), (25, 26, .6), (27, 28, .8)):
        points[:, left] = [.4, y, .95]
        points[:, right] = [.6, y, .95]
    label, probability = predict_form(points, path)
    assert label in (0, 1) and .5 <= probability <= 1
    with pytest.raises(ValueError, match="No trained model"):
        predict_form(points, tmp_path / "missing.pt")
