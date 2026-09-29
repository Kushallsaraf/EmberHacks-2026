"""CPU CNN/MLP cross-validation against the current Spotter rules."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.extract_features import sha256
from ml.features import FEATURE_NAMES, summary_features

LOG = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all RNGs and enable deterministic, single-threaded Torch execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class FormDataset(Dataset):
    """Normalized (N,T,6) sequences; optional train-only noise and smooth warp."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.x, self.y, self.augment = np.asarray(x, dtype=np.float32), np.asarray(y), augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index].copy()
        if self.augment:
            t = np.linspace(0, 1, len(x))
            warp = t + np.random.uniform(-.04, .04) * np.sin(np.pi * t)
            x = np.stack([np.interp(warp, t, x[:, i]) for i in range(6)], axis=1)
            x += np.random.normal(0, .015, x.shape)
        return torch.tensor(x.T, dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.long)


class CNN(nn.Module):
    """Accept (batch, 6, time); return two class logits per sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(6, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """Accept (batch, 24) summary features; return two class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_fit(x: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit on training rows only; tiny variances are replaced by one."""
    axes = (0, 1) if method == "cnn" else 0
    mean, std = x.mean(axis=axes), x.std(axis=axes)
    return mean, np.where(std < 1e-6, 1., std)


def fit(x: np.ndarray, y: np.ndarray, method: str, seed: int, epochs: int = 60) -> tuple[nn.Module, np.ndarray, np.ndarray, list[float]]:
    """Return the fitted model, training mean/std and per-epoch losses."""
    seed_everything(seed)
    data = x if method == "cnn" else summary_features(x)
    mean, std = normalize_fit(data, method)
    data = (data - mean) / std
    model = CNN() if method == "cnn" else MLP()
    dataset = FormDataset(data, y, augment=True) if method == "cnn" else torch.utils.data.TensorDataset(
        torch.tensor(data, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=min(16, len(y)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        losses.append(total / len(y))
    model.eval()
    return model, mean, std, losses


def probabilities(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, method: str) -> np.ndarray:
    """Return class probabilities using the fitted fold's normalization."""
    data = x if method == "cnn" else summary_features(x)
    data = (data - mean) / std
    if method == "cnn":
        data = data.transpose(0, 2, 1)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(data, dtype=torch.float32)).softmax(-1).numpy()


def make_folds(y: np.ndarray, groups: np.ndarray, seed: int, split: str = "subject") -> list[tuple[np.ndarray, np.ndarray]]:
    """Small grouped data use leave-one-subject-out to prevent identity leakage."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Need both good and bad real clips.")
    if split == "subject":
        if len(np.unique(groups)) < (2 if len(y) < 40 else 5):
            raise ValueError("Need >=2 independent subjects for leave-one-subject-out or >=5 for five-fold group CV.")
        splitter = LeaveOneGroupOut() if len(y) < 40 else StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    else:
        splitter = LeaveOneOut() if len(y) < 40 else StratifiedKFold(5, shuffle=True, random_state=seed)
        if len(y) >= 40 and np.bincount(y).min() < 5:
            raise ValueError("Five-fold CV needs >=5 clips per class.")
    folds = list(splitter.split(np.zeros(len(y)), y, groups if split == "subject" else None))
    for train, test in folds:
        if not len(test):
            raise ValueError("An evaluation fold is empty; add independent subjects.")
        if len(np.unique(y[train])) != 2:
            raise ValueError("A training fold lacks a class. Add independent subjects with both labels.")
        if split == "subject" and set(groups[train]) & set(groups[test]):
            raise ValueError("Subject leakage detected.")
    return folds


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)), "precision_bad": float(precision[0]),
            "recall_bad": float(recall[0]), "f1_bad": float(f1[0]),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def load_data(path: Path) -> dict:
    """Validate the extraction archive and keep only the common assessable cohort."""
    if not path.exists():
        raise ValueError(f"Missing {path}. Add real clips and run python -m ml.extract_features first.")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"X", "y", "baseline", "groups", "clip_names", "feature_names", "provenance", "skipped", "format_version"}
    if not required <= data.keys() or int(data["format_version"]) != 1:
        raise ValueError("Unsupported archive: regenerate from real clips with ml.extract_features.")
    x, y = data["X"], data["y"]
    if (x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 6 or not np.isfinite(x).all()
            or y.shape != (len(x),) or not np.isin(y, [0, 1]).all()
            or list(data["feature_names"]) != list(FEATURE_NAMES)):
        raise ValueError("Invalid feature archive shape, labels, or feature order.")
    for key in ("baseline", "groups", "clip_names"):
        if data[key].shape != (len(y),):
            raise ValueError(f"Invalid archive {key} shape.")
    if not np.isin(data["baseline"], [-1, 0, 1]).all():
        raise ValueError("Invalid baseline labels.")
    provenance = json.loads(str(data["provenance"]))
    if len(provenance) != len(y) or any(not r.get("sha256") or not r.get("label_source") for r in provenance):
        raise ValueError("Missing clip provenance; run extraction from real videos.")
    mask = data["baseline"] >= 0
    data["excluded"] = data["clip_names"][~mask].tolist()
    data["original_counts"] = np.bincount(y, minlength=2).tolist()
    for key in ("X", "y", "baseline", "groups", "clip_names"):
        data[key] = data[key][mask]
    if not mask.any():
        raise ValueError("Rules completed no repetitions. Include standing holds and complete reps; no comparable results can be generated.")
    return data


def evaluate(data: dict, seeds: list[int], epochs: int, split: str) -> dict:
    """Identical held-out clips for all methods, one prediction per clip per seed."""
    x, y, groups = data["X"], data["y"], data["groups"]
    fold_sets = [make_folds(y, groups, seed, split) for seed in seeds]
    output: dict = {name: {"per_seed": [], "predictions": []} for name in ("cnn", "mlp", "rules")}
    splits = []
    for seed, folds in zip(seeds, fold_sets):
        predictions = {name: np.full(len(y), -1, dtype=int) for name in output}
        predictions["rules"] = data["baseline"].copy()
        for fold, (train, test) in enumerate(folds):
            LOG.info("Seed %d fold %d/%d: train=%d test=%d", seed, fold + 1, len(folds), len(train), len(test))
            splits.append({"seed": seed, "fold": fold, "train": data["clip_names"][train].tolist(), "test": data["clip_names"][test].tolist()})
            for method in ("cnn", "mlp"):
                model, mean, std, _ = fit(x[train], y[train], method, seed + fold, epochs)
                predictions[method][test] = probabilities(model, x[test], mean, std, method).argmax(1)
        for method, prediction in predictions.items():
            if np.any(prediction < 0):
                raise RuntimeError("Missing held-out predictions.")
            output[method]["per_seed"].append({"seed": seed, **metrics(y, prediction)})
            output[method]["predictions"].append(prediction.tolist())
    for method, item in output.items():
        for metric in ("accuracy", "precision_bad", "recall_bad", "f1_bad"):
            values = [row[metric] for row in item["per_seed"]]
            item[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        item["pooled_confusion_matrix"] = confusion_matrix(np.tile(y, len(seeds)), np.asarray(item["predictions"]).ravel(), labels=[0, 1]).tolist()
    return {"methods": output, "folds": splits}


def write_results(report: dict, folder: Path) -> None:
    """Write metrics JSON, a Markdown table and pooled confusion matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Squat-form experiment", "", report["conclusion"], "",
             "| Method | Accuracy mean ± SD | Bad precision | Bad recall | Bad F1 |",
             "|---|---:|---:|---:|---:|"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (method, result) in zip(axes, report["methods"].items()):
        acc = result["accuracy"]
        lines.append(f"| {method} | {acc['mean']:.3f} ± {acc['std']:.3f} | {result['precision_bad']['mean']:.3f} | {result['recall_bad']['mean']:.3f} | {result['f1_bad']['mean']:.3f} |")
        matrix = np.asarray(result["pooled_confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"], yticklabels=["good", "bad"],
               xlabel="Predicted", ylabel="Actual", title=method)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    lines += ["", f"Split: {report['split']}. Evaluated clips by class: {report['class_counts']}.",
              f"Rule-unavailable clips excluded from ALL methods: {len(report['excluded_rule_unavailable'])}.",
              "This common-cohort comparison can favor the rules by excluding difficult/incomplete reps.",
              "Pooled matrices repeat each clip once per seed; these are not independent observations.",
              "SD is population SD across seeds, not a confidence interval. No generalization claim."]
    (folder / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("ml/data/features.npz"))
    parser.add_argument("--results", type=Path, default=Path("ml/results"))
    parser.add_argument("--model-dir", type=Path, default=Path("ml/models"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 123])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--split", choices=["subject", "clip"], default="subject")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 100000 for s in args.seeds):
            raise ValueError("Use positive epochs and distinct nonnegative seeds below 2**32-100000.")
        data = load_data(args.features)
        LOG.info("Class counts good/bad=%s; excluded rule-unavailable=%d", np.bincount(data["y"], minlength=2), len(data["excluded"]))
        if args.split == "clip":
            LOG.warning("Clip splitting can leak subject identity; results do not measure unseen-person generalization.")
        report = evaluate(data, args.seeds, args.epochs, args.split)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = "unavailable", None
        better = [m for m in ("cnn", "mlp") if report["methods"][m]["accuracy"]["mean"] > report["methods"]["rules"]["accuracy"]["mean"]]
        conclusion = (f"Higher mean accuracy than rules: {', '.join(better)}. This is not evidence of statistical significance."
                      if better else "Neither neural model beats the rules on mean accuracy in this experiment.")
        report.update(seeds=args.seeds, epochs=args.epochs, split=args.split, git_commit=commit, git_dirty=dirty,
                      class_counts=np.bincount(data["y"], minlength=2).tolist(), original_class_counts=data["original_counts"],
                      excluded_rule_unavailable=data["excluded"], clip_names=data["clip_names"].tolist(), labels=data["y"].tolist(),
                      feature_sha256=sha256(args.features), provenance=json.loads(str(data["provenance"])),
                      skipped_clips=json.loads(str(data["skipped"])), conclusion=conclusion,
                      packages={p: importlib.metadata.version(p) for p in ("torch", "numpy", "mediapipe", "scikit-learn", "opencv-contrib-python", "matplotlib")},
                      python=platform.python_version(), platform=platform.platform(),
                      installed_packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                      optimizer={"name": "Adam", "lr": .001, "weight_decay": .0001}, frames=int(data["X"].shape[1]))
        write_results(report, args.results)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        for method in ("cnn", "mlp"):
            model, mean, std, _ = fit(data["X"], data["y"], method, args.seeds[0], args.epochs)
            torch.save({"state_dict": model.state_dict(), "method": method, "mean": torch.tensor(mean),
                        "std": torch.tensor(std), "frames": int(data["X"].shape[1]), "feature_names": list(FEATURE_NAMES),
                        "seed": args.seeds[0], "feature_sha256": report["feature_sha256"]}, args.model_dir / f"{method}.pt")
        LOG.info("%s Results: %s; full-data inference models: %s", conclusion, args.results, args.model_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
