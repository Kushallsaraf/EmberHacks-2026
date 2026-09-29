"""Standalone inference; no connection to the Flask application."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ml.features import FEATURE_NAMES, sequence_features
from ml.train_form_classifier import CNN, MLP, probabilities


def predict_form(landmark_sequence: np.ndarray, model_path: str | Path = "ml/models/cnn.pt") -> tuple[int, float]:
    """Return (0=good/1=bad, probability of returned class).

    Input: uniformly sampled (N,33,3) x/y/visibility, including invalid frames
    as NaNs. Softmax scores are not calibrated probabilities or medical advice.
    """
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"No trained model at {path}; extract real clips and train first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != list(FEATURE_NAMES) or checkpoint["method"] not in ("cnn", "mlp"):
        raise ValueError("Unsupported model feature schema.")
    model = CNN() if checkpoint["method"] == "cnn" else MLP()
    model.load_state_dict(checkpoint["state_dict"])
    x = sequence_features(landmark_sequence, checkpoint["frames"])[None]
    scores = probabilities(model, x, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), checkpoint["method"])[0]
    label = int(scores.argmax())
    return label, float(scores[label])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("landmarks", type=Path, help=".npy array (N,33,3): x, y, visibility")
    parser.add_argument("--model", type=Path, default=Path("ml/models/cnn.pt"))
    args = parser.parse_args()
    try:
        label, probability = predict_form(np.load(args.landmarks, allow_pickle=False), args.model)
        print(f"label={label} ({'bad' if label else 'good'}), predicted-class probability={probability:.4f}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Prediction stopped: {error}\n")


if __name__ == "__main__":
    main()
