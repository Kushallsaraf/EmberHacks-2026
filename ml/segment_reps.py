"""Split the multi-repetition Mendeley squat videos into one-repetition clips.

Each source video holds ~10-15 back-to-back reps under one provider label.
Reps are found from hip height above the ankles (view-independent, unlike 2D
knee angles in front views), and every rep inherits its video's label. This is
weak labelling: a "bad" video may contain acceptable reps. Rows are appended to
the manifest with the source file and frame interval, so they can be reviewed.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ml.extract_features import sha256

LOG = logging.getLogger(__name__)
SOURCE_URL = "https://data.mendeley.com/datasets/kgbb3yn47p/3"
NAME = re.compile(r"subject_(\d+)_squat_(good|bad)_(front|side|diagonal)\.mp4")
PREFIX = "mendeley_"
COLUMNS = ["clip", "subject_id", "source_url", "license", "label_source"]
# Machine-readable clip origins; ml.video_rules runs the app rules on the source.
SEGMENT_COLUMNS = ["clip", "source", "start_frame", "end_frame", "fps"]


def smooth_height(height: np.ndarray, window: int = 5) -> np.ndarray:
    """Interpolate NaN gaps and apply a centered moving average."""
    height = np.asarray(height, dtype=float)
    valid = np.isfinite(height)
    if valid.sum() < 2:
        raise ValueError("Need at least two frames with a detected hip and ankle.")
    index = np.arange(len(height))
    filled = np.interp(index, index[valid], height[valid])
    padded = np.pad(filled, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, "valid")


def detect_reps(height: np.ndarray, fps: float, min_valid: float = .35,
                prominence: float = .35, standing: float = .15) -> list[tuple[int, int]]:
    """Return [start, end) frame ranges, one per repetition.

    height is per-frame ankle y minus hip y in normalized image coordinates
    (NaN when undetected); squatting lowers it. Boundaries sit at the middle
    of the standing interval between bottoms, so each clip keeps part of the
    standing holds the rule baseline needs. Reps with fewer than min_valid
    measured frames are dropped rather than trusted to interpolation.
    """
    from scipy.signal import find_peaks

    raw = np.asarray(height, dtype=float)
    h = smooth_height(raw)
    top, bottom = np.percentile(h, 95), np.percentile(h, 5)
    span = top - bottom
    if not np.isfinite(span) or span <= 1e-6:
        return []
    bottoms, _ = find_peaks(-h, prominence=prominence * span, distance=max(1, int(fps)))
    if not len(bottoms):
        return []

    def stand_mid(lo: int, hi: int) -> int:
        # Middle of the frames near this interval's highest point.
        segment = h[lo:hi]
        near = np.flatnonzero(segment >= segment.max() - standing * span)
        return lo + int(round((near[0] + near[-1]) / 2))

    edges = [stand_mid(0, bottoms[0])] if bottoms[0] > 0 else [0]
    edges += [stand_mid(a, b) for a, b in zip(bottoms[:-1], bottoms[1:])]
    edges.append(stand_mid(bottoms[-1] + 1, len(h)) + 1 if bottoms[-1] + 1 < len(h) else len(h))
    reps = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start >= 8 and np.isfinite(raw[start:end]).mean() >= min_valid:
            reps.append((start, end))
    return reps


def hip_height(path: Path) -> tuple[np.ndarray, float]:
    """Per-frame ankle-minus-hip height from MediaPipe, NaN when unreliable."""
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    values = []
    try:
        # Static mode re-detects every frame; tracking loses blurred subjects.
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=.5) as pose:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                result = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                value = np.nan
                if result.pose_landmarks is not None:
                    p = np.array([[l.y, l.visibility] for l in result.pose_landmarks.landmark])
                    if (p[[23, 24, 27, 28], 1] >= .5).all():
                        value = p[[27, 28], 0].mean() - p[[23, 24], 0].mean()
                values.append(value)
    finally:
        capture.release()
    return np.asarray(values), fps


def write_clip(source: Path, target: Path, start: int, end: int, fps: float) -> None:
    """Re-encode frames [start, end) at the source FPS (constant frame rate)."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    try:
        for _ in range(end - start):
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def run(source: Path, clips: Path, manifest: Path, views: list[str], min_valid: float) -> None:
    videos = sorted(p for p in source.rglob("*.mp4") if (m := NAME.fullmatch(p.name)) and m.group(3) in views)
    if not videos:
        raise ValueError(f"No source videos for views {views} under {source}; run python -m ml.download_dataset first.")
    # The provider ships some byte-identical files under different labels or
    # subjects (e.g. subject 003 diagonal is both good and bad); the true label
    # is unknowable, so every copy is excluded.
    digests = {video: sha256(video) for video in sorted(source.rglob("*.mp4"))}
    counts = Counter(digests.values())
    duplicates = [v for v in videos if counts[digests[v]] > 1]
    for video in duplicates:
        LOG.warning("Excluded %s: byte-identical to another source video.", video.name)
    videos = [v for v in videos if v not in duplicates]
    for folder in ("good", "bad"):
        (clips / folder).mkdir(parents=True, exist_ok=True)
        for old in (clips / folder).glob(f"{PREFIX}*.mp4"):
            old.unlink()
    # Keep any non-Mendeley rows, e.g. personal recordings.
    kept = []
    if manifest.exists():
        with manifest.open(newline="") as stream:
            kept = [r for r in csv.DictReader(stream) if not r["clip"].split("/")[-1].startswith(PREFIX)]
    rows, segments = [], []
    for video in videos:
        subject, label, view = NAME.fullmatch(video.name).groups()
        height, fps = hip_height(video)
        reps = detect_reps(height, fps, min_valid)
        LOG.info("%s: %d reps from %d frames (%.0f%% measured)", video.name, len(reps), len(height),
                 100 * np.isfinite(height).mean())
        for number, (start, end) in enumerate(reps, 1):
            clip = f"{label}/{PREFIX}s{subject}_{view}_rep{number:02d}.mp4"
            write_clip(video, clips / clip, start, end, fps)
            rows.append({"clip": clip, "subject_id": f"mendeley_kgbb3yn47p_s{subject}", "source_url": SOURCE_URL,
                         "license": "CC BY 4.0",
                         "label_source": (f"provider video label '{label}' for {video.relative_to(source).as_posix()}; "
                                          f"frames {start}-{end - 1} ({start / fps:.2f}-{end / fps:.2f} s) auto-segmented; "
                                          "label propagated to rep; not independently reviewed")})
            segments.append({"clip": clip, "source": video.as_posix(), "start_frame": start, "end_frame": end, "fps": fps})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept + rows)
    with manifest.with_name("segments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(segments)
    counts = {label: sum(r["clip"].startswith(label) for r in rows) for label in ("good", "bad")}
    LOG.info("Wrote %d clips (%s) and %s", len(rows), counts, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("ml/data/downloads/mendeley_kgbb3yn47p_v3"))
    parser.add_argument("--clips", type=Path, default=Path("ml/data/clips"))
    parser.add_argument("--manifest", type=Path, default=Path("ml/data/manifest.csv"))
    parser.add_argument("--views", nargs="+", choices=["front", "side", "diagonal"], default=["front", "diagonal"])
    parser.add_argument("--min-valid", type=float, default=.35, help="Minimum measured-frame fraction per rep.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args.source, args.clips, args.manifest, args.views, args.min_valid)
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Segmentation stopped: {error}\n")


if __name__ == "__main__":
    main()
