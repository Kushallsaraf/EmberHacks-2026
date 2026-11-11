"""Extract pose features and rule predictions from labeled video clips."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ml.features import FEATURE_NAMES, frame_features, resample
from ml.rule_baseline import SquatRules

LOG = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Index provenance rows by clip path; reject missing fields and duplicates."""
    if not path.exists():
        raise ValueError(f"Missing {path}. Copy ml/data/manifest.example.csv and document each real clip.")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"clip", "subject_id", "source_url", "license", "label_source"}
    if not rows or any(not all(row.get(key, "").strip() for key in required) for row in rows):
        raise ValueError(f"Manifest needs nonempty columns {sorted(required)} for every clip.")
    if len({r["clip"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manifest clip names.")
    return {row["clip"]: row for row in rows}


def extract_clip(path: Path, frames: int) -> tuple[np.ndarray, int, dict]:
    """Decode each frame; retain original time for rule dwell and interpolation."""
    import cv2
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise RuntimeError("This extractor needs mediapipe==0.10.21 with mp.solutions.pose; install requirements-ml.txt in Python 3.11.")
    capture = cv2.VideoCapture(str(path))
    tracker = SquatRules()
    values, times = [], []
    count = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot decode video or determine FPS.")
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    min_detection_confidence=.65, min_tracking_confidence=.65) as pose:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                time_ms = count * 1000 / fps
                count += 1
                result = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks is None or result.pose_world_landmarks is None:
                    tracker.update(None, None, time_ms)
                    continue
                image = np.array([[p.x, p.y, p.visibility] for p in result.pose_landmarks.landmark])
                world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks.landmark])
                tracker.update(image, world, time_ms)
                feature = frame_features(image)
                if np.isfinite(feature).all():
                    values.append(feature)
                    times.append(time_ms)
        if len(values) < 8:
            raise ValueError(f"Only {len(values)} valid frames (need >= 8); decoded {count}.")
        baseline = -1 if tracker.result is None else tracker.result["label"]
        info = {"decoded_frames": count, "valid_frames": len(values), "fps": fps,
                "rule_faults": None if tracker.result is None else tracker.result["faults"]}
        return resample(np.asarray(values), frames, np.asarray(times)), baseline, info
    finally:
        capture.release()


def read_video_rules(path: Path | None, segments: Path) -> dict[str, tuple[int, list[str]]]:
    """Full-video rule labels from ml.video_rules, refusing results older than the clips."""
    if path is None or not path.exists():
        return {}
    if segments.exists() and path.stat().st_mtime < segments.stat().st_mtime:
        raise ValueError(f"{path} predates {segments}; rerun python -m ml.video_rules.")
    with path.open(newline="") as stream:
        return {row["clip"]: (int(row["label"]), json.loads(row["faults"])) for row in csv.DictReader(stream)}


def run(clips: Path, manifest: Path, output: Path, frames: int = 32, video_rules: Path | None = None) -> None:
    """Save features, labels, provenance and skip reasons to a compressed archive.

    Clips listed in video_rules take their rule label from a continuous pass over
    the source video (as the app runs); others are assessed as one-rep clips.
    """
    files = [(p, label) for folder, label in (("good", 0), ("bad", 1))
             for p in sorted((clips / folder).glob("*")) if p.suffix.lower() in VIDEO_SUFFIXES]
    if not files:
        raise ValueError(f"No real clips found. Add one-repetition videos to {clips}/good/ and {clips}/bad/, and document them in {manifest}. No results generated.")
    if frames < 2:
        raise ValueError("--frames must be >= 2.")
    metadata = read_manifest(manifest)
    full_video = read_video_rules(video_rules, manifest.with_name("segments.csv"))
    missing = [p.relative_to(clips).as_posix() for p, _ in files if p.relative_to(clips).as_posix() not in metadata]
    if missing:
        raise ValueError(f"Clips missing provenance: {missing}")
    x, y, names, groups, baseline, records, skipped = [], [], [], [], [], [], []
    hashes: set[str] = set()
    for path, label in files:
        name = path.relative_to(clips).as_posix()
        digest = sha256(path)
        if digest in hashes:
            raise ValueError(f"Duplicate video content: {name}; remove duplicate copies before extraction.")
        hashes.add(digest)
        try:
            features, prediction, info = extract_clip(path, frames)
            info["rule_source"] = "clip"
            if name in full_video:
                prediction, faults = full_video[name]
                info.update(rule_source="full_video", rule_faults=faults if prediction >= 0 else None)
        except ValueError as error:
            skipped.append({"clip": name, "reason": str(error)})
            LOG.warning("Skipped %s: %s", name, error)
            continue
        x.append(features)
        y.append(label)
        names.append(name)
        groups.append(metadata[name]["subject_id"])
        baseline.append(prediction)
        records.append({**metadata[name], **info, "sha256": digest, "label": label})
        LOG.info("%s: %s valid frames; rule label %s", name, info["valid_frames"], prediction)
    if not x:
        raise ValueError("No clips have >= 8 valid frames. No archive or results generated.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.int64),
                        clip_names=np.asarray(names), feature_names=np.asarray(FEATURE_NAMES),
                        groups=np.asarray(groups), baseline=np.asarray(baseline),
                        provenance=np.asarray(json.dumps(records)), skipped=np.asarray(json.dumps(skipped)),
                        format_version=np.asarray(1))
    LOG.info("Saved %s: good=%d, bad=%d; skipped=%d; baseline unavailable=%d", output,
             y.count(0), y.count(1), len(skipped), baseline.count(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("ml/data/clips"))
    parser.add_argument("--manifest", type=Path, default=Path("ml/data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("ml/data/features.npz"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--video-rules", type=Path, default=Path("ml/data/video_rules.csv"),
                        help="Full-video rule labels from ml.video_rules; used when present.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args.clips, args.manifest, args.output, args.frames, args.video_rules)
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Extraction stopped: {error}\n")


if __name__ == "__main__":
    main()
