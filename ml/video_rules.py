"""Run the app's squat rules over whole source videos and assign reps to clips.

The browser tracker calibrates once and stays armed across a set, so back-to-back
reps need no fresh standing hold. Assessing each one-rep clip on its own would
demand one and leave most clips unassessable. Here each source video listed in
ml/data/segments.csv is streamed through SquatRules as one continuous set, with
the same MediaPipe settings as extraction, and every completed rule rep is
assigned to the clip it overlaps most in time.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.rule_baseline import SquatRules

LOG = logging.getLogger(__name__)
COLUMNS = ["clip", "label", "faults", "rule_reps"]


def video_reps(path: Path) -> tuple[list[dict], float]:
    """Completed rule reps (label, faults, start_ms, end_ms) and the source FPS."""
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(path))
    tracker = SquatRules(target_reps=100)  # The app's maximum set size.
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot decode {path} or determine FPS.")
        # Identical to extract_features.extract_clip.
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    min_detection_confidence=.65, min_tracking_confidence=.65) as pose:
            count = 0
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
    finally:
        capture.release()
    return tracker.reps, fps


def assign(reps: list[dict], clips: list[dict], fps: float) -> dict[str, list[dict]]:
    """Map each rule rep to the clip covering most of it (at least half).

    clips have start_frame/end_frame [start, end) in the same source video.
    """
    assigned: dict[str, list[dict]] = defaultdict(list)
    for rep in reps:
        duration = max(rep["end_ms"] - rep["start_ms"], 1e-9)
        best, overlap = None, 0.
        for clip in clips:
            start, end = clip["start_frame"] * 1000 / fps, (clip["end_frame"] - 1) * 1000 / fps
            shared = min(end, rep["end_ms"]) - max(start, rep["start_ms"])
            if shared > overlap:
                best, overlap = clip["clip"], shared
        if best is not None and overlap >= duration / 2:
            assigned[best].append(rep)
    return assigned


def clip_label(reps: list[dict]) -> tuple[int, list[str]]:
    """-1 when the rules completed no rep here; bad if any assigned rep faulted."""
    if not reps:
        return -1, []
    faults = sorted({f for rep in reps for f in rep["faults"]})
    return int(bool(faults)), faults


def read_segments(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise ValueError(f"Missing {path}; run python -m ml.segment_reps first.")
    by_source: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            by_source[row["source"]].append({"clip": row["clip"], "start_frame": int(row["start_frame"]),
                                             "end_frame": int(row["end_frame"])})
    return by_source


def run(segments: Path, output: Path) -> None:
    rows, total, unmatched = [], 0, 0
    for source, clips in sorted(read_segments(segments).items()):
        reps, fps = video_reps(Path(source))
        assigned = assign(reps, clips, fps)
        total += len(reps)
        unmatched += len(reps) - sum(len(r) for r in assigned.values())
        for clip in clips:
            label, faults = clip_label(assigned.get(clip["clip"], []))
            rows.append({"clip": clip["clip"], "label": label, "faults": json.dumps(faults),
                         "rule_reps": len(assigned.get(clip["clip"], []))})
        LOG.info("%s: %d rule reps over %d clips", Path(source).name, len(reps), len(clips))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assessed = sum(r["label"] >= 0 for r in rows)
    LOG.info("Saved %s: %d/%d clips assessed; %d rule reps, %d not matched to a clip",
             output, assessed, len(rows), total, unmatched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=Path("ml/data/segments.csv"))
    parser.add_argument("--output", type=Path, default=Path("ml/data/video_rules.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args.segments, args.output)
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Rule assessment stopped: {error}\n")


if __name__ == "__main__":
    main()
