"""Hand-built rep/clip timings are test-only; never an experiment data source."""
import os

import pytest

pytest.importorskip("numpy")
from ml.extract_features import read_video_rules
from ml.video_rules import assign, clip_label

CLIPS = [{"clip": "good/a.mp4", "start_frame": 0, "end_frame": 30},
         {"clip": "good/b.mp4", "start_frame": 30, "end_frame": 60}]


def rep(start_ms, end_ms, faults=()):
    return {"label": int(bool(faults)), "faults": list(faults), "start_ms": start_ms, "end_ms": end_ms}


def test_rep_goes_to_clip_with_most_overlap():
    # At 15 fps clip a spans 0-1933 ms and clip b 2000-3933 ms.
    assigned = assign([rep(300, 1800), rep(1700, 3500, ["shallow_depth"])], CLIPS, 15.)
    assert [r["start_ms"] for r in assigned["good/a.mp4"]] == [300]
    assert [r["start_ms"] for r in assigned["good/b.mp4"]] == [1700]


def test_rep_mostly_outside_clips_is_unassigned():
    assert not assign([rep(3800, 6000)], CLIPS, 15.)


def test_clip_labels():
    assert clip_label([]) == (-1, [])
    assert clip_label([rep(0, 1, [])]) == (0, [])
    assert clip_label([rep(0, 1, []), rep(1, 2, ["fast_cadence"])]) == (1, ["fast_cadence"])


def test_stale_video_rules_are_refused(tmp_path):
    rules, segments = tmp_path / "video_rules.csv", tmp_path / "segments.csv"
    rules.write_text('clip,label,faults,rule_reps\ngood/a.mp4,1,"[""shallow_depth""]",1\n')
    segments.write_text("clip,source,start_frame,end_frame,fps\n")
    os.utime(rules, (1, 1))
    with pytest.raises(ValueError, match="predates"):
        read_video_rules(rules, segments)
    os.utime(rules, None)
    assert read_video_rules(rules, segments) == {"good/a.mp4": (1, ["shallow_depth"])}
    assert read_video_rules(tmp_path / "missing.csv", segments) == {}
