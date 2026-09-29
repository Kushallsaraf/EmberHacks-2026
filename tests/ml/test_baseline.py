"""Test-only hand-built motion; compare against actual JavaScript execution."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

np = pytest.importorskip("numpy")
from ml.rule_baseline import SquatRules


def make_frames(bottom=90, asymmetry=0, interval=100, standing=175):
    angles = [standing] * 5 + [165, 150, 138, 130, 120, 110, bottom, bottom, bottom,
                              bottom, bottom + 4, bottom + 12, bottom + 22, 125, 145, 165, standing, standing, standing]
    frames = []
    for i, angle in enumerate(angles):
        world = np.zeros((33, 3))
        for side, (a, b, c) in enumerate(((23, 25, 27), (24, 26, 28))):
            radians = np.radians(angle - (asymmetry if side and 8 <= i <= 16 else 0))
            world[a] = [side, 1, 0]
            world[b] = [side, 0, 0]
            world[c] = [side, np.cos(radians), np.sin(radians)]
        frames.append({"image": np.tile([.5, .5, .95], (33, 1)).tolist(), "world": world.tolist(), "time": i * interval})
    return frames


@pytest.mark.parametrize("case", ["good", "shallow", "uneven", "fast", "missing", "gap", "degenerate", "calibrated", "partial", "boundary"])
def test_js_parity(case):
    if not shutil.which("node"):
        pytest.fail("Node >=20 is required to verify parity with the real JavaScript rules.")
    frames = make_frames(bottom=128 if case == "shallow" else 105 if case == "boundary" else 90,
                         asymmetry=30 if case == "uneven" else 0,
                         interval=70 if case == "fast" else 100,
                         standing=158 if case == "calibrated" else 175)
    if case == "missing":
        frames[10]["image"] = None
    elif case == "gap":
        for frame in frames[10:]:
            frame["time"] += 500
    elif case == "degenerate":
        frames[10]["world"] = np.zeros((33, 3)).tolist()
    elif case == "partial":
        frames = frames[:12]
    tracker = SquatRules()
    for frame in frames:
        tracker.update(None if frame["image"] is None else np.array(frame["image"]),
                       np.array(frame["world"]), frame["time"])
    oracle = Path(__file__).with_name("baseline_oracle.mjs")
    actual = json.loads(subprocess.check_output(["node", str(oracle)], input=json.dumps(frames), text=True))
    assert tracker.result == actual
    if case == "good":
        assert actual == {"label": 0, "faults": []}
    if case == "shallow":
        assert "shallow_depth" in actual["faults"]
    if case == "uneven":
        assert "uneven_range" in actual["faults"]
