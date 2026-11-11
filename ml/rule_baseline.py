"""Port of the browser squat tracker, one repetition or a continuous set.

Source: app/static/js/movement.js at ad5176d, lines 7-19, 36-45, 50-66,
143-402; framing: app/static/js/calibration.js lines 37-79.
Use raw 3D world coordinates and original timestamps, NEVER resampled 2D angles.
An unassessable/incomplete repetition returns None, not a made-up good/bad label.
"""
from __future__ import annotations

import math
import numpy as np

from ml.features import JOINTS


class SquatRules:
    """Stateful squat assessment using world coordinates and frame timestamps."""

    def __init__(self, target_reps: int = 1) -> None:
        self.target_reps = target_reps
        self.last_time: float | None = None
        self.standing: np.ndarray | None = None
        self.reps: list[dict] = []
        self.reset()

    @property
    def result(self) -> dict | None:
        """First completed rep's label and faults, as the one-rep oracle reports."""
        return None if not self.reps else {"label": self.reps[0]["label"], "faults": self.reps[0]["faults"]}

    def reset(self) -> None:
        self.phase = "start"
        self.armed = False
        self.start_since = self.depart_since = self.reverse_since = self.finish_since = None
        self.window: dict | None = None
        self.rep: dict | None = None

    def invalidate(self) -> None:
        self.standing = None
        self.reset()

    def threshold(self, fixed: float, offset: float) -> np.ndarray:
        return np.minimum(fixed, self.standing - offset) if self.standing is not None else np.full(2, fixed)

    def update(self, image: np.ndarray | None, world: np.ndarray | None, time_ms: float) -> dict | None:
        """Consume one fresh frame, including missing frames to reset partial reps.

        image is (33,3) x/y/visibility; world is (33,3) x/y/z. Offline frames
        use now=capturedAt, as they have no browser inference latency.
        """
        chronological = np.isfinite(time_ms) and (self.last_time is None or time_ms > self.last_time)
        gap = 0 if self.last_time is None else time_ms - self.last_time
        if np.isfinite(time_ms) and (self.last_time is None or time_ms > self.last_time):
            self.last_time = time_ms
        if image is None or world is None:
            self.invalidate()
            return None
        p = np.asarray(image)[JOINTS]
        if (not chronological or gap > 350 or not np.isfinite(p).all()
                or np.any(p[:, 2] < .7) or np.any(p[:, :2] < .025) or np.any(p[:, :2] > .975)):
            self.invalidate()
            return None
        angles = []
        for a, b, c in ((23, 25, 27), (24, 26, 28)):
            u, v = world[a] - world[b], world[c] - world[b]
            # Match JS Math.hypot/Math.acos and operation order at boundaries.
            denominator = math.hypot(*u) * math.hypot(*v)
            if not np.isfinite(denominator) or denominator < 1e-8:
                self.invalidate()
                return None
            cosine = sum(float(first) * float(second) for first, second in zip(u, v)) / denominator
            angles.append(math.acos(max(-1., min(1., cosine))) * 180 / math.pi)
        a = np.asarray(angles)
        flexion, extension = float(a.max()), float(a.min())
        if len(self.reps) >= self.target_reps:
            return self.result
        if not self.armed:
            if extension < 145:
                self.start_since = self.window = None
                return None
            w = self.window
            if w is None or np.any(np.maximum(w["max"], a) - np.minimum(w["min"], a) > 6):
                self.window = {"min": a.copy(), "max": a.copy(), "sum": a.copy(), "count": 1}
                self.start_since = time_ms
            else:
                w["min"], w["max"] = np.minimum(w["min"], a), np.maximum(w["max"], a)
                w["sum"] += a
                w["count"] += 1
            if time_ms - self.start_since >= 300:
                self.standing = self.window["sum"] / self.window["count"]
                self.armed = True
            return None
        if self.rep is None:
            self.depart_since = (time_ms if self.depart_since is None else self.depart_since) if np.all(a <= self.threshold(148, 15)) else None
            if self.depart_since is None or time_ms - self.depart_since < 120:
                return None
            self.rep = {"start": self.depart_since, "peak": flexion, "asym": 0., "asym_since": None, "reached": False}
            self.phase, self.depart_since = "eccentric", None
        rep = self.rep
        rep["reached"] |= bool(np.all(a <= self.threshold(140, 25)))
        if time_ms - rep["start"] > 30000:
            self.invalidate()
            return None
        if flexion < rep["peak"] - 1:
            rep["peak"] = flexion
        asymmetry = float(abs(a[0] - a[1]))
        if asymmetry > 20:
            if rep["asym_since"] is None:
                rep["asym_since"] = time_ms
            if time_ms - rep["asym_since"] >= 150:
                rep["asym"] = max(rep["asym"], asymmetry)
        else:
            rep["asym_since"] = None
        if self.phase == "eccentric" and flexion <= 105:
            self.phase = "bottom"
        if self.phase in ("eccentric", "bottom"):
            self.reverse_since = (time_ms if self.reverse_since is None else self.reverse_since) if flexion >= rep["peak"] + 9 else None
            if self.reverse_since is not None and time_ms - self.reverse_since >= 120:
                self.phase = "bottom" if self.phase == "eccentric" else "concentric"
        if self.phase == "concentric" and flexion <= rep["peak"] + 3:
            self.phase, self.reverse_since, self.finish_since = "bottom", None, None
        if self.phase == "concentric":
            self.finish_since = (time_ms if self.finish_since is None else self.finish_since) if np.all(a >= self.threshold(155, 5)) else None
            if self.finish_since is not None and time_ms - self.finish_since >= 180:
                duration = time_ms - rep["start"]
                if not rep["reached"] or duration < 700:
                    self.reset()
                    return None
                faults = []
                if rep["peak"] > 105:
                    faults.append("shallow_depth")
                if rep["asym"] > 20:
                    faults.append("uneven_range")
                if duration < 1400:
                    faults.append("fast_cadence")
                completed = {"label": int(bool(faults)), "faults": faults,
                             "start_ms": rep["start"], "end_ms": time_ms}
                self.reps.append(completed)
                # As in the app, completion is a sustained start pose: stay
                # armed with the same calibration so the next rep may depart.
                self.reset()
                self.armed = True
                return completed
        return None
