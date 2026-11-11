// Test-only bridge executes the actual, unchanged browser state machine.
// Input: a frame array (first rep's label) or {frames, targetReps} (every rep).
import { readFileSync } from "node:fs";
import { MovementTracker } from "../../app/static/js/movement.js";
const input = JSON.parse(readFileSync(0, "utf8"));
const set = !Array.isArray(input);
const frames = set ? input.frames : input;
const tracker = new MovementTracker({ exerciseId: "squat", targetReps: set ? input.targetReps : 1 });
for (const frame of frames) {
  tracker.update({
    landmarks: frame.image?.map(([x, y, visibility]) => ({ x, y, visibility })),
    worldLandmarks: frame.world?.map(([x, y, z]) => ({ x, y, z })),
    now: frame.time, capturedAt: frame.time,
  });
}
const label = (rep) => ({ label: Number(rep.faults.length > 0), faults: rep.faults });
const reps = tracker.summary().reps;
console.log(JSON.stringify(set
  ? reps.map((rep) => ({ ...label(rep), start_ms: rep.started_at_ms, end_ms: rep.completed_at_ms }))
  : reps[0] ? label(reps[0]) : null));
