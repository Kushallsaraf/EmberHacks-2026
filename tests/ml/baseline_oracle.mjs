// Test-only bridge executes the actual, unchanged browser state machine.
import { readFileSync } from "node:fs";
import { MovementTracker } from "../../app/static/js/movement.js";
const frames = JSON.parse(readFileSync(0, "utf8"));
const tracker = new MovementTracker({ exerciseId: "squat", targetReps: 1 });
for (const frame of frames) {
  tracker.update({
    landmarks: frame.image?.map(([x, y, visibility]) => ({ x, y, visibility })),
    worldLandmarks: frame.world?.map(([x, y, z]) => ({ x, y, z })),
    now: frame.time, capturedAt: frame.time,
  });
}
const rep = tracker.summary().reps[0];
console.log(JSON.stringify(rep ? { label: Number(rep.faults.length > 0), faults: rep.faults } : null));
