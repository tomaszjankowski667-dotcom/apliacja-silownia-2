---
name: Cross-platform pose retry
description: Safe retry policy for CPU-dependent MediaPipe confidence-boundary differences.
---

Allow one alternate pose-confidence attempt only after the exact failure indicating that no complete repetitions were found. The retry must rerun every existing identity, grip, anatomy, equipment, and repetition quality gate.

**Why:** The same video and model can fall on opposite sides of a MediaPipe confidence boundary across CPU delegates, causing the default pass to lose an otherwise valid hand/equipment track.

**How to apply:** Keep the default conservative pass first, use one tested alternate threshold, record all attempted thresholds in the result, and immediately propagate reference, ambiguity, or other quality errors without retrying.