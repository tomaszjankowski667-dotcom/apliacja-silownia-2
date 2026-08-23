---
name: Vision confidence policy
description: Reliability rule for exercise-video analysis under occlusion, multiple people, and camera ambiguity.
---

Exercise-video analysis must reject ambiguous subject selection, camera-side selection, equipment identity, anatomy, or repetition boundaries rather than score inferred data as fact.

**Why:** Gym footage can include spotters, plates that hide joints, and background circular objects. A plausible-looking score from the wrong person or object is worse than a clear request for a better clip.

**How to apply:** Keep confidence gates independent across pose, anatomy, and equipment tracking. Full anatomy requires a complete frame-one shoulder–elbow–wrist–hip observation plus continuous, calibrated arm-chain and body-scale evidence during the set; this rule is exercise-agnostic. Do not render reconstructed joints as observed anatomy. When anatomy is ambiguous but equipment is unambiguous, label the result as bar-path-only rather than presenting a full anatomical technique score.