---
name: Independent technique scores
description: Product rule for separating equipment-path quality from observed limb-motion quality.
---

Score equipment trajectory and limb movement as independent components. Limb motion must be unavailable when the anatomical observation gate fails; never infer it from equipment motion. Produce an overall technique score only when both components are reliable for every scored repetition.

**Why:** A single blended number hid whether poor results came from the load path or arm geometry, and could imply an anatomical judgment on bar-only footage.

**How to apply:** Keep per-repetition component records aligned, preserve explicit unavailable values, and require complete component coverage before averaging an overall result.