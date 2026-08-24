---
name: Video corpus regression policy
description: How supplied gym recordings are used when evolving requested-exercise analysis.
---

Any change to requested-exercise analysis must be checked against the entire supplied video corpus, not tuned to one successful recording. The corpus must include both matching examples and confidently mismatched exercises; a mismatch is a successful fail-closed result, never a partial score.

**Why:** The supplied recordings intentionally contain multiple movement families. Optimizing only around a bench-press clip previously allowed unrelated movements to receive plausible-looking repetition counts.

**How to apply:** Maintain a readable manifest of the source recordings and their expected compatibility for each supported requested exercise. Run that manifest after changes to exercise gating, equipment tracking, rep detection, or scoring. A negative case passes only through the explicit requested-exercise mismatch decision; missing files, model failures, tracker errors, and unrelated exceptions are regression failures. Treat uncertain labels as explicitly unresolved rather than as training truth.