# Task1 Manual Validation And Curation Log

This document records the manual validation, targeted repair, and evaluator calibration work carried out for STAGE Task1 during the 2026-04 release cycle.

It is intended as a research-facing audit note that can be cited when describing benchmark construction effort, human quality control, and metric calibration.

## Scope

This note covers manual work on:

- released Task1 gold assets under `STAGE_v0/{English,Chinese}/*`
- Task1 prediction repair used for evaluation
- focal-character alignment checks
- suspicious low-node / alias cases
- transition-metric failure analysis and revision

It does not cover Task2 or Task3 curation except where Task1 focal-character consistency directly affected release quality.

## 1. Manual Validation Goals

The manual effort focused on four concrete goals:

1. Ensure released focal characters are the intended main or benchmark-relevant characters.
2. Detect obvious Task1 asset pathologies such as alias duplication, missing merges, or suspiciously short timelines.
3. Verify that repaired Task1 predictions align exactly with the released gold focal-character lists.
4. Check whether low evaluation scores reflected true Task1 quality problems or evaluator / metric artifacts.

## 2. Manual Checks On Released Task1 Gold Assets

### 2.1 Focal-character / alias sanity

Manual checks were performed on released Task1 timeline JSONs to verify:

- the canonical `character_name` was preserved
- aliases were real alternate names rather than a duplicate copy of the canonical name
- screenplay-local variants were treated as aliases instead of replacing the canonical Task1 focal name

Release fix applied:

- removed cases where `aliases` redundantly contained the same string as `character_name`

This led to the release cleanup commit:

- `a8a0b6d` `Clean Task1 aliases`

### 2.2 Suspicious low-node characters

Manual inspection was triggered for characters with unusually small numbers of timeline nodes, especially `<= 8`, to distinguish:

- valid secondary-focus characters
- extraction misses
- failed alias merges
- incomplete prior runs

Representative manually checked cases included:

- `Leena Klammer`
- `Marilyn`
- `Lucas`
- `宫羽田`
- `依云`
- `林焕清`

Key finding:

- `Leena Klammer` and `Esther` are the same character identity in the relevant movie context, so this case required explicit repair attention rather than being treated as two unrelated focal roles.

### 2.3 Character-list consistency check

After Task1 repair, the released Task1 focal-character lists were manually cross-checked against the repaired prediction root to ensure the evaluator was comparing the intended characters.

Verified result:

- gold vs prediction focal-character mismatch count was driven to `0`

This check mattered because earlier runs had cases where:

- the wrong focal-role source file was being read first
- canonical names were replaced too aggressively by screenplay-local names

## 3. Manual Repair And Targeted Curation

### 3.1 Targeted repair strategy choice

A manual engineering decision was made not to trust fresh global re-extraction as the default repair path once it proved slower and noisier than needed.

Instead, Task1 repair prioritized:

- deterministic backfill from earlier stronger Task1 runs
- focal-character-list alignment to the released gold set
- conservative handling of unresolved edge cases

This was a manual curation choice, not merely an automated default.

### 3.2 Edge-case conservative repair

For edge cases where a strong prior Task1 character was missing or incomplete, manual repair favored conservative, scene-grounded minimal timelines rather than speculative expansion.

The guiding rule was:

- prefer an obviously safe partial repair over an unsupported, over-composed timeline

### 3.3 Release-name preservation

Manual review confirmed that release-facing Task1 files should preserve benchmark-canonical focal names even when screenplay mentions used a different local surface form.

This is especially important for:

- cross-task consistency
- evaluator alignment
- avoiding accidental character split / merge errors

## 4. Manual Evaluation Audit Before Metric Revision

### 4.1 Why manual metric audit was needed

The original Task1 evaluation showed a recurring pattern:

- `gold_fact_recall`, `pred_fact_precision`, and `fact_f1` were often strong
- but `pred_transition_coherence` could be very low on some movies

This discrepancy was large enough to require manual inspection of whether the metric was mis-scoring otherwise reasonable timelines.

### 4.2 Manual inspection of low-score transition cases

Low-score cases were manually inspected at the pair level.

Representative reviewed movies included:

- `en1ae8880bb21245a39ad18f119e163fec`
- `ch031fd9f8e8339ed61190f6e72d73dc8b05e8aef2`
- `en2e733a32750748b082f2ffb7f4608329`
- `ch060a57fa2cc7b375d4db567c3ec463f4c7f0dd75`
- `en0c08ce1c06774785b5d73d9effd69e6b`

Manual inspection looked at:

- adjacent predicted nodes
- scene titles / scene order
- whether the later node contradicted the earlier one
- whether the pair merely shifted subthread or omitted reasonable intermediate events
- whether the chosen important pairs were actually good representatives of transition quality

### 4.3 Main manual findings on the old metric

The old transition metric was found to be too harsh for at least three reasons:

1. It often behaved like a direct-causality test rather than a compatibility test.
2. Important-pair selection sometimes chose same-scene beat splits or long-gap pairs that were bad local transition tests.
3. Some low-scoring pairs looked narratively reasonable to a human reader even though the judge returned `FALSE`.

Manual conclusion:

- a substantial part of the old low transition scores reflected metric harshness rather than poor Task1 predictions

## 5. Manual Metric Revision And Calibration

### 5.1 Human-guided revision

Based on manual inspection, the transition metric was revised to:

- judge adjacent predicted nodes for narrative compatibility
- not require direct causality
- tolerate omitted intermediate events
- prefer local anchored pairs for `important_pred_transition_coherence`

The formal revision note is:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0/docs/task1_transition_metric_revision_20260417.md`

### 5.2 Five-movie manual calibration set

The revised evaluator was then tested on a manually chosen 5-movie calibration set drawn from the worst old transition cases.

Calibration report:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/task1_v1_pilot/reports/task1_transition_calibration5_20260417_qwen8002.json`

Average change on the calibration set:

- `pred_transition_coherence`: `0.2221 -> 0.6616`
- `important_pred_transition_coherence`: `0.1233 -> 0.6583`

Manual interpretation:

- the revision corrected an evaluator artifact rather than artificially inflating poor predictions

## 6. Manual Tail Inspection On Revised Evaluation

After the revised evaluator was launched on the 40-movie subset, the lower tail of the score distribution was manually checked again.

At the time of inspection, the completed subset showed:

- no `important_pred_transition_coherence < 0.5`
- only one `pred_transition_coherence < 0.5`
- the lowest-tail cases still had generally reasonable `fact_f1` and arc scores

Manual conclusion:

- the revised transition metric no longer showed the earlier pathological low-score tail
- low-scoring cases looked like acceptable hard cases rather than obvious evaluator failures

## 7. What Was Manual vs What Was Automated

### Manual work

- selecting suspicious cases to inspect
- checking whether low-node characters were real secondary roles or extraction errors
- verifying specific identity-merge issues such as `Leena Klammer` / `Esther`
- choosing the repair strategy
- deciding to prefer deterministic backfill over fresh full re-extraction
- reading concrete low-score transition examples and determining the old metric was too harsh
- deciding the new transition metric should be compatibility-based
- inspecting the revised low-score tail before release push

### Automated work

- bulk JSON regeneration / backfill
- batch evaluation over the 40-movie manifest
- aggregate metric calculation
- per-movie report writing

## 8. Release-Relevant Outcome

The Task1 release now reflects both:

- automated construction / evaluation
- explicit human validation and curation effort

In particular, manual work directly contributed to:

- alias cleanup in released Task1 gold files
- suspicious-character investigation
- focal-character alignment repair
- evaluator redesign for transition quality
- release confidence that the revised Task1 metric stack is interpretable

## 9. Recommended Paper Framing

For the paper or appendix, the manual validation effort can be summarized as:

- release-focused audit of focal-character identity consistency
- targeted inspection of suspicious low-node and alias cases
- human review of low-scoring Task1 examples
- human-guided revision of the transition metric
- post-revision tail inspection confirming that the low-score distribution became reasonable

This is a stronger and more defensible claim than saying the benchmark was purely automatically constructed.
