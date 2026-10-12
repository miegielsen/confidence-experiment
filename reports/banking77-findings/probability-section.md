## Comparing the probability curves

Reliability plots answer whether a reported probability matches observed correctness. A point below the diagonal indicates overconfidence in that bin; a point above it indicates underconfidence. Each comparison holds the classifier's selected labels fixed. The histogram beneath each panel shows how much evidence supports its points, and nominal Wilson intervals show sampling uncertainty. The plots use shared equal-width bins without smoothing or connecting sparse estimates.

The second figure separates scores assigned to correct and incorrect predictions. Greater separation helps prioritize review, but does not by itself establish calibrated probabilities. Each outcome distribution is normalized separately; its legend reports the actual number of cases. Cross-classifier comparisons also change which examples are errors, so the within-classifier comparisons are the stronger method comparison.

The raw embedding scores are underconfident in several populated bins. Temperature scaling moves more predictions toward high confidence and reduces probability error. The Claude panels compare each model's original and learned confidence while keeping selected labels unchanged. Their score distributions and error counts show whether learned evidence separates mistakes from correct predictions. The tested OpenAI temperature transformation makes several bins underconfident and worsens Brier error. Sparse individual points should be interpreted with their counts and intervals.

| Method | Scores at least 90% | Errors in that group | Observed error rate |
|---|---:|---:|---:|
| Embedding LR calibrated | 899 | 4 | 0.44% |
| Haiku original | 503 | 37 | 7.36% |
| Haiku learned | 592 | 12 | 2.03% |
| OpenAI native probability | 511 | 39 | 7.63% |
| Sonnet original | 400 | 12 | 3.00% |
| Sonnet learned | 669 | 11 | 1.64% |
| Opus original | 359 | 4 | 1.11% |
| Opus learned | 720 | 14 | 1.94% |

The 90% cutoff is a descriptive reference point shared across methods. It is separate from the threshold chosen for the 1% automation criterion and does not constitute a deployment guarantee. Probability quality, error ranking, and coverage should be assessed together. The OpenAI temperature comparison uses log returned probabilities as a surrogate for native logits; the learned correctness comparisons also add supervised evidence and extra calls.
