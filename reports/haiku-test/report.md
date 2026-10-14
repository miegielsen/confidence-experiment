# Haiku: independent test validation

This smaller sweep evaluates **12 frozen Haiku confidence variants on 3,080 official test messages**. Sonnet and Opus remain deferred. Models, calibrators, prompts, and development-selected thresholds are unchanged. No test threshold search or refitting was performed.

Classification accuracy was **75.70% on development and 78.18% on test**, a change of **+2.48 percentage points**. The nominal test accuracy interval is **76.69%–79.60%**. All confidence variants retain the same classifications.

Native confidence test Brier was **0.13207**; learned correctness plus the already-fitted logistic-on-score calibrator was **0.05948**. Their paired difference, learned minus native, has nominal 95% interval **[-0.07993, -0.06502]**. Negative differences favor learned correctness. That pipeline combines extra API calls and supervised TF-IDF features, so this comparison does not isolate the effect of learning alone.

| Confidence variant | Dev Brier | Test Brier | Test error AUROC | Test coverage | Accepted errors / accepted |
|---|---:|---:|---:|---:|---:|
| haiku_same.raw | 0.14536 | 0.13207 | 0.823 | 0.00% | 0 / 0 |
| haiku_same.logistic_score | 0.14375 | 0.13269 | 0.823 | 0.00% | 0 / 0 |
| haiku_same.logistic_logit | 0.14168 | 0.13122 | 0.823 | 0.00% | 0 / 0 |
| haiku_same.isotonic | 0.14164 | 0.13096 | 0.823 | 0.00% | 0 / 0 |
| haiku_verifier.raw | 0.13878 | 0.13805 | 0.811 | 0.00% | 0 / 0 |
| haiku_verifier.logistic_score | 0.14114 | 0.13727 | 0.811 | 0.00% | 0 / 0 |
| haiku_verifier.logistic_logit | 0.13775 | 0.13569 | 0.811 | 0.00% | 0 / 0 |
| haiku_verifier.isotonic | 0.13667 | 0.13574 | 0.810 | 0.00% | 0 / 0 |
| haiku_correctness.raw | 0.05958 | 0.06099 | 0.957 | 0.00% | 0 / 0 |
| haiku_correctness.logistic_score | 0.05780 | 0.05948 | 0.957 | 0.00% | 0 / 0 |
| haiku_correctness.logistic_logit | 0.05851 | 0.06004 | 0.957 | 0.00% | 0 / 0 |
| haiku_correctness.isotonic | 0.05851 | 0.05992 | 0.957 | 0.00% | 0 / 0 |


Zero acceptance means the frozen development policy selected no threshold. It does not establish that no possible threshold could work. Any nonzero acceptance policy must be assessed using its saved error interval in evidence.json. Low Brier error does not itself establish a 1% accepted-error guarantee.

The sweep recorded **9,240 unique requests** and an estimated **$0.8642** in batch API fees. Native-only classification costs **$0.1116 per 1,000 messages**; classification plus verification costs **$0.1705**; the full learned-correctness pipeline costs **$0.2806**. Calibrators reuse responses and add no calls. These estimates use returned usage at configured rates, not invoice totals; labeling, fitting, compute, and review costs are excluded.

The accompanying test-cost-frontiers.csv and evidence.json combine these 12 variants with the previous 24 local/OpenAI test variants. Frontiers use point estimates, retain ties, and exclude Sonnet/Opus development results. Paired intervals use 5,000 shared example resamples and condition on fitted models; they omit fitting variability and are not adjusted for multiple comparisons. The test includes a normalized duplicate pair, making independence assumptions approximate.

The main local/OpenAI report is preserved at ../banking77-findings/report.md. This addendum supplies the newly validated Haiku results without changing the original primary-policy conclusion.
