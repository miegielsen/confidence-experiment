# Learning When to Trust a Classifier

**BANKING77 findings · 13 October 2026 · Independent test validation for local models and OpenAI**

The independent test set changed the most important conclusion. The primary EmbeddingGemma classifier retained high accuracy, **94.74%**, but its frozen automation policy accepted **2,715 messages with 53 errors**. That is **1.95% error**, with a nominal 95% interval of **1.50–2.54%**. The intended maximum was 1%. **The policy failed independent validation.**

Development had suggested that temperature-scaled embedding LR could accept 89.5% of messages with 3 errors in 895 accepted predictions. On the official test set, coverage was **88.15%**, with substantially more accepted errors. The development result was useful for choosing a candidate; it overstated the evidence for dependable 1% automation.

All **24 local and OpenAI score variants** were evaluated on the same **3,080 official test messages**, using the existing fitted models, calibrators, and development-selected thresholds. No test outcomes were used to refit a calibrator or choose a replacement threshold. Claude test evaluation is deferred; Haiku, Sonnet, and Opus findings below remain development-only.

## The experiment in brief

BANKING77 asks a classifier to assign short banking messages to 77 closely related intents. It is useful for studying confidence because a fluent, plausible interpretation can still select the wrong category. The original dataset contains 13,083 annotated examples. [Casanueva and colleagues, 2020](https://aclanthology.org/2020.nlp4convai-1.5/)

The data audit removed seven normalized training examples overlapping the official test text and four duplicate training rows. The remaining 9,992 non-test examples were divided into separate roles, using stratified splits and seed 42.

The official test retains one normalized duplicate pair; independence-based uncertainty intervals are approximate. Test labels first entered model evaluation during this final stage.

| Data role | Examples | Use |
|---|---|---|
| Classifier fitting | 5992 | Fit TF-IDF and embedding heads |
| Correctness training | 2000 | Fit the historical Claude correctness estimators |
| Calibration | 1000 | Fit temperature, logistic and isotonic calibrators |
| Development | 1000 | Compare candidates and select thresholds |
| Official test | 3080 | Independent evaluation with all choices fixed |

The final plan sealed 60 variants before test inference. At the user's request, execution was restricted to the local models and OpenAI: 24 variants across five confidence streams. The remaining 36 Claude variants retain their frozen specifications but have no test results. This execution change did not alter any fitted model, calibrator, or threshold.

## Classification on independent data

| Classifier | Development accuracy | Test accuracy | Test accuracy 95% interval | Test errors | Test macro F1 |
|---|---|---|---|---|---|
| EmbeddingGemma LR | 97.10% | 94.74% | 93.89–95.47% | 162 | 0.947 |
| EmbeddingGemma MLP | 96.90% | 93.90% | 92.99–94.69% | 188 | 0.939 |
| TF-IDF LR | 87.10% | 87.08% | 85.85–88.22% | 398 | 0.871 |
| OpenAI probability | 73.30% | 76.46% | 74.93–77.93% | 725 | 0.759 |

Embedding LR remained more accurate than the tested alternatives. Its paired test accuracy difference from the MLP was **+0.84 percentage points**, with a nominal 95% interval of **0.42–1.30%**. These models use the same fixed embedding representation, so this is a comparison of the fitted classifier heads, not evidence that nonlinear heads cannot help with other settings or seeds.

The embedding LR made **162 test errors**, compared with 29 development errors. The two datasets differ in size and difficulty: the error rate rose from 2.9% to 5.26%. TF-IDF remained near its earlier 87.1% accuracy. The official test results therefore make the earlier embedding advantage smaller without eliminating it.

OpenAI Decisions correctly classified **2,355 of 3,080 messages (76.46%)**. Its **7 refusals** remain in the denominator as unsuccessful classifications and always require review. API-selected labels are preserved even when a returned probability distribution has a different argmax. The local classifiers received 5,992 fitting labels; OpenAI received zero-shot intent names, so cross-family accuracy differences combine model choice and access to supervision.

## The frozen automation policy failed its target

| Frozen policy | Accepted | Errors | Test coverage | Accepted error rate | 95% error interval |
|---|---|---|---|---|---|
| EmbeddingGemma LR Raw | 2683 | 53 | 87.11% | 1.98% | 1.51–2.57% |
| EmbeddingGemma LR Temperature | 2715 | 53 | 88.15% | 1.95% | 1.50–2.54% |
| EmbeddingGemma LR Logistic log-odds | 2683 | 53 | 87.11% | 1.98% | 1.51–2.57% |
| EmbeddingGemma LR Isotonic | 2510 | 36 | 81.49% | 1.43% | 1.04–1.98% |
| EmbeddingGemma MLP Raw | 2629 | 46 | 85.36% | 1.75% | 1.31–2.33% |
| EmbeddingGemma MLP Temperature | 2615 | 44 | 84.90% | 1.68% | 1.26–2.25% |
| EmbeddingGemma MLP Isotonic | 2425 | 28 | 78.73% | 1.15% | 0.80–1.66% |
| TF-IDF LR Temperature | 0 | 0 | 0.00% | — | — |
| OpenAI probability Logistic log-odds | 0 | 0 | 0.00% | — | — |

The primary temperature threshold was fixed at **0.91727740** before any test prediction was scored. Its error interval lies entirely above 1%, so this failure is more than an upper interval merely crossing the target. The system would send 365 of 3,080 messages to review while leaving 53 incorrect classifications among the automatically accepted messages.

Isotonic embedding LR accepted fewer messages, **2,510**, but still made **36 errors (1.43%)**. Every tested embedding policy had a 95% upper error endpoint above 1%. A more flexible calibrator did not rescue the existing automation rule.

TF-IDF and all OpenAI variants retained null acceptance thresholds because no threshold met the development criterion. Their zero test coverage follows that frozen policy. It does not establish that every possible test threshold would fail; searching the test curves for a new threshold would turn this final set into tuning data.

The original selection rule searched development thresholds for the largest acceptance set whose upper 95% Wilson error endpoint was at most 1%. A nominal interval at a threshold chosen using the same outcomes does not account for that search. The independent test exposed the gap between a promising development operating point and a validated operational policy.

## Temperature logistic and isotonic calibration

Brier error is the average squared difference between estimated correctness probability and the observed correct/incorrect outcome. Lower is better. Error AUROC measures how well low confidence ranks incorrect predictions above correct predictions; 0.5 is chance ranking and 1.0 is perfect ranking. Brier error and log loss capture both calibration and discrimination, so they should be read alongside reliability plots. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

The scalar comparison adds two logistic calibrators and isotonic regression. Logistic on score learns a slope and intercept for the original confidence; it is the sigmoid calibration already used in the LLM runs. Logistic on log-odds instead uses logit(p) as its feature. Both use regularization C=1. Isotonic learns a flexible non-decreasing mapping, with greater risk of overfitting when evidence is sparse. It can introduce tied scores and change ranking metrics. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

The table compares binary correctness Brier error on official test data. Within each row, every variant retains the same predicted labels. Smaller values are better. Temperature optimizes multiclass calibration loss; the scalar methods target selected-label correctness. Their objectives differ, so a binary Brier result does not establish superiority of the full class distribution.

| Confidence source | Raw | Temperature | Logistic score | Logistic log-odds | Isotonic |
|---|---|---|---|---|---|
| EmbeddingGemma LR | 0.04777 | 0.03957 | 0.04026 | 0.04043 | 0.04168 |
| EmbeddingGemma MLP | 0.04336 | 0.04259 | 0.05179 | 0.04442 | 0.04487 |
| TF-IDF LR | 0.14838 | 0.07719 | 0.08654 | 0.08550 | 0.08799 |
| OpenAI confidence | 0.13326 | — | 0.13422 | 0.13201 | 0.13383 |
| OpenAI probability | 0.13326 | 0.19126 | 0.13419 | 0.13203 | 0.13383 |

For embedding LR, temperature reduced Brier error from **0.04777 to 0.03957**. Its test binary log loss was **0.14693**. Isotonic minus temperature Brier was **+0.00211**, with a paired nominal 95% interval of **[+0.00096, +0.00327]**. Positive differences favor temperature. This comparison uses frozen fits and provides no license to choose a new deployment configuration after looking at test outcomes.

The result supports the usefulness of simple scaling while showing its limit: improved average probability quality did not ensure that a high-confidence subset had less than 1% error. Temperature cannot repair missing information or guarantee that mistakes will be ranked below every accepted case. Logistic maps can adjust slope and intercept; positive slopes preserve raw score ranking. Isotonic can introduce tied groups and exact zero/one probabilities. Log loss clips at 10⁻⁷ consistently with the earlier analyses.

All calibrators still learned from only 1,000 calibration examples. More test mistakes improve evaluation precision, but they do not add training evidence to an already fitted isotonic model. The calibration partition contained only 39 embedding LR errors, which remains a constraint on flexible fitted mappings.

## Comparing the probability curves

Reliability plots answer whether a reported probability matches observed correctness. A point below the diagonal indicates overconfidence in that bin; a point above it indicates underconfidence. Each comparison holds the classifier's selected labels fixed. The histogram beneath each panel shows how much evidence supports its points, and nominal Wilson intervals show sampling uncertainty. The plots use shared equal-width bins without smoothing or connecting sparse estimates.

The second figure separates scores assigned to correct and incorrect predictions. Greater separation helps prioritize review, but does not by itself establish calibrated probabilities. Each outcome distribution is normalized separately; its legend reports the actual number of cases. Cross-classifier comparisons also change which examples are errors, so the within-classifier comparisons are the stronger method comparison.

## Why the embedding curves looked volatile

Connecting these estimates can make the curve look erratic even when the underlying score is useful. The immediate cause is sparse bin counts and concentrated confidence. High accuracy contributes by leaving only 29 errors, but high accuracy alone does not imply volatile calibration. The same small error count also makes risk–coverage curves advance in visible steps.

The appropriate response is to show counts and uncertainty, avoid smoothing away the scarcity of evidence, and obtain a larger representative evaluation sample. A harder representative task may generate more errors and reveal differences between confidence methods, but increasing evaluation size can improve precision without deliberately weakening classification. A representative document task would also test whether these findings transfer beyond short banking messages.

For embedding LR it was **0.903** after temperature scaling; for embedding MLP temperature it was **0.910**. Ranking, average probability accuracy, and performance at the frozen acceptance threshold should be assessed separately.

## Cost and Pareto frontiers

| Fixed representative pipeline | API USD per 1,000 | Test accuracy | Test Brier | Test error AUROC |
|---|---|---|---|---|
| EmbeddingGemma LR | $0.0000 | 94.74% | 0.03957 | 0.903 |
| EmbeddingGemma MLP | $0.0000 | 93.90% | 0.04259 | 0.910 |
| TF-IDF LR | $0.0000 | 87.08% | 0.07719 | 0.896 |
| OpenAI probability | $0.0690 | 76.46% | 0.13203 | 0.828 |

The OpenAI test cohort used **3,080 distinct requests** and recorded **$0.2125** in API fees. Duplicate-message cache reuse is counted once for cohort costs. The per-1,000 figure divides those unique recorded fees by all 3,080 evaluated messages. All OpenAI calibrators reuse the same predictions and add no API requests. Fee amounts are client-ledger estimates computed from returned token usage at the configured rate; they are not invoice totals. No Claude test batch was accepted or charged.

Local provider fees are zero; local computation, labeling, setup, and human review are not free. On this CPU, encoding the 3,080 test messages took **183.3 seconds** with two threads and batch size eight, excluding **4.4 seconds** to load the pinned encoder. This is one observed batch-throughput measurement, not a service-latency or electricity-cost estimate. The embedding heads reuse those encodings.

The test Pareto sets include the 24 completed variants only. A point is dominated when another costs no more, performs at least as well, and improves at least one selected objective. Accuracy, Brier, error ranking, and coverage have separate frontiers. Exact equivalents remain together; small differences are descriptive point estimates. Claude development points are excluded from test frontiers because mixing evaluation datasets would create a misleading comparison.

For a deployment of N messages, let F be fixed setup and annotation cost in USD, v local variable cost per message, A API fees per 1,000, r the fraction sent to human review, and R the cost of reviewing one message. Then **total USD per 1,000 = A + 1,000v + 1,000F/N + 1,000rR**. No measured values for F, v or R are available here. The calibrated supervised classifiers used 5,992 fitting labels and 1,000 calibration labels. Learned LLM correctness used the same TF-IDF fitting labels plus 2,000 correctness and 1,000 calibration labels; calibrating native LLM scores required 1,000 labeled outcomes. Those requirements matter when applying the experiment to a new task.

A policy's review savings cannot justify claiming a 1% quality target that failed independent validation.

## Claude findings remain development only

| Claude model | Development accuracy | Native Brier | Learned Brier | Learned plus logistic Brier | Learned API USD per 1,000 |
|---|---|---|---|---|---|
| Haiku 5.5 | 75.70% | 0.14536 | 0.05958 | 0.05780 | $0.28 |
| Sonnet 5.5 | 83.00% | 0.12007 | 0.05314 | 0.05052 | $5.63 |
| Opus 5.5 | 87.60% | 0.09720 | 0.05092 | 0.04991 | $12.34 |

Learned correctness improved the development confidence scores for Haiku, Sonnet, and Opus while keeping their selected labels unchanged. It combined original confidence, verifier confidence, blind-classification agreement, length/missing-content signals, and supervised TF-IDF support. Those gains cannot be attributed to learning alone because extra calls and supervised features were added together, with no feature ablation.

The historical Opus configuration used adaptive thinking and a 2,048-token output allowance; Sonnet and Haiku used their supported thinking settings and 256 tokens. These are comparisons of implemented configurations. The learned pipeline requires three calls per message; full Claude test replication was estimated near $56 for all three models combined, predominantly Opus. These development findings have not yet been tested on the official test messages and should not be compared directly with local/OpenAI test metrics as a provider ranking.

The full illustrated development study is preserved as [HTML](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.html), [PDF](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.pdf), and [Markdown](https://github.com/miegielsen/confidence-experiment/blob/main/reports/banking77-findings/report.development.md), including all 60 calibration variants, Opus, Sonnet, probability diagnostics, and the separate development cost frontier. The historical observations remain useful; the independent test changes the automation recommendation.

## What the findings justify next

EmbeddingGemma LR remains the strongest tested classifier by accuracy in this implementation. Temperature scaling remains useful for average probability quality. **The existing high-coverage policy should not be deployed as a validated 1% error system.** Improving classification, gathering more representative calibration/threshold-selection data, or using a selection procedure with explicit risk-control assumptions are reasonable next experiments. Any revised policy requires new independent validation data; the official test outcomes are now known for the completed arms.

**Expand the evaluation around the intended deployment population.** Repeat the strongest methods on a larger document-classification task, audit ambiguous disagreements, and evaluate unsupported inputs separately. Confidence on known banking intents does not establish out-of-scope detection or calibration under distribution shift. Full encoder fine-tuning can remain deferred until there is a clear unresolved question that warrants its resource cost.

Multiple training/split seeds and feature ablations would separate stable effects from one fitted configuration. Full encoder fine-tuning remains deferred; this experiment supplies no evidence about its benefit.

## Scope and reproducibility

The paired intervals use 5,000 shared test-example resamples and condition on the frozen fits. They omit retraining/calibration-training variability and are not adjusted for all comparisons. The primary policy was declared before testing; the remaining tables and Pareto frontiers describe the prespecified variants rather than establishing universal winners. Public benchmark exposure, ambiguous intent labels, and differences in supervision constrain transfer to enterprise documents.

The final runner validates model, calibrator, dataset, dependency, and scientific-code hashes before API work. It records completion only after result artifacts are written; report generation is separate and can be repeated without inference or paid requests. All results retain per-example predictions, reliability bins, risk curves, fixed thresholds, source hashes, and the sealed plan.
