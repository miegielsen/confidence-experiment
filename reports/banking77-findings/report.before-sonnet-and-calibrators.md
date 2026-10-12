# Learning When to Trust a Classifier

**BANKING77 experiment findings · 9 October 2026 · Development results**

A frozen embedding model with a linear classifier correctly classified **971 of 1,000 banking queries**. Haiku 5.5 classified 757 correctly; OpenAI Decisions classified 733 correctly. For this task, with labeled examples available, the supervised embedding approach is the strongest candidate for further validation.

The confidence experiment produced a second, distinct finding. Keeping every Haiku prediction unchanged, a small learned correctness model reduced its Brier error by **59%** and substantially improved error detection. The useful advance came from predicting correctness using observed outcomes and several inference-time signals. A separate verifier delivered a much smaller improvement, with uncertainty intervals that included no improvement.

The operational results show why both findings matter. The calibrated embedding classifier could accept **895 development predictions with three errors** under the experiment’s threshold-selection rule. Haiku’s learned confidence substantially improved triage, but none of its thresholds met the same stringent 1% error criterion. These are promising development findings; the official test set remains unevaluated.

## The experiment in brief

BANKING77 asks a classifier to assign short banking messages to 77 closely related intents. It is useful for studying confidence because a fluent, plausible interpretation can still select the wrong category. The original dataset contains 13,083 annotated examples. [Casanueva and colleagues, 2020](https://aclanthology.org/2020.nlp4convai-1.5/)

The data audit removed seven normalized training examples overlapping the official test text and four duplicate training rows. The remaining 9,992 non-test examples were divided into separate roles, using stratified splits and seed 42.

| Data role | Examples | Purpose |
|---|---:|---|
| Classifier fitting | 5,992 | Train the supervised classifiers |
| Correctness training | 2,000 | Learn when Haiku’s selected label is correct |
| Calibration | 1,000 | Fit probability transformations |
| Development | 1,000 | Compare methods and choose acceptance thresholds |
| Official test | 3,080 | Reserved for final evaluation |

All reported comparisons use the same 1,000 development messages and reference labels. The supervised models learned from 5,992 labeled examples. The LLM classifiers were zero-shot: they received the 77 intent names, without labeled demonstrations or richer category definitions. This comparison describes the implemented operating conditions; it does not isolate model architecture from access to task-specific labels.

## Frozen embeddings were the strongest classifier

| Classifier | Accuracy | Macro F1 | Errors out of 1,000 |
|---|---:|---:|---:|
| EmbeddingGemma plus logistic regression | **97.1%** | **0.972** | **29** |
| EmbeddingGemma plus a small MLP | 96.9% | 0.970 | 31 |
| TF-IDF plus logistic regression | 87.1% | 0.873 | 129 |
| Anthropic Haiku 5.5 | 75.7% | 0.756 | 243 |
| OpenAI Decisions using gpt-6-luna | 73.3% | 0.735 | 267 |

OpenAI’s unsuccessful predictions include one development refusal, retained in the evaluation denominator and routed to review.

The embedding classifier’s lead is substantial: **21.4 percentage points over Haiku** and **23.8 points over OpenAI Decisions**. Paired bootstrap intervals were 18.8–24.0 points and 21.1–26.5 points, respectively. Those gaps are large within this development sample, although they remain conditional on the fitted models and the particular prompting setup.

The linear head and MLP were effectively tied. Their accuracy difference was only 0.2 percentage points, with a paired interval from −0.6 to +1.0 points. The MLP also reached its iteration limit, so this run cannot establish the best achievable neural-head performance. It provides no persuasive reason to prefer the more complex head at present.

This is a useful practical result: a frozen representation and a modest supervised head are enough to reach high accuracy here. Full encoder fine-tuning was cancelled because of local resource constraints. The experiment therefore provides no evidence about its possible additional benefit.

## Learning correctness materially improved Haiku confidence

The cleanest confidence comparison holds the classifier fixed. Every Haiku method below uses the same 757 correct and 243 incorrect predictions; only the confidence assigned to each prediction changes.

| Confidence method | Brier error ↓ | Binary log loss ↓ | Error AUROC ↑ |
|---|---:|---:|---:|
| Haiku’s original same-call confidence | 0.1454 | 0.4434 | 0.815 |
| Separate Haiku verifier | 0.1388 | 0.4261 | 0.831 |
| Learned correctness model | **0.0596** | **0.2093** | **0.961** |
| Learned correctness plus sigmoid calibration | **0.0578** | 0.2100 | **0.961** |

Brier error is the average squared difference between estimated correctness probability and the observed correct/incorrect outcome. Lower is better. Error AUROC measures how well low confidence ranks incorrect predictions above correct predictions; 0.5 is chance ranking and 1.0 is perfect ranking. Brier error and log loss capture both calibration and discrimination, so they should be read alongside reliability plots. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

The learned model reduced Brier error from 0.1454 to 0.0596: a **59.0% relative reduction**. Its absolute improvement was 0.0858, with a paired 95% bootstrap interval of 0.0724–0.0990. Error AUROC improved by 0.146, with an interval of 0.120–0.173. Both comparisons support a substantial improvement on this sample.

The separate verifier’s Brier improvement was only 0.0066, with an interval of −0.0024 to +0.0156. Its AUROC improvement also had an interval spanning zero. Rechecking with the same model may help, but this experiment does not establish a reliable gain from verification alone.

The learned estimator used regularized logistic regression trained on the 2,000-example correctness partition. Its features were original confidence, verifier confidence, agreement with a blind second classification, input length, an empty-input flag, and two signals from the supervised TF-IDF classifier: support for Haiku’s chosen label and the margin between its top two classes. All were available without the development example’s true label. The blind classifier and verifier were additional Haiku calls; their errors need not be independent.

This qualification matters: the winning estimator combines learning, extra calls, and task-specific supervised evidence. There was no feature ablation, so the gain cannot be attributed to one ingredient. The completed run did not use embedding-neighbor features. Learning correctness from black-box model signals has research precedent, although earlier QA and summarization results do not establish performance on this task. [Large Language Model Confidence Estimation via Black-Box Access](https://arxiv.org/abs/2406.04370)

One diagnostic example illustrates the change. For “I would like to get a physical card, but I don’t know how. Can you help me?”, Haiku selected “get_physical_card” with 95% confidence. The reference label was “order_physical_card”; learned confidence fell to approximately 20%, and the embedding classifier matched the reference. This selected example shows how additional evidence can flag confident disagreements. It also exposes a benchmark limitation: closely related label names can be ambiguous, and measured correctness means agreement with the reference annotation.

## Better error ranking did not guarantee low-risk automation

The experiment selected the largest confidence-threshold acceptance set whose **upper endpoint of a 95% Wilson error interval was at most 1%**. This is stricter than requiring an observed error rate below 1%.

| Candidate | Accepted | Errors among accepted | Observed error rate | Development coverage |
|---|---:|---:|---:|---:|
| Embedding LR with temperature scaling | **895** | **3** | **0.34%** | **89.5%** |
| Embedding MLP with raw probabilities | 867 | 2 | 0.23% | 86.7% |
| TF-IDF, all tested scores | 0 | — | — | 0% |
| Haiku, all tested scores | 0 | — | — | 0% |
| OpenAI Decisions, all tested scores | 0 | — | — | 0% |

For the calibrated embedding LR, the selected confidence threshold was approximately 0.9173. The nominal Wilson interval for its accepted error rate was **0.11%–0.98%**. That threshold was selected using these same development outcomes, so the interval does not account for searching over thresholds. **The 89.5% figure is a candidate operating point, not a validated deployment guarantee.** Its performance must be measured with the threshold frozen on independent data.

Haiku shows why a strong AUROC is insufficient for this target. Even its 500 highest learned-confidence predictions contained eight errors: **1.6% observed error**. The score helped place errors toward the review end of the queue, but it did not produce a sufficiently clean accepted group under the selection rule. Zero accepted coverage here means no tested threshold qualified; it does not mean that every prediction was wrong or that the scores have no review value.

Native LLM confidence also contained large ties. A threshold accepting no more than half the examples accepted 381 Haiku predictions, with 19 errors. This cannot be treated as an exactly matched 500-example comparison. The risk–coverage figures preserve tied-score groups rather than breaking them arbitrarily.

## Calibration helped selectively

Temperature scaling improved the embedding LR’s Brier error from **0.0328 to 0.0208**, while retaining its predicted labels. Its paired improvement interval was 0.0082–0.0157. Binary log loss also fell from 0.1326 to 0.0710. This is a useful improvement from a simple transformation fitted on separate calibration examples. Temperature scaling is an established baseline for calibrating class scores. [Guo and colleagues, 2017](https://proceedings.mlr.press/v70/guo17a.html)

Other transformations were less helpful. Sigmoid calibration of Haiku’s original confidence changed Brier error only from 0.1454 to 0.1437; its paired improvement interval included zero. For learned correctness, sigmoid calibration slightly improved Brier error but slightly worsened log loss. A monotonic sigmoid transformation preserves the ordering of scores, so it cannot improve error AUROC through reranking.

OpenAI Decisions’ native confidence and selected-label probability behaved almost identically, with error AUROC of approximately 0.824. They did not provide two meaningfully distinct signals in this run. The tested temperature transformation worsened selected-label Brier error from **0.1474 to 0.1903**. Its input was the logarithm of returned probabilities clipped at 10⁻¹², used as a surrogate for native logits; fitting minimized multiclass log loss. The result applies to this transformation and objective. It does not establish that temperature scaling generally fails or that native logits were tested.

These results support evaluating every calibration step on held-out outcomes. Adding a calibrator does not, by itself, establish better probabilities or better automation.

## Why the embedding curves looked volatile

The calibrated embedding LR placed **899 of 1,000 predictions in the 90–100% confidence bin**. Only 101 predictions populated all lower-confidence bins combined. The 30–40% bin contained six examples, of which one was correct; the 40–50% bin contained eight, of which five were correct. One changed outcome moves those bin accuracies by 16.7 or 12.5 percentage points.

Connecting these estimates can make the curve look erratic even when the underlying score is useful. The immediate cause is sparse bin counts and concentrated confidence. High accuracy contributes by leaving only 29 errors, but high accuracy alone does not imply volatile calibration. The same small error count also makes risk–coverage curves advance in visible steps.

The appropriate response is to show counts and uncertainty, avoid smoothing away the scarcity of evidence, and obtain a larger representative evaluation sample. A harder representative task may generate more errors and reveal differences between confidence methods, but increasing evaluation size can improve precision without deliberately weakening classification. A representative document task would also test whether these findings transfer beyond short banking messages.

## What the findings justify next

**Advance frozen EmbeddingGemma plus logistic regression as the main candidate.** It had the best observed classification and automation results, and the MLP showed no convincing advantage. Freeze the classifier, temperature transformation, and selected threshold before evaluating the untouched official test set.

**Use learned correctness when retaining the LLM classifier has a business reason.** Its gain is substantial, but the next experiment should remove feature groups in turn: native confidence alone, added supervised TF-IDF evidence, blind agreement, and verifier evidence. This will identify which signals justify additional calls and labeled-data requirements. Compare that system against simply using the supervised classifier directly.

**Expand the evaluation around the intended deployment population.** Repeat the strongest methods on a larger document-classification task, audit ambiguous disagreements, and evaluate unsupported inputs separately. Confidence on known banking intents does not establish out-of-scope detection or calibration under distribution shift. Full encoder fine-tuning can remain deferred until there is a clear unresolved question that warrants its resource cost.

## Scope and reproducibility

These are development results from one split and one training seed. The 5,000-resample paired bootstrap uses the same sampled example IDs for each compared method and reports percentile intervals. It measures example-sampling uncertainty conditional on the fitted models and calibrators; it does not capture retraining variation, model selection, threshold search, or multiple-comparison adjustment. The confidence-method comparison within Haiku is stronger than a causal comparison across classifier families because its predictions are held fixed.

The API comparison covers the exact zero-shot configurations tested, not an optimized provider benchmark. Limited intent definitions, annotation ambiguity, and possible public-benchmark exposure during pretraining constrain interpretation. Two OpenAI calibration refusals and one development refusal were preserved; refusals received operational confidence zero and were always routed to review. Calibration and evaluation examples were not silently dropped to improve reported accuracy.

Early Haiku requests encountered truncation with adaptive thinking and a 256-token output limit. The comparison was restarted uniformly with thinking disabled; results do not mix selectively retained predictions from the two configurations. Historical calls remain in the cost ledger.

Recorded API spending was approximately **$2.25**, a token-based ledger estimate including exploratory and discarded calls. It excludes local compute and is not a billing invoice. Batch waiting and resumed cached work make the recorded run durations unsuitable for a clean provider speed comparison.

The accompanying analysis records input SHA-256 hashes and verifies matching development IDs and unchanged predictions within each LLM confidence family. The scientific source outputs are the development reports and prediction files under runs/embeddings, runs/pilot, runs/anthropic-batch-v2, and runs/openai-decisions-v2. The dataset revision is 57ec275d8078af65b7731c2a98be812d844a6d6b; the frozen embedding revision is 57c266a740f537b4dc058e1b0cda161fd15afa75. The report analysis uses bootstrap seed 20261009. No official test predictions were generated for this report.
