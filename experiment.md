**I would start with BANKING77, then repeat the strongest approaches on the LEDGAR subset of LexGLUE.** Add CLINC150 if you also want to test whether confidence detects inputs outside the supported categories.

Your five approaches make sense. I would add two inexpensive controls: **post-hoc calibration of the LLM’s own confidence**, and **logistic regression on frozen embeddings** before trying a neural network.

I’m assuming the input is text, including text extracted from documents. These datasets let you study classification and confidence without introducing OCR or visual-layout differences.

**The dataset choices are:**

| Dataset | Content and size | Why I would use it | Main limitation |
|---|---|---|---|
| **[BANKING77](https://huggingface.co/datasets/PolyAI/banking77)** | Short banking queries; **77 intents**, **10,003 training / 3,080 test** examples. | Best first experiment: inexpensive LLM calls, relevant domain, closely related categories. | Short messages rather than documents; documented potential label errors. [GitHub](https://github.com/PolyAI-LDN/task-specific-datasets?utm_source=chatgpt.com) |
| **[LEDGAR through LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue)** | Contract provisions/paragraphs; **100 categories**, **60,000 train / 10,000 validation / 10,000 test**. | Better match for document intelligence, with more data for training a secondary confidence model. The official splits are chronological. | Legal English and paragraph classification, rather than whole-document classification. Use the **LexGLUE version**; the original LEDGAR task differs. [aclanthology.org](https://aclanthology.org/2022.acl-long.297.pdf?utm_source=chatgpt.com) |
| **[CLINC150](https://github.com/clinc/oos-eval)** | **150 supported intents**: 15,000 train / 3,000 validation / 4,500 test in-scope examples, plus explicit out-of-scope examples, including **1,000 in the test set**. | Tests whether a model knows when none of its supported categories applies. | Short assistant queries; confidence on known classes and out-of-scope detection need separate evaluation. [GitHub](https://github.com/clinc/oos-eval?utm_source=chatgpt.com) |

BANKING77’s label quality deserves attention specifically because your target is *“was the model correct?”* A defensible prediction that disagrees with an ambiguous reference label becomes an incorrect example for the confidence model. Keep the official labels for comparable benchmark results, and separately audit a sample of disagreements.

For conclusions about your enterprise projects, public benchmarks should be a first screening step. They cannot establish calibration on your own document population, and prior exposure during model pretraining is difficult to rule out.

**Define confidence consistently across all five approaches.**

For your purpose, the common target should be:

\[
z_i=\mathbf{1}[\hat y_i=y_i], \qquad
q_i \approx P(z_i=1\mid\text{information available at inference})
\]

In words: **the probability that this particular selected label is correct**. You do not need a full probability distribution over all categories to compare that target.

Then implement the approaches as follows:

| Approach | Practical first implementation | What to watch |
|---|---|---|
| **1. Same-call LLM confidence** | Return a valid label and a probability of that label being correct, using structured output. | Treat the number as a raw score until tested; wording and model version can change its behavior. |
| **2. Separate LLM confidence** | Give another LLM the input, category definitions and proposed label. Ask for probability of correctness; hide the initial confidence and rationale. | A separate call does **not** imply statistically independent errors. |
| **3. Learned correctness model** | Train regularized logistic regression on inference-time signals, targeting correct/incorrect. Try boosted trees after establishing this baseline. | Leakage, insufficient incorrect examples, and changes to the underlying classifier. |
| **4. Fine-tuned encoder classifier** | Fine-tune **ModernBERT-base** with a classification head. Retain its logits for calibration. | Softmax probabilities also need empirical calibration. |
| **5. Frozen embeddings + classifier** | **EmbeddingGemma-300m → multinomial logistic regression**, then compare with a small MLP with one hidden layer. | A nonlinear head is useful only if it improves held-out results over the linear baseline. |

ModernBERT-base is a concrete modern BERT-style option: its official model card describes an English/code-pretrained encoder with support for up to 8,192 tokens. Fine-tuning the pretrained encoder is sufficient; there is no need to train from scratch. [Hugging Face](https://huggingface.co/answerdotai/ModernBERT-base?utm_source=chatgpt.com)

EmbeddingGemma-300m is a suitable frozen embedding baseline. Its model card explicitly supports classification and recommends the input prefix `task: classification | query: {content}`. [Hugging Face](https://huggingface.co/google/embeddinggemma-300m?utm_source=chatgpt.com)

For method 2, I would also test a **blind second classification**: the second LLM chooses its own label before seeing the first label. Agreement/disagreement can then become a feature for method 3. This reduces anchoring, although shared model weaknesses can still produce confident agreement on the wrong label.

For method 3, start with a small, interpretable feature set:

| Feature | What it might capture |
|---|---|
| Original LLM confidence | The classifier’s own uncertainty signal |
| Second LLM confidence and label agreement | External support or disagreement |
| Stability across a few controlled prompt/label-order variations | Sensitivity of the decision |
| Nearest-neighbor distance and neighborhood label agreement in embedding space | Similarity to known training examples |
| Embedding classifier’s support for the LLM-selected label and top-two margin | An alternative model’s evidence |
| Input length, missing-content flags, predicted category | Systematic difficulty differences |

All features must be available without the example’s true label. Use only the classifier-training partition as the labeled nearest-neighbor reference set.

Your third approach has methodological precedent: *Large Language Model Confidence Estimation via Black-Box Access* trains logistic regression on engineered signals to estimate confidence. Its experiments concern QA and summarization, so it supports the approach rather than proving it will work for banking classification. [arxiv.org](https://arxiv.org/abs/2406.04370?utm_source=chatgpt.com)

**The most useful extra comparison is “raw score versus calibrated score.”**

For methods 1 and 2, fit a simple sigmoid calibrator mapping the raw confidence to observed correctness. For methods 4 and 5, start with temperature scaling of the class logits. Temperature scaling is a well-established baseline for neural-network calibration. [proceedings.mlr.press](https://proceedings.mlr.press/v70/guo17a?utm_source=chatgpt.com)

This separates two questions:

- **Does the score rank difficult cases well?**
- **Does a reported 90% correspond to roughly 90% correctness?**

A score can rank errors well while being overly optimistic; simple recalibration may fix that. Conversely, a model can output the overall accuracy for every example and be calibrated in aggregate, while offering no useful guidance for human review.

I would also avoid assuming that verbalized LLM confidence must perform poorly. Research has found settings where it is better calibrated than token probabilities. Your proposed experiment is worthwhile precisely because that behavior needs checking on the actual task. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.330/?utm_source=chatgpt.com)

**For BANKING77, a straightforward experiment would use five separate data roles:**

| Role | Approximate allocation |
|---|---:|
| Fit supervised classifiers; supply LLM examples/retrieval references | 6,000 |
| Train the correctness model | 2,000 |
| Fit post-hoc calibrators | 1,000 |
| Develop prompts, tune models and choose acceptance thresholds | 1,000 |
| Final evaluation | Official 3,080-example test set |

Freeze prompts and classifier settings before generating the final correctness-model training data. For methods **1–3, cache the same initial LLM predictions**, then add confidence estimation: their classification accuracy should be identical, allowing you to isolate the quality of the confidence method.

The main constraint is the number of errors. **At 95% accuracy, 2,000 examples provide only about 100 incorrect predictions.** That is a modest sample for learning an error model, particularly with 77 categories. Start with few features; move to LEDGAR or carefully designed cross-fitting if this becomes limiting. If you oversample errors or use class weighting, restore calibration against the natural error prevalence.

Record whether the LLM is zero-shot or few-shot alongside the supervised training budget. Those are different operating conditions, and both are useful to compare.

**Evaluate classification, probability quality and automation usefulness separately.**

| Question | Measurements |
|---|---|
| Is classification good? | Accuracy, macro-F1, per-category errors |
| Are probabilities meaningful? | Binary Brier score/log loss against correct–incorrect; reliability plots with counts and uncertainty intervals |
| Does confidence identify errors? | Error-detection AUROC and precision–recall |
| Can it support automation? | Risk–coverage curve: error rate among accepted predictions as you increase automatic-processing coverage |
| Does it reject unsupported inputs? | Separate out-of-scope detection results on CLINC150 |
| What does it cost? | Calls, latency and cost per example |

Brier score and log loss reflect both calibration and discrimination, so use them alongside reliability plots rather than interpreting them as pure calibration measures. [scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com)

For your use cases, I would make **“coverage achievable at a chosen maximum error rate”** the principal operational result—for example, how much can be processed automatically while accepted predictions stay below 1% error. Select thresholds on development data and report the final test error with a confidence interval.

My strongest starting candidates would be **EmbeddingGemma plus logistic regression** as the inexpensive supervised baseline, and **your LLM plus a small learned correctness model** when preserving the LLM classifier matters. Whether either wins on calibration or usable automation coverage is exactly what this experiment should establish.