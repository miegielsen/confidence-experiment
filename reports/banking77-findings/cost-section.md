## Cost and Pareto frontiers

**Embeddings remain the strongest candidate on accuracy and API fees; the LLM trade-off depends on which confidence objective matters.** This is an API-fee comparison, not a claim that local computation or labeled data are free. Costs below are reconstructed from the recorded token usage for the same 1,000 development messages, counting only the calls each inference pipeline needs. Calibration transforms add no API calls. Learned correctness requires the original, blind, and verifier calls plus local TF-IDF support; verification alone requires the original and verifier calls.

| Inference pipeline | API calls per message | Recorded API USD per 1,000 | Standard-rate proxy | Accuracy | Brier error |
|---|---:|---:|---:|---:|---:|
| TF-IDF LR | 0 | $0.0000 | $0.0000 | 87.1% | 0.07541 |
| Embedding LR | 0 | $0.0000 | $0.0000 | 97.1% | 0.02081 |
| Embedding MLP | 0 | $0.0000 | $0.0000 | 96.9% | 0.02235 |
| OpenAI probability | 1 | $0.0692 | $0.0692 | 73.3% | 0.14400 |
| Haiku native | 1 | $0.1118 | $0.2235 | 75.7% | 0.14168 |
| Haiku verifier | 2 | $0.1707 | $0.3414 | 75.7% | 0.13775 |
| Haiku learned | 3 | $0.2810 | $0.5619 | 75.7% | 0.05780 |
| Sonnet native | 1 | $2.2390 | $4.4646 | 83.0% | 0.10652 |
| Sonnet verifier | 2 | $3.4214 | $6.8224 | 83.0% | 0.10735 |
| Sonnet learned | 3 | $5.6270 | $11.2204 | 83.0% | 0.05052 |
| Opus native | 1 | $4.6311 | $9.2354 | 87.6% | 0.08363 |
| Opus verifier | 2 | $7.6868 | $15.3233 | 87.6% | 0.08111 |
| Opus learned | 3 | $12.3428 | $24.6089 | 87.6% | 0.04991 |

The table uses temperature for TF-IDF and embedding LR, raw embedding MLP, logistic on log-odds for OpenAI, Haiku native and all verifiers, and logistic on score for Sonnet and Opus native and all learned pipelines. All 60 variants enter the frontier calculation; these representative fixed choices are not assertions that each is the best calibrator. Local inference remains unpriced. Sonnet and Opus each reuse three synchronous smoke-test messages per call stage; their recorded costs are slightly above all-batch estimates. The standard-rate proxy reprices identical recorded tokens and does not measure synchronous quality or latency. Opus token costs include its adaptive reasoning.

Claude batch processing discounted input and output charges by 50%. Haiku’s standard input/output rates were $0.10/$0.50 per million tokens and Sonnet’s were $2/$10, and Opus’s were $4/$20. Decisions used $0.10 per million chargeable input tokens, with no output or cache charges. These rates were checked on 9 October 2026; workload length, caching and future rates can change costs. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI Decisions pricing](https://developers.openai.com/api/docs/guides/decisions)

A method is **Pareto dominated** for a chosen pair of objectives when another tested method costs no more, performs at least as well, and improves at least one objective. Equivalent points are retained together. The frontiers use development point estimates and the same cost denominator; tiny metric differences can move membership and do not establish statistical superiority. Brier comparisons across classifiers also reflect different error prevalence and discrimination, rather than pure calibration quality.

For **hosted classification accuracy**, the empirical frontier is OpenAI confidence ($0.0692/1,000; 73.3%) → Haiku native ($0.1118/1,000; 75.7%) → Sonnet native ($2.2390/1,000; 83.0%) → Opus native ($4.6311/1,000; 87.6%). Verifier and learned correctness variants retain the base classifier’s selected labels, so extra calls cannot improve accuracy. They can earn their cost through better probability estimates or review ranking.

For **hosted cost versus Brier error and error ranking**, the updated frontier members below include Opus. Exact equivalent cost–metric coordinates appear once; strictly increasing logistic variants can tie in AUROC. Membership is determined from all tested variants, rather than from the representative cost table. A method absent from one frontier may remain useful for another objective. The joint cost–accuracy–Brier frontier is retained in the evidence file.

| Objective | Frontier pipeline and calibrator | API USD per 1,000 | Development metric |
|---|---|---:|---:|
| Brier error ↓ | OpenAI probability · logistic logit | $0.0692 | 0.14400 |
| Brier error ↓ | Haiku native · isotonic | $0.1118 | 0.14164 |
| Brier error ↓ | Haiku verifier · isotonic | $0.1707 | 0.13667 |
| Brier error ↓ | Haiku learned · logistic score | $0.2810 | 0.05780 |
| Brier error ↓ | Sonnet learned · logistic score | $5.6270 | 0.05052 |
| Brier error ↓ | Opus learned · logistic score | $12.3428 | 0.04991 |
| Error AUROC ↑ | OpenAI confidence · raw | $0.0692 | 0.82427 |
| Error AUROC ↑ | Haiku verifier · raw | $0.1707 | 0.83051 |
| Error AUROC ↑ | Haiku learned · raw | $0.2810 | 0.96125 |

When local models enter the **API-fee-only** comparison, temperature-scaled embedding LR is at least as good on accuracy, Brier, error AUROC and exploratory coverage as **45 of 45 hosted variants**, with lower API fees. This is provider-fee dominance, not total economic dominance. The highest hosted development coverage under the Wilson rule is **0.0%**, from Haiku native with raw. These development-selected acceptance results require independent validation.

## Local cost sensitivity and study spending

For a deployment of N messages, let F be fixed setup and annotation cost in USD, v local variable cost per message, A API fees per 1,000, r the fraction sent to human review, and R the cost of reviewing one message. Then **total USD per 1,000 = A + 1,000v + 1,000F/N + 1,000rR**. No measured values for F, v or R are available here. The calibrated supervised classifiers used 5,992 fitting labels and 1,000 calibration labels. Learned LLM correctness used the same TF-IDF fitting labels plus 2,000 correctness and 1,000 calibration labels; calibrating native LLM scores required 1,000 labeled outcomes. Those requirements matter when applying the experiment to a new task.

Holding the observed hosted fees fixed, embedding LR can retain pairwise cost-and-accuracy dominance over OpenAI, Haiku native, or Sonnet native only while its all-in local cost is no higher than approximately **$0.0692, $0.1118, or $2.2390 per 1,000 messages**, respectively. The scenario figure assumes local costs of $0, $0.10, $1 or $3 per 1,000 and recomputes the accuracy frontier. The comparison now also includes Opus native at **$4.6311/1,000 and 87.6% accuracy**. Pairwise cost-and-accuracy dominance additionally requires the embedding accuracy to be at least as high. These are sensitivity assumptions, not estimates for this computer. Hosted setup and review costs would also have to enter a complete deployment comparison.

Under the development-selected temperature threshold, embedding LR would send **105 of 1,000 messages to review**. The highest-coverage hosted variant would send **1000**, using the same strict Wilson rule. Review cost may outweigh token charges, but these workloads are exploratory development results requiring independent validation. Cached embedding training durations cannot price fresh-message CPU inference or electricity; a fresh inference benchmark is needed to estimate v.

| Retained experiment requests | Correctness partition | Calibration partition | Development partition |
|---|---:|---:|---:|
| Haiku three-call protocol | $0.5626 | $0.2810 | $0.2810 |
| Sonnet three-call protocol | $11.2222 | $5.6095 | $5.6270 |
| Opus three-call protocol | $24.6816 | $12.3260 | $12.3428 |
| OpenAI one-call protocol | — | $0.0691 | $0.0692 |

The retained cohorts account for approximately **$73.07** of the **$74.06** combined ledger estimate. The residual **$0.99** covers other historical requests outside these retained cohorts. Reused cache entries can have been charged to earlier ledgers, so a retained cohort’s cost need not equal its run’s ledger increment. Local compute, labeling and human review are excluded from both totals. Calibration fitting and this cost analysis incurred no new API calls.
