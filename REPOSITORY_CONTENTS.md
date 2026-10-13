# Repository contents

This repository preserves the experiment's source, configuration, data, results,
and reports. Start with [the findings report](reports/banking77-findings/report.md)
or download its [PDF](reports/banking77-findings/report.pdf). The HTML reports are
self-contained and can be opened locally after cloning.

Included:

- Experiment specification, Python source, scripts, tests, configurations, and
  pinned dependency lists.
- BANKING77 source/prepared data, split assignments, duplicate audit, labels, and
  provenance manifests under `data/banking77/`.
- Saved Anthropic/OpenAI request payloads, responses, parsed caches, batch records,
  and spending ledgers under `data/`. These preserve completed paid work.
- Run metadata, status/history, predictions, metrics, reliability bins,
  risk/coverage curves, and disagreement/category analyses under `runs/`.
- Saved embedding arrays and bootstrap samples. These are derived research data,
  not encoder weights.
- Markdown/HTML/PDF reports, scientific figures, calibration comparisons, cost
  and Pareto analyses, and report publication/Page snapshots under `reports/`.

Excluded by `.gitignore`:

- Model and calibrator objects (`*.joblib`, pickles), encoder weights, and training
  checkpoints. Their existing hashes and metadata remain in the run manifests.
- API key files, encrypted key stores, environment files, and other credentials.
- Python environments, bytecode, temporary caches, and local editor state.

API keys are stored outside this project using Windows DPAPI. The key setup and
loader scripts are included because they contain instructions, not key material.
On a new machine, provide your own credentials; do not commit them.

Results can be inspected without model objects. Some analysis and final-evaluation
commands require the omitted fitted objects and will need the original local
artifacts or a matching reconstruction. Rebuilding fitted artifacts can depend on
the pinned package versions and model revisions; compare hashes before treating a
reconstruction as identical. Do not rerun paid API requests merely to view results.

The official BANKING77 test partition is retained as data. Independent test
evaluation now covers local models and OpenAI Decisions; Claude test arms are
deferred. The test outcomes never enter fitting, calibration, or threshold
selection. Archived interrupted and superseded
runs are retained with their status so their outputs are not mistaken for the
active comparison.

The completed consolidated comparison includes EmbeddingGemma, TF-IDF, Haiku 5.5,
Sonnet 5.5, Opus 5.5, and OpenAI Decisions.
