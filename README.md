# Confidence experiment

Read the [consolidated findings](reports/banking77-findings/report.md) or download
the [PDF report](reports/banking77-findings/report.pdf). The completed comparison
includes EmbeddingGemma, TF-IDF, Haiku 5.5, Sonnet 5.5, Opus 5.5, and OpenAI
Decisions, with temperature, logistic, and isotonic calibration and cost/Pareto
analyses. Independent test validation is described below.

## Independent test validation

The official-test phase now uses `scripts/final_evaluation.py`. Its sealed plan
includes all 60 development variants and their already selected thresholds.
Execution currently includes the local classifiers and OpenAI Decisions only
(24 variants on all 3,080 official test messages). A subsequently authorized
smaller sweep evaluates Haiku alone (12 additional frozen variants); Sonnet and
Opus remain deferred. The original report scope is recorded in
`runs/final-evaluation/execution-scope.json`; the supplemental sweep is recorded
in `runs/final-evaluation/haiku-sweep.json`.

```powershell
. ./scripts/load-openai-key.ps1
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" scripts/final_evaluation.py run
```

This entry point validates frozen evidence before any API request, never fits a
calibrator or selects a threshold on test outcomes, and skips verified completed
arms when resuming. Reports can be regenerated separately with
`python scripts/build_final_report.py`, without inference or paid API calls.
Results are under `runs/final-evaluation/<arm>/`; `complete.json` is written only
after every result artifact has been saved and hashed. The original development
source and fitted objects stay unchanged to preserve their historical identities.
Use this entry point for the complete calibration comparison rather than the
legacy `--final` commands documented below.

The primary local test policy failed the 1% error target: it accepted
2,715 messages with 53 errors (1.95%; 95% Wilson interval 1.50%–2.54%). The report
includes the complete requested test comparison. OpenAI test accuracy was 76.46%;
the test cohort's estimated API fees were $0.2125. The development study is
preserved separately as `reports/banking77-findings/report.development.*`.

The Haiku test sweep uses the same sealed plan and existing fitted objects:

```powershell
. ./scripts/load-anthropic-key.ps1
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" scripts/final_evaluation.py arm --arm haiku
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" scripts/report_haiku_test.py
```

While accepted batches are processing, a separate `scripts/report_haiku_test.py
--wait` process generates the results addendum automatically when the arm is
verified complete. It never submits requests or retries failed inference. The
addendum is written to `reports/haiku-test/`, including development/test metrics,
paired Brier comparisons, costs, and a combined 36-variant test Pareto analysis.
Batch progress is under `runs/final-evaluation/haiku/batch_status.json`; reporting
status is under `runs/final-evaluation/haiku-report/status.json`. The existing main
report and ChatGPT Page remain separately versioned and are not rewritten by
this supplemental runner.

The matched Sonnet/Opus sample uses `scripts/run_claude_sample.py`. Its separate
sealed plan selects eight random test messages per intent (616 per model) with
seed 20261014 and a shared $12 limit for new API calls. It inherits every fitted
object, prompt and development threshold from the original final plan. Work is
submitted in matched blocks whose worst-case reservations fit the remaining
budget; the sample can stop early if usage exhausts that limit. The two models
share a serialized spending ledger, while their batch jobs are isolated from
the previously prepared full-test jobs. No Sonnet/Opus full-test completion
markers are written.

```powershell
. ./scripts/load-anthropic-key.ps1
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" scripts/run_claude_sample.py freeze
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" scripts/run_claude_sample.py run
```

Status, accepted batch IDs, the selected sample and spending are saved under
`runs/claude-sample-20261014/`. After inference, the same worker generates
`reports/claude-sample-test/report.md` and `.html`, per-example outputs, paired
intervals, and cost/Pareto results for all 60 variants on the same selected IDs.
Local, OpenAI and Haiku results are reused. To repeat reporting without paid
requests, use `python scripts/run_claude_sample.py report`.

This GitHub archive includes data, API response caches, results, and reports, but
excludes model objects and credentials. See [repository contents and reproduction
limits](REPOSITORY_CONTENTS.md). Before pushing additional files, stage them and
run `python scripts/audit_git_export.py` to check the staged export.

Implementation of [experiment.md](experiment.md), starting with BANKING77. Defaults are CPU execution and Anthropic for LLM calls. The original API budget was $25; at the user's request, **budget enforcement is now disabled for the active API configurations**. OpenAI and Anthropic run concurrently with independent ledgers; spending is still recorded.

Current batch configuration: **`configs/anthropic-batch-v2.toml`**, with explicit `thinking = "disabled"`. The original batch run used Haiku's default adaptive thinking with a 256-token total output limit; 59 responses in its first 2,000-request batch were truncated. That exploratory run is archived, its usable results and charges retained, and the comparison is restarted uniformly with thinking disabled. The revised thinking setting is part of every request/cache key, so old and revised predictions cannot be mixed. Anthropic documents [thinking defaults and disabling thinking for Haiku 5.5](https://platform.claude.com/docs/en/build-with-claude/thinking).

Run the current experiment with `./scripts/start-background.ps1 -Config configs/anthropic-batch-v2.toml`. Its progress/report paths begin with `runs/anthropic-batch-v2/`. Background jobs now default to two CPU threads and below-normal Windows priority to keep the computer responsive; `-CpuThreads` can override the thread count. The status file includes a heartbeat every 15 seconds. The embedding run resumes its completed embedding files after an interrupted worker.

The initial runnable control is TF-IDF plus multinomial logistic regression. It validates the experiment machinery; it is an additional control, not a substitute for EmbeddingGemma or ModernBERT.

The illustrated findings report is `reports/banking77-findings/report.html`, with a printable `report.pdf`. Its probability comparisons include reliability plots with shared equal-width bins, Wilson intervals, and bin counts; separately normalized score distributions for correct and incorrect predictions; and a descriptive table for scores at least 90%. The dedicated view is `reports/banking77-findings/probability-comparisons.html`. Both figures preserve tied confidence scores and show the underlying sample sizes. `scripts/add_probability_comparisons.py` verifies the bins against saved experiment outputs and exports PNG/SVG/PDF figures. The Sonnet report finisher adds the same diagnostics with Sonnet included after the run completes.

Initial development results (1,000 examples): accuracy **87.1%**, macro-F1 **0.873**. Temperature scaling reduced binary Brier score from **0.1452 to 0.0754** and log loss from **0.4384 to 0.2486**. Neither score supplied an acceptance threshold meeting the default Wilson upper-bound rule. These are development findings, not official test results. Open `runs/pilot/development/report.html` for the full report.

## Sonnet 5.5 replication

The matched Sonnet experiment uses `configs/anthropic-sonnet-5-5.toml` and the same BANKING77 partitions, prompts, three-call protocol, TF-IDF features, correctness estimator, calibrators, and acceptance rule as the completed Haiku comparison. Classification, blind classification, and verification all use `claude-sonnet-5-5`. The official test set remains reserved.

Sonnet 5.5 rejects `thinking = "disabled"`. Its supported `thinking = "between_tools"` setting turns off up-front thinking and returns only text for requests without tools. This model-specific setting is recorded in request/cache keys. The synchronous smoke check passed the exact three-call protocol for three development examples; those nine cached calls are reused by the batch run. [Anthropic thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking)

Start or safely resume with `./scripts/start-background.ps1 -Config configs/anthropic-sonnet-5-5.toml`. Progress is under `runs/anthropic-sonnet-5-5/background_status.json`, `batch_status.json`, and `background.log`; the development report is generated at `runs/anthropic-sonnet-5-5/development/report.html` after both batch waves finish. The worker uses two CPU threads and below-normal priority. It does not load or train BERT.

Sonnet uses a separate cache and ledger in `data/anthropic_sonnet_cache/`. Budget enforcement is disabled as authorized; accounting and request-count limits remain active. Configured standard prices are $2/M input and $10/M output, with the batch adapter applying the 50% discount ($1/M input and $5/M output), verified on 2026-10-09. Add this ledger to the two historical ledgers for total experiment spending. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)

The separate `scripts/finish_sonnet_replication.py` process waits for successful worker completion, verifies matching development IDs and TF-IDF support, runs a 5,000-resample paired bootstrap, and writes `reports/sonnet-5-5-findings/report.html` with three scientific figures. Its state and logs are in `runs/anthropic-sonnet-5-5/comparison_status.json`, `comparison.log`, and `comparison-errors.log`. It also adds a link to the original local findings report. It never submits API requests or retries a failed worker. Run it with the project Python to resume report generation if needed; a lock prevents duplicate finishers. The ChatGPT Page is a separately saved report and is not automatically changed by this local process.

## OpenAI Decisions comparison

`configs/openai-decisions.toml` adds the dedicated `POST /v1/decisions` endpoint with `gpt-6-luna`. The [official Decisions guide](https://developers.openai.com/api/docs/guides/decisions) documents a public beta, 2–255 options per choice question, and $0.10 per million input tokens without output/cache charges. The 77 BANKING77 intents fit into one choice question. Requests use standard global processing and short text, without regional or long-context premiums. The worker uses the documented HTTP API directly, so no OpenAI SDK upgrade is required.

```powershell
./scripts/set-openai-key.ps1
./scripts/decisions.ps1 plan
./scripts/decisions.ps1 smoke
./scripts/start-background.ps1 -Config configs/openai-decisions.toml
```

Key entry is masked. Windows DPAPI encrypts it for this Windows account in `%LOCALAPPDATA%/confidence-experiment/secrets/openai.clixml`, outside the project. Both the wrapper and background starter load it automatically. Do not paste keys into chat or config files.

This arm uses the existing 1,000 calibration and 1,000 development examples: 2,000 uncached requests. It needs neither supervised fitting nor correctness-model training. It evaluates five signals while preserving the API-selected label: native `confidence`, selected-option probability, sigmoid calibration of each, and temperature scaling of the full probability distribution. The raw native confidence is treated as a candidate correctness score; calibration is tested rather than assumed. All runs use the same development examples, metrics and threshold rule. Refusals remain in the full evaluation denominator as unsuccessful classifications and always require review. Raw API scores and distributions remain null; their operational correctness score is zero. Temperature fitting uses answered calibration examples only. Invalid distributions stop the run and remain billed; there are no silent exclusions.

The OpenAI worker retains `data/anthropic_cache/spending.json`, which includes original exploratory Anthropic charges. New Anthropic batch charges are recorded separately in `data/anthropic_spending/spending.json`; add the two ledgers for total historical spending. Each ledger has its own lock, so API runs can proceed concurrently. `enforce_budget = false` disables spending caps in both active configurations while retaining accounting and request-count limits. OpenAI raw responses and parsed caches live in `data/openai_decisions_cache/`. Raw responses can be imported after interruption without another request or duplicate billing. Ambiguous network failures retain reservations and require inspection before resubmission. The initial plan reserves conservatively below $1.46 for 2,000 development requests; actual billing is based on returned usage.

Reports, progress and frozen calibration are saved under `runs/openai-decisions-v2/`. The first run stopped on a refusal; its status is archived under `runs/openai-decisions/`. The revised abstention policy uses a new run identity while reusing all completed raw responses and charges. Use `./scripts/decisions.ps1 run --final` only after candidate selection; it reuses frozen calibration and thresholds and rejects repeated final evaluation. The official test set is not evaluated by development or smoke commands.

Start both independent API runs:

```powershell
./scripts/start-background.ps1 -Config configs/openai-decisions.toml
./scripts/start-background.ps1 -Config configs/anthropic-batch-v2.toml
```

Background supervisors refuse duplicate launches for a run already supervised or completed. `-ThenConfig` remains available for deliberately sequential runs, but is no longer needed for budget coordination.

Embedding development results are now available in `runs/embeddings/development/report.html`: logistic regression accuracy **97.1%**, MLP **96.9%**. Temperature-calibrated logistic regression selected **89.5%** development coverage with **3/895** accepted errors. These acceptance results reuse the threshold-selection data and are optimistic; official test performance has not been measured. A report-only fix clamps tiny negative plotted Wilson interval lengths after CSV rounding; archived artifacts and identical-metric recovery records are retained in the pilot and embedding run directories.

## Full ModernBERT fine-tuning on CPU

`configs/modernbert.toml` pins ModernBERT-base revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, trains all weights for three epochs on the same 5,992 fit examples, and preserves the separate calibration/development/test roles. It can run independently of either API provider:

```powershell
./scripts/start-background.ps1 -Config configs/modernbert.toml -DownloadEncoder
```

The supervisor first resumes and verifies the approximately 599 MB model download. It then waits for at least 4 GiB of available physical RAM before loading the training worker, using two CPU threads and below-normal priority. The FP32 weights, gradients and two Adam states alone require about 2.2 GiB, before activation and framework overhead. While waiting, `runs/modernbert/background_status.json` reports `waiting_memory`, current free memory, and the required amount. The gate is based on machine resources, not API-run completion.

`scripts/run_modernbert_cpu.py` instruments the existing AdamW trainer without changing its updates. Once training starts, `training_progress.json` reports actual step times and estimates the remaining portion of 4,494 optimizer steps. Evaluation and report generation take additional time. The first steps are part of the real three-epoch run; no extra benchmark updates are added. Runtime cannot be estimated from measured full-model steps until enough RAM is available to start them.

## Run on this Windows computer

```powershell
./scripts/setup.ps1
./scripts/experiment.ps1 prepare
./scripts/experiment.ps1 plan
./scripts/experiment.ps1 run
Start-Process './runs/pilot/development/report.html'
```

The environment lives in `%LOCALAPPDATA%/confidence-experiment/venv`, outside Google Drive. The source, datasets and outputs stay in this project. To run tests:

```powershell
& "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" -m pytest
```

On other machines, create a Python 3.11+ environment, install `pip install -e '.[test,llm]'`, and use `python -m confidence_experiment` in place of the wrapper. `requirements-cpu.lock` records the exact initially validated CPU/API dependency versions; install it followed by `pip install --no-deps -e .` to reproduce those versions.

## Anthropic experiment

Run `./scripts/set-anthropic-key.ps1` to enter the key in a masked local prompt. It is saved with Windows DPAPI encryption for your account under `%LOCALAPPDATA%/confidence-experiment/secrets/anthropic.clixml`, outside this repository. The PowerShell runner decrypts it only into the child process environment. Alternatively, set `ANTHROPIC_API_KEY` using your preferred secret manager. Do not put credentials in TOML files or commit them. The runner uses the first-party Anthropic API.

```powershell
./scripts/smoke-anthropic.ps1
./scripts/experiment.ps1 plan --config configs/anthropic.toml
./scripts/experiment.ps1 run --config configs/anthropic.toml
```

The optional smoke check validates 10 development messages with all three calls before the full run. It shares the cache and $25 ledger with the main experiment and saves `runs/anthropic/smoke.json`; it is a connectivity check, not a benchmark estimate. Avoid running it concurrently with the main API experiment.

For a long run independent of the terminal, use `./scripts/start-background.ps1 -Config configs/anthropic.toml`. Inspect `runs/anthropic/background_status.json` and `runs/anthropic/background.log`; the report is written automatically on completion. Background runs still hold the same run/cache locks and enforce the same budget. To stop one, use `taskkill /PID <pid-from-background_status.json> /T /F`; completed calls are preserved, but abrupt termination may leave a lock or unresolved API reservation that must be inspected before resuming.

The live Haiku 5.5 smoke check succeeded on 2026-10-09: 30 calls in 57.5 seconds, costing $0.00583. At that observed speed the 12,000-call development experiment takes roughly 6.4 hours; actual speed, rate limits and input length can differ.

### Recommended: Message Batches API

The large experiment uses `configs/anthropic-batch.toml`. Start it with:

```powershell
./scripts/start-background.ps1 -Config configs/anthropic-batch.toml
# Or keep it attached to this terminal:
./scripts/experiment.ps1 run --config configs/anthropic-batch.toml
```

Anthropic [charges 50% of standard token prices for batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing); Haiku 5.5 short-prompt batch rates are $0.05/M input and $0.25/M output as checked on 2026-10-09. Most batches finish within an hour, but individual batches can take up to 24 hours. This experiment needs **two dependent waves**: initial labels and blind classifications together, followed by verification of the selected labels. These are split into batches of at most 2,000 requests. Results are joined using `custom_id`, never output order. Individual request latency is unavailable and recorded as null; batch wall times and request counts are retained instead.

Completed synchronous predictions are reused with identical cache keys. Pending batch IDs, payloads and raw results live under `data/anthropic_cache/batches`. The shared $25 budget reserves the maximum configured output allowance for each submitted batch and reconciles actual discounted usage after completion. Unsuccessful requests are not billed. Completed batches are accounted only once on restart, including after a partially completed local cache import. Ambiguous submissions with no locally recorded batch ID halt for inspection rather than risking duplicate charges. A failed polling/download operation can be resumed with the same config; remote batches continue while the local process is stopped.

Progress is in `runs/anthropic-batch/batch_status.json`, the supervisor state in `background_status.json`, and logs in `background.log`. The final development report is `runs/anthropic-batch/development/report.html`. Once development/model selection is complete, use `./scripts/experiment.ps1 run --config configs/anthropic-batch.toml --final` for the separate official-test batches.

The initial sequential run was stopped after 144 successful calls ($0.02807 total, including the smoke check) to migrate to batches. Those cached results remain reusable; no API reservation was outstanding when it stopped.

This configuration uses `claude-haiku-5-5` for classification and verification. The model ID and short-prompt rates ($0.10/M input, $0.50/M output) were checked against [Anthropic's model overview](https://platform.claude.com/docs/en/models/overview) and [pricing](https://platform.claude.com/docs/en/about-claude/pricing) on 2026-10-09. Change model IDs and matching prices together before starting a new run. Access depends on your Anthropic account. The initial classifier is zero-shot; category names supply the category definitions. Verification uses a fresh conversation and sees neither the original confidence nor any rationale. The blind classification sees no proposed label.

Development requires **12,000 calls** without cache: 3 calls for each of 2,000 correctness-training, 1,000 calibration and 1,000 development examples. Final evaluation adds **9,240 calls**. Each response records billed token usage, model ID, latency and estimated USD cost. API calls run sequentially with automatic retries disabled.

Before every request, the runner persists a conservative cost reservation using UTF-8 input size plus schema overhead and the maximum output-token allowance. It reconciles this against returned usage. Calls stop if a reservation would exceed $25. This is a client-side pre-tax spending control using configured prices, not an Anthropic account limit. Do not run other tools against this budget ledger, delete it, or share its cache across concurrent processes. The CLI prevents concurrent API runners with a lock file.

If a request fails without returned usage, its reservation stays in the ledger and the request cannot silently run again. Inspect the provider's usage records before resolving that reservation. Interrupted runs preserve completed responses and the supervised checkpoint; rerun the same command to resume. Changing `max_new_calls`, `enabled`, or `budget_usd` does not invalidate the experiment; increasing the authorized $25 limit requires a separate decision.

## Neural comparisons on CPU

Install the optional dependencies into the same environment (`requirements-neural.lock` records the versions installed during setup):

```powershell
uv pip install --python "$env:LOCALAPPDATA/confidence-experiment/venv/Scripts/python.exe" -e '.[test,llm,neural]'
```

`configs/full.toml` implements EmbeddingGemma with logistic regression and a one-hidden-layer MLP, ModernBERT fine-tuning, and the LLM comparisons. EmbeddingGemma requires accepting its [Hugging Face license conditions](https://huggingface.co/google/embeddinggemma-300m) and authenticating locally. Its classification prefix is `task: classification | query: ` as specified in the model card. Model revisions are resolved to commit hashes, recorded and reused during final evaluation. Fine-tuning uses only the fit partition, fixed hyperparameters, AdamW and gradient clipping. It can take substantial time on CPU.

Individual local configurations are supplied as `configs/embeddings.toml` and `configs/modernbert.toml`. Run either with `./scripts/experiment.ps1 run --config <path>`. To run all comparisons, configure the `[llm]` section of `configs/full.toml` as in `configs/anthropic.toml` and keep the same shared cache path and $25 budget. ModernBERT fine-tuning needs substantially more memory than inference; the computer had approximately 2.8 GB available during initial setup, so a full encoder run may require freeing memory or using another machine.

## Protocol and outputs

The official dataset is pinned to PolyAI commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. Preparation saves source hashes, a label list, stratified role assignments, file hashes and a duplicate audit in `data/banking77/manifest.json`. Training texts are case-folded and whitespace-normalized for duplicate detection only; models use the original text. Conflicting training duplicates and train/test text overlaps are excluded from training; the remaining training messages are deduplicated. Consequently the fit count can be slightly below 6,003. The official 3,080 test rows and labels are preserved.

| Role | Purpose |
| --- | --- |
| fit | Supervised classifiers and labeled neighbor references |
| correctness (2,000) | Learn whether the original LLM prediction is correct |
| calibration (1,000) | Sigmoid confidence calibration and class-logit temperature scaling |
| development (1,000) | Compare methods and select acceptance thresholds |
| official test (3,080) | One final evaluation after models and thresholds are frozen |

Methods 1–3 always reuse the identical cached initial LLM label. The correctness model uses original and verifier confidence, blind-label agreement, message length, missing-content status, and an alternative classifier's support/margin when available. Embedding runs also add nearest-neighbor distance and agreement with the LLM-selected label using fit references only. No true label enters an inference-time feature. A regularized logistic model is used without error oversampling or class weighting. Raw and sigmoid-calibrated correctness scores are both reported.

`run` evaluates development only and saves `frozen.joblib`. Once you have completed model/prompt selection, final evaluation loads these fitted models and calibration objects:

```powershell
./scripts/experiment.ps1 run --config configs/anthropic.toml --final
```

Choose the candidate configurations before running their final evaluations. Existing final results cannot be overwritten by the CLI. A run directory rejects changed scientific settings, Python source or data; choose a new directory for another experiment. Saved model/calibration artifacts are hashed and checked before final evaluation.

Each report directory contains:

- `report.html`, a standalone report with an embedded reliability/risk-coverage figure;
- `metrics.json` and `summary.csv`, including accuracy, macro-F1, binary Brier/log loss, ECE, error AUROC/AP and AURC;
- per-method predictions, disagreements for label audit, per-category precision/recall/F1;
- reliability bin counts with Wilson intervals, and risk–coverage CSVs;
- acceptance coverage and error rate at the threshold chosen on development;
- run metadata, package versions, timing and API ledger where applicable.

Thresholds default to the largest development coverage whose 95% Wilson upper error bound is at most 1%. This is conservative for a single fixed threshold, but scanning thresholds is adaptive: it does not establish a 1% guarantee. Development acceptance metrics reuse the selection data and are optimistic. Final test error intervals are descriptive binomial intervals, not simultaneous bounds across methods. Confidence ties are accepted as entire groups. Zero acceptance produces undefined risk, rather than a misleading zero error rate.

Brier/log loss measure both calibration and discrimination. ECE depends on binning; the reliability table and error-ranking metrics provide complementary evidence. Model changes require new correctness/calibration data. Public benchmark performance may reflect pretraining exposure and does not establish calibration on enterprise documents.

LEDGAR replication, CLINC150 out-of-scope evaluation, prompt-order stability features, cross-fitting, and a formal label-audit annotation workflow are follow-up extensions. They are not part of the initial BANKING77 implementation.

Implementation references: [PolyAI dataset](https://github.com/PolyAI-LDN/task-specific-datasets), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [ModernBERT API](https://huggingface.co/docs/transformers/model_doc/modernbert), and [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
