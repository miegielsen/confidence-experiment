"""Audit cached call costs and compute descriptive development Pareto sets.

No requests, fitted models, or reserved test data are accessed. Local compute,
annotation and review are explicit unpriced inputs, not assumed free resources.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from confidence_experiment.io import read_json, write_json
from confidence_experiment.llm import CLASSIFY, BLIND, VERIFY
from run_anthropic_batch import specification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/banking77-findings"
FIG = OUT / "figures"
CAPTIONS = {
    "cost_pareto": "Empirical Pareto frontiers on the same 1,000 development examples. Costs are recorded API fees for the calls each pipeline requires, including Claude batch discounts; local compute and fitting costs are excluded. Rings mark non-dominated points and connecting lines are visual guides, not available mixtures. Small Brier differences may be sampling noise. The API-fee-only comparison including local models is intentionally separate from the hosted-only comparisons.",
    "cost_sensitivity": "Illustrative sensitivity of the accuracy frontier to the embedding pipeline’s all-in cost per 1,000 future messages. This local cost includes any amortized setup and local inference cost supplied by the decision maker. The four scenario prices are assumptions, not measured PC costs; hosted prices retain the observed batch fees and exclude their own unpriced overhead. Rings indicate empirical non-dominated choices at each assumed cost.",
}


def frontier(rows, metric, minimize=False, cost_field="api_usd_per_1000"):
    """Keep all exact equivalents; strict improvement in at least one objective."""
    answer = []
    sign = 1 if minimize else -1
    for row in rows:
        x, y = row[cost_field], sign * row[metric]
        dominated = any(other[cost_field] <= x and sign * other[metric] <= y and
                        (other[cost_field] < x or sign * other[metric] < y) for other in rows)
        if not dominated:
            answer.append(row["id"])
    return answer


def main():
    sources = {}

    def tracked(relative):
        p = ROOT / relative
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    labels = read_json(tracked("data/banking77/manifest.json"))["labels"]
    dev = pd.read_csv(tracked("data/banking77/development.csv"))
    category_text = "\n".join(labels)
    components, cohort_costs = {}, {}
    claude_runs = [("haiku", "anthropic-batch-v2"), ("sonnet", "anthropic-sonnet-5-5")]
    include_opus = (OUT / "opus-comparison.json").exists()
    if include_opus:
        claude_runs.append(("opus", "anthropic-opus-5-5"))
    for family, run in claude_runs:
        metadata = read_json(tracked(f"runs/{run}/development/metadata.json"))
        settings = metadata["config"]["llm"]
        cache = ROOT / settings["cache_dir"]
        # Read a few bulk batch files instead of thousands of individual cache files.
        indexed = {}
        for p in sorted((cache / "batches").glob("*.results.json")):
            for item in read_json(tracked(p)):
                result = item["result"]
                if result["type"] == "succeeded":
                    indexed[item["custom_id"]] = result["message"]
        rows = pd.read_csv(tracked(f"runs/{run}/llm_development.csv"), float_precision="round_trip")
        rows = rows.set_index("id").loc[dev.id].reset_index()
        stage_costs, requests = {k: [] for k in ("original", "blind", "verifier")}, []
        for example, saved in zip(dev.itertuples(index=False), rows.itertuples(index=False)):
            specs = {
                "original": specification(settings, settings["model"], CLASSIFY.format(labels=category_text), example.text, labels),
                "blind": specification(settings, settings["verifier_model"], BLIND.format(labels=category_text), example.text, labels),
                "verifier": specification(settings, settings["verifier_model"], VERIFY.format(labels=category_text),
                    json.dumps({"message": example.text, "proposed_intent": saved.label}, ensure_ascii=False), labels, True),
            }
            costs = {}
            for stage, spec in specs.items():
                key = spec["key"]
                if key in indexed:
                    message, discount = indexed[key], .5
                    parsed = json.loads("".join(c["text"] for c in message["content"] if c["type"] == "text"))
                    usage, billed = message["usage"], None
                else:
                    cached = read_json(tracked(cache / f"{key}.json"))
                    parsed, usage, billed = cached["parsed"], cached["usage"], cached["usd"]
                    discount = .5 if cached.get("transport") == "batch" else 1
                assert not (usage.get("cache_creation_input_tokens") or 0)
                standard = (usage["input_tokens"] * settings["input_usd_per_million"] +
                            usage["output_tokens"] * settings["output_usd_per_million"] +
                            (usage.get("cache_read_input_tokens") or 0) * settings["cached_input_usd_per_million"]) / 1e6
                cost = standard * discount
                if billed is not None:
                    assert np.isclose(cost, billed, rtol=0, atol=1e-12)
                if stage == "original":
                    assert parsed["label"] == saved.label and parsed["confidence"] == saved.confidence
                elif stage == "blind":
                    assert parsed["label"] == saved.blind_label
                else:
                    assert parsed["confidence"] == saved.verifier
                stage_costs[stage].append({"actual":cost, "standard_rate_proxy":standard,"discount":discount})
                costs[stage] = cost
                requests.append({"id":example.id, "stage":stage, "request_sha256":key,
                                 "usd":cost, "standard_rate_proxy_usd":standard,
                                 "transport":"batch" if discount==.5 else "synchronous"})
            assert np.isclose(sum(costs.values()), saved.usd, rtol=0, atol=1e-12)
        assert len({r["request_sha256"] for r in requests}) == 3000
        pd.DataFrame(requests).to_csv(OUT / f"{family}-development-costs.csv", index=False)
        components[family] = {stage:{kind:float(sum(r[kind] for r in records))
                                   for kind in ("actual", "standard_rate_proxy")}
                              for stage, records in stage_costs.items()}
        for stage,records in stage_costs.items():
            components[family][stage].update(batch_calls=sum(r['discount']==.5 for r in records),
                                             synchronous_calls=sum(r['discount']==1 for r in records))
        cohort_costs[family] = {}
        for role in ("correctness", "calibration", "development"):
            frame = pd.read_csv(tracked(f"runs/{run}/llm_{role}.csv"), float_precision="round_trip")
            cohort_costs[family][role] = float(frame.usd.sum())

    openai = {}
    for role in ("calibration", "development"):
        f = pd.read_csv(tracked(f"runs/openai-decisions-v2/decisions_{role}.csv"), float_precision="round_trip")
        openai[role] = float(f.usd.sum())
    calibration = read_json(tracked("reports/calibrator-comparison/analysis.json"))
    names = {"embedding_lr":"Embedding LR", "embedding_mlp":"Embedding MLP", "tfidf":"TF-IDF LR",
             "haiku_same":"Haiku native", "haiku_verifier":"Haiku verifier", "haiku_correctness":"Haiku learned",
             "sonnet_same":"Sonnet native", "sonnet_verifier":"Sonnet verifier", "sonnet_correctness":"Sonnet learned",
             "decisions_confidence":"OpenAI confidence", "decisions_probability":"OpenAI probability"}
    names.update({"opus_same":"Opus native", "opus_verifier":"Opus verifier", "opus_correctness":"Opus learned"})
    rows = []
    for key, methods in calibration["results"].items():
        hosted, calls, amount, standard = False, 0, 0., 0.
        if key.startswith(("haiku", "sonnet", "opus")):
            hosted = True
            family, suffix = key.split("_",1)
            stages = ["original"] + (["verifier"] if suffix in ("verifier", "correctness") else [])
            if suffix == "correctness": stages += ["blind"]
            calls = len(stages)
            amount = sum(components[family][stage]["actual"] for stage in stages)
            standard = sum(components[family][stage]["standard_rate_proxy"] for stage in stages)
        elif key.startswith("decisions"):
            hosted, calls, amount, standard = True, 1, openai["development"], openai["development"]
        for method, metrics in methods.items():
            rows.append({"id":key+"."+method, "stream":key, "method":method, "label":names[key],
                         "hosted":hosted, "api_calls_per_message":calls, "api_usd_per_1000":amount,
                         "standard_rate_proxy_per_1000":standard, "accuracy":metrics["accuracy"],
                         "brier":metrics["brier"], "error_auroc":metrics["error_auroc"],
                         "coverage":metrics["acceptance"]["coverage"]})
    hosted = [r for r in rows if r["hosted"]]
    frontiers = {scope:{metric:frontier(subset,metric,minimize=metric=="brier")
                       for metric in ("accuracy","brier","error_auroc","coverage")}
                 for scope,subset in (("all_api_fee_only",rows),("hosted_api_fee_only",hosted))}
    # A joint frontier can retain Sonnet native for higher accuracy even when
    # a cheaper Haiku learned pipeline has a better binary Brier score.
    joint = []
    for row in hosted:
        if not any(other["api_usd_per_1000"]<=row["api_usd_per_1000"] and
                   other["accuracy"]>=row["accuracy"] and other["brier"]<=row["brier"] and
                   (other["api_usd_per_1000"]<row["api_usd_per_1000"] or other["accuracy"]>row["accuracy"] or other["brier"]<row["brier"])
                   for other in hosted): joint.append(row["id"])
    frontiers["hosted_cost_accuracy_brier"] = joint
    selected_ids = ["tfidf.temperature", "embedding_lr.temperature", "embedding_mlp.raw",
                    "decisions_probability.logistic_logit", "haiku_same.logistic_logit", "haiku_verifier.logistic_logit",
                    "haiku_correctness.logistic_score", "sonnet_same.logistic_score", "sonnet_verifier.logistic_logit",
                    "sonnet_correctness.logistic_score"]
    if include_opus:
        selected_ids += ["opus_same.logistic_score", "opus_verifier.logistic_logit", "opus_correctness.logistic_score"]
    by_id = {r["id"]:r for r in rows}
    selected = [by_id[k] for k in selected_ids]
    pd.DataFrame(rows).to_csv(OUT / "cost-frontiers.csv", index=False)
    # Exact duplicate objective coordinates remain equivalent members; plot once.
    fig, axes = plt.subplots(2,2,figsize=(11.5,8.5),layout="constrained")
    for ax,scope,metric,title in zip(axes.flat,
            ["all_api_fee_only","hosted_api_fee_only","hosted_api_fee_only","hosted_api_fee_only"],
            ["accuracy","accuracy","brier","error_auroc"],
            ["API fee versus accuracy including local", "Hosted API fee versus accuracy",
             "Hosted API fee versus Brier error", "Hosted API fee versus error ranking"]):
        subset = rows if scope.startswith("all") else hosted
        seen = set()
        for r in subset:
            coordinate = (r["api_usd_per_1000"],r[metric],r["stream"])
            if coordinate in seen:continue
            seen.add(coordinate)
            color = "#087f78" if not r["hosted"] else "#b34d28" if r["stream"].startswith("haiku") else "#254768" if r["stream"].startswith("sonnet") else "#854397" if r["stream"].startswith("opus") else "#657485"
            ax.scatter(r["api_usd_per_1000"],r[metric],s=26,color=color,alpha=.65)
        points = sorted({(by_id[k]["api_usd_per_1000"],by_id[k][metric]) for k in frontiers[scope][metric]})
        ax.plot([p[0] for p in points],[p[1] for p in points],color="#203644",linewidth=1)
        ax.scatter([p[0] for p in points],[p[1] for p in points],s=100,facecolors="none",edgecolors="#203644",linewidths=1.4)
        annotated=set()
        for key in frontiers[scope][metric]:
            r=by_id[key]
            coordinate=(r["api_usd_per_1000"],r[metric])
            if coordinate in annotated:continue
            annotated.add(coordinate)
            label=r['label'].replace(' probability','').replace(' confidence','').replace(' verifier',' verify')
            offset=(6,8)
            if r['stream'].startswith(('sonnet','opus')):offset=(-80,10)
            if metric=='brier' and r['stream'].startswith('opus'):offset=(-75,30)
            if metric=='brier' and r['stream']=='haiku_same':offset=(12,-15)
            if metric=='brier' and r['stream']=='haiku_verifier':offset=(14,-30)
            if metric=='error_auroc' and r['stream'].startswith('haiku'):offset=(6,-15)
            if not r['hosted']:offset=(7,-18)
            ax.annotate(label,coordinate,xytext=offset,textcoords='offset points',fontsize=8,
                        arrowprops={'arrowstyle':'-','color':'#87909e','linewidth':.6})
        if scope.startswith("all"):
            ax.set_xscale("symlog",linthresh=.05)
            ax.set_xticks([0,.1,1,6,20] if include_opus else [0,.1,1,6],labels=["0","0.10","1","6","20"] if include_opus else ["0","0.10","1","6"])
        else:
            ax.set_xscale("log")
            ax.set_xticks([.07,.1,.3,1,3,6,20] if include_opus else [.07,.1,.3,1,3,6],labels=["0.07","0.10","0.30","1","3","6","20"] if include_opus else ["0.07","0.10","0.30","1","3","6"])
            if include_opus:ax.tick_params(axis='x',labelsize=8)
        if metric=="accuracy":ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:ax.set_ylabel("Brier error (lower is better)" if metric=="brier" else "Error AUROC (higher is better)")
        if metric=="accuracy" and scope.startswith("hosted"):
            ax.set_ylim(.725,max(r['accuracy'] for r in hosted)+.025)
        ax.set(title=title,xlabel="Recorded API USD per 1,000 messages")
        ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=color,label=label)
                        for label,color in [('Local models','#087f78'),('Haiku','#b34d28'),
                                            ('Sonnet','#254768'),('OpenAI','#657485')]+([('Opus','#854397')] if include_opus else [])],
               loc='lower center',bbox_to_anchor=(.5,-.035),ncol=5,frameon=False,fontsize=9)
    for ext in ("svg","png","pdf"):fig.savefig(FIG/f"cost_pareto.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)

    native = [by_id[k] for k in ("decisions_probability.logistic_logit","haiku_same.logistic_logit","sonnet_same.logistic_score")]
    if include_opus:
        native.append(by_id['opus_same.logistic_score'])
    scenarios = []
    fig,axes=plt.subplots(1,4,figsize=(12.5,3.7),layout="constrained",sharey=True)
    for ax,local in zip(axes,(0.,.10,1.,3.)):
        subset=[dict(r) for r in native]+[dict(by_id["embedding_lr.temperature"],api_usd_per_1000=local)]
        ids=frontier(subset,"accuracy")
        scenarios.append({"assumed_embedding_all_in_usd_per_1000":local,"accuracy_frontier":ids})
        for r in subset:
            color="#087f78" if not r["hosted"] else "#657485"
            ax.scatter(r["api_usd_per_1000"],r["accuracy"],color=color,s=40)
            if r["id"] in ids:ax.scatter(r["api_usd_per_1000"],r["accuracy"],s=120,facecolors="none",edgecolors="#203644")
            label='Embedding' if not r['hosted'] else r['label'].split()[0]
            offset=(-45,8) if r['stream'].startswith(('sonnet','opus')) else ((-57,-15) if local>=1 else (5,-15)) if not r['hosted'] else (5,5)
            ax.annotate(label,(r['api_usd_per_1000'],r['accuracy']),xytext=offset,textcoords='offset points',fontsize=8)
        ax.set_xscale("symlog",linthresh=.05)
        ax.set(title=f"Local assumption ${local:.2f}/1k",xlabel="USD per 1,000 messages",ylim=(.70,1),xlim=(-.01,max(3.7,max(r['api_usd_per_1000'] for r in native)*1.3)))
        ax.set_xticks([0,.1,1,3],labels=["0","0.10","1","3"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Development accuracy")
    for ext in ("svg","png","pdf"):fig.savefig(FIG/f"cost_sensitivity.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)

    spending = read_json(tracked("reports/banking77-findings/" + ("opus-comparison.json" if include_opus else "sonnet-comparison.json")))["spending"]["combined_usd"]
    retained = sum(sum(v.values()) for v in cohort_costs.values())+sum(openai.values())
    assert spending>=retained
    c=components
    cost_table="\n".join(f"| {r['label']} | {r['api_calls_per_message']} | ${r['api_usd_per_1000']:.4f} | ${r['standard_rate_proxy_per_1000']:.4f} | {r['accuracy']:.1%} | {r['brier']:.5f} |" for r in selected)
    study_table="\n".join(f"| {family.title()} three-call protocol | ${v['correctness']:.4f} | ${v['calibration']:.4f} | ${v['development']:.4f} |" for family,v in cohort_costs.items())
    study_table+=f"\n| OpenAI one-call protocol | — | ${openai['calibration']:.4f} | ${openai['development']:.4f} |"
    sonnet_native=c['sonnet']['original']['actual']
    haiku_native=c['haiku']['original']['actual']
    native_ratio=sonnet_native/haiku_native
    markdown=f"""## Cost and Pareto frontiers

**Embeddings remain the strongest candidate on accuracy and API fees; the LLM trade-off depends on which confidence objective matters.** This is an API-fee comparison, not a claim that local computation or labeled data are free. Costs below are reconstructed from the recorded token usage for the same 1,000 development messages, counting only the calls each inference pipeline needs. Calibration transforms add no API calls. Learned correctness requires the original, blind, and verifier calls plus local TF-IDF support; verification alone requires the original and verifier calls.

| Inference pipeline | API calls per message | Recorded API USD per 1,000 | Standard-rate proxy | Accuracy | Brier error |
|---|---:|---:|---:|---:|---:|
{cost_table}

The table uses temperature for TF-IDF and embedding LR, raw embedding MLP, logistic on log-odds for OpenAI, Haiku native and both verifiers, and logistic on score for Sonnet native and both learned pipelines. All 48 variants enter the frontier calculation; the table lists representative fixed choices rather than asserting that each is the best calibrator. Local inference costs remain unpriced. Sonnet reuses three synchronous smoke-test messages per call stage, so its recorded cohort cost is slightly above an all-batch estimate. The standard-rate proxy applies standard prices to identical recorded tokens; it does not measure synchronous response quality or latency.

Claude batch processing discounted input and output charges by 50%. Haiku’s standard input/output rates were $0.10/$0.50 per million tokens and Sonnet’s were $2/$10. Decisions used $0.10 per million chargeable input tokens, with no output or cache charges. These rates were checked on 9 October 2026; workload length, caching and future rates can change costs. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI Decisions pricing](https://developers.openai.com/api/docs/guides/decisions)

A method is **Pareto dominated** for a chosen pair of objectives when another tested method costs no more, performs at least as well, and improves at least one objective. Equivalent points are retained together. The frontiers use development point estimates and the same cost denominator; tiny metric differences can move membership and do not establish statistical superiority. Brier comparisons across classifiers also reflect different error prevalence and discrimination, rather than pure calibration quality.

For **hosted classification accuracy**, the frontier progresses from OpenAI Decisions (${openai['development']:.4f}/1,000, 73.3%) to Haiku native (${haiku_native:.4f}, 75.7%) to Sonnet native (${sonnet_native:.4f}, 83.0%). Sonnet native cost approximately **{native_ratio:.1f} times** Haiku native for a 7.3-point accuracy gain. The verifier and learned pipelines retain each base classifier’s accuracy, so their extra calls are dominated for accuracy alone.

For **hosted cost versus Brier error**, learned Haiku is cheaper and has lower Brier error than Sonnet native or verifier confidence. Sonnet learned confidence remains a frontier option because its Brier point estimate, 0.05052, is lower than Haiku learned’s 0.05780, at substantially greater API cost. For **cost versus error ranking**, Haiku learned’s AUROC of 0.961 exceeds Sonnet learned’s 0.952 at lower cost. Sonnet native remains relevant in a joint cost–accuracy–Brier analysis because it has higher classification accuracy than Haiku; dominance in one projection does not settle the overall choice.

When local models enter the **API-fee-only** comparison, embedding LR with temperature scaling dominates hosted pipelines on cost plus accuracy, Brier, error ranking, and exploratory accepted coverage. All embedding LR calibration variants tie on the accuracy projection. This result concerns provider fees only; the sensitivity analysis below makes the missing local costs explicit. All hosted methods have zero accepted coverage under the current Wilson rule, so none has yet demonstrated automated-review savings at that stringent target.

## Local cost sensitivity and study spending

For a deployment of N messages, let F be fixed setup and annotation cost in USD, v local variable cost per message, A API fees per 1,000, r the fraction sent to human review, and R the cost of reviewing one message. Then **total USD per 1,000 = A + 1,000v + 1,000F/N + 1,000rR**. No measured values for F, v or R are available here. The calibrated supervised classifiers used 5,992 fitting labels and 1,000 calibration labels. Learned LLM correctness used the same TF-IDF fitting labels plus 2,000 correctness and 1,000 calibration labels; calibrating native LLM scores required 1,000 labeled outcomes. Those requirements matter when applying the experiment to a new task.

Holding the observed hosted fees fixed, embedding LR can retain pairwise cost-and-accuracy dominance over OpenAI, Haiku native, or Sonnet native only while its all-in local cost is no higher than approximately **${openai['development']:.4f}, ${haiku_native:.4f}, or ${sonnet_native:.4f} per 1,000 messages**, respectively. The scenario figure assumes local costs of $0, $0.10, $1 or $3 per 1,000 and recomputes the accuracy frontier. These are sensitivity assumptions, not estimates for this computer. Hosted setup and review costs would also have to enter a complete deployment comparison.

Under the development-selected temperature threshold, embedding LR would send 105 of 1,000 messages to review, versus all 1,000 for the hosted pipelines under the same strict rule. This illustrates why review cost may outweigh token charges, but the 105-case workload is an exploratory development result requiring independent validation. Cached embedding training durations cannot support a reliable new-message CPU or electricity estimate; a fresh inference benchmark would be needed to price v.

| Retained experiment requests | Correctness partition | Calibration partition | Development partition |
|---|---:|---:|---:|
{study_table}

The retained cohorts account for approximately **${retained:.2f}** of the **${spending:.2f}** combined ledger estimate. The residual **${spending-retained:.2f}** covers other historical requests outside these retained cohorts. Reused cache entries can have been charged to earlier ledgers, so a retained cohort’s cost need not equal its run’s ledger increment. Local compute, labeling and human review are excluded from both totals. Calibration fitting and this cost analysis incurred no new API calls.
"""
    if include_opus:
        from report_opus import update_cost_prose
        markdown = update_cost_prose(markdown, rows, frontiers, components)
    (OUT/"cost-section.md").write_text(markdown,encoding="utf-8")
    for path,expected in sources.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"evaluation_role":"development","n":1000,
            "cost_scope":"Recorded inference API fees; local compute, setup, labels and review excluded",
            "frontier_scope":"Descriptive point estimates, not significance-adjusted; equivalent points retained",
            "stage_costs":components,"retained_cohort_costs":cohort_costs,"openai_costs":openai,
            "recorded_total_usd":spending,"retained_total_usd":retained,"other_historical_usd":spending-retained,
            "pipelines":rows,"frontiers":frontiers,"sensitivity":scenarios,"captions":CAPTIONS,
            "source_sha256":sources,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "official_test_evaluated":False,"new_api_calls":0}
    write_json(OUT/"cost-frontiers.json",result)
    print(json.dumps({"stage_costs":components,"frontiers":frontiers,"retained_total":retained},indent=2))


if __name__=="__main__":main()
