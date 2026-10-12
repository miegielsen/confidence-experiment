"""Integrate Opus evidence into the existing report without changing predictions."""
import re


def export_opus_probabilities(directory,frames,bins):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    keys=[('Opus original','anthropic-opus-5-5:llm_same_raw'),('Opus learned','anthropic-opus-5-5:llm_correctness_raw')]
    fig=plt.figure(figsize=(11.6,4),layout='constrained');outer=fig.add_gridspec(1,2)
    for column,(title,key) in enumerate(keys):
        grid=outer[0,column].subgridspec(2,1,height_ratios=(3,1),hspace=.03)
        ax,count_ax=fig.add_subplot(grid[0]),fig.add_subplot(grid[1]);records=bins[key]
        x=np.array([r['confidence'] for r in records]);y=np.array([r['accuracy'] for r in records]);ci=np.array([r['ci95'] for r in records])
        ax.plot([0,1],[0,1],'--',color='#657485',linewidth=1)
        ax.errorbar(x,y,yerr=np.maximum([y-ci[:,0],ci[:,1]-y],0),fmt='o',markersize=4,capsize=3,color='#087f78',linewidth=1.3)
        ax.set(title=title,xlim=(0,1.025),ylim=(0,1.04),ylabel='Observed fraction correct');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.tick_params(axis='x',labelbottom=False);ax.grid(alpha=.13)
        for r in records:
            center=(r['bin']+.5)/10;count_ax.bar(center,r['n'],width=.08,color='#087f78',alpha=.65)
            count_ax.text(center,r['n']*1.12,str(r['n']),ha='center',fontsize=8)
        count_ax.set(yscale='log',ylim=(.8,2000),xlim=(0,1.025),ylabel='Bin n',xlabel='Reported correctness probability')
        count_ax.set_yticks([1,100,1000],labels=['1','100','1k']);count_ax.grid(axis='y',alpha=.13)
    for ext in ('svg','png','pdf'):fig.savefig(directory/f'opus_probability_reliability.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.5),layout='constrained')
    for ax,(title,key) in zip(axes,keys):
        frame=frames[key]
        for correct,color,label in [(True,'#087f78','Correct'),(False,'#b34d28','Incorrect')]:
            q=frame.loc[frame.correct==correct,'confidence'].to_numpy()
            if len(q):
                shares=np.bincount(np.minimum((q*10).astype(int),9),minlength=10)/len(q)
                ax.stairs(shares,np.linspace(0,1,11),color=color,linewidth=1.8,label=f'{label} (n={len(q)})')
        ax.set(title=title,xlim=(0,1),ylim=(0,1),xlabel='Reported correctness probability',ylabel='Share within outcome group');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8,loc='upper left',frameon=False);ax.grid(alpha=.13)
    for ext in ('svg','png','pdf'):fig.savefig(directory/f'opus_probability_outcomes.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def export_opus_calibration(out, streams, calibrators, colors, eps):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.special import logit
    from confidence_experiment.metrics import wilson
    keys=('opus_same','opus_correctness')
    for kind in ('reliability','mappings'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,key in zip(axes,keys):
            if kind=='reliability':
                for variant in ('raw','logistic_score','logistic_logit','isotonic'):
                    bins=pd.read_csv(out/f'{key}.{variant}.reliability.csv')
                    ci=np.array([wilson(int(round(r.accuracy*r.n)),int(r.n)) for r in bins.itertuples()])
                    ax.errorbar(bins.confidence,bins.accuracy,yerr=np.maximum(0,np.stack([bins.accuracy-ci[:,0],ci[:,1]-bins.accuracy])),fmt='none',ecolor=colors[variant],alpha=.25,linewidth=.8)
                    ax.scatter(bins.confidence,bins.accuracy,s=20+np.sqrt(bins.n)*3,label=variant,color=colors[variant],alpha=.8)
                ax.plot([0,1],[0,1],color='#aaa',linestyle='--',linewidth=1)
                ax.set(xlabel='Mean predicted correctness',ylabel='Observed correctness')
            else:
                fit=calibrators[key];grid=np.linspace(0,1,1001)
                ax.plot(grid,grid,'--',color=colors['raw'],label='identity')
                ax.plot(grid,fit['logistic_score'].predict(grid),color=colors['logistic_score'],label='logistic on score')
                ax.plot(grid,fit['logistic_logit'].predict_proba(logit(np.clip(grid,eps,1-eps)).reshape(-1,1))[:,1],color=colors['logistic_logit'],label='logistic on log-odds')
                ax.plot(grid,fit['isotonic'].predict(grid),color=colors['isotonic'],label='isotonic')
                ax.set(xlabel='Original correctness score',ylabel='Calibrated correctness probability')
            ax.set(title=streams[key]['title'],xlim=(-.02,1.02),ylim=(-.02,1.02));ax.legend(fontsize=8)
        for ext in ('svg','png','pdf'):fig.savefig(out/f'opus_{kind}.{ext}',dpi=160)
        plt.close(fig)


def replace_paragraph(markdown, prefix, replacement):
    blocks=re.split(r'\n\s*\n',markdown.strip())
    matches=[i for i,b in enumerate(blocks) if b.startswith(prefix)]
    assert len(matches)==1,prefix
    blocks[matches[0]]=replacement
    return '\n\n'.join(blocks)+'\n'


def update_cost_prose(markdown, rows, frontiers, components):
    by_id={r['id']:r for r in rows}
    hosted=[r for r in rows if r['hosted']]
    markdown=replace_paragraph(markdown,'The table uses temperature',
        f"The table uses temperature for TF-IDF and embedding LR, raw embedding MLP, logistic on log-odds for OpenAI, Haiku native and all verifiers, and logistic on score for Sonnet and Opus native and all learned pipelines. All {len(rows)} variants enter the frontier calculation; these representative fixed choices are not assertions that each is the best calibrator. Local inference remains unpriced. Sonnet and Opus each reuse three synchronous smoke-test messages per call stage; their recorded costs are slightly above all-batch estimates. The standard-rate proxy reprices identical recorded tokens and does not measure synchronous quality or latency. Opus token costs include its adaptive reasoning.")
    markdown=markdown.replace("Sonnet’s were $2/$10.","Sonnet’s were $2/$10, and Opus’s were $4/$20.")
    def points(metric):
        selected={}
        for name in frontiers['hosted_api_fee_only'][metric]:
            r=by_id[name];coord=(r['api_usd_per_1000'],r[metric])
            if coord not in selected:selected[coord]=r
        return sorted(selected.values(),key=lambda r:r['api_usd_per_1000'])
    accuracy=' → '.join(f"{r['label']} (${r['api_usd_per_1000']:.4f}/1,000; {r['accuracy']:.1%})" for r in points('accuracy'))
    markdown=replace_paragraph(markdown,'For **hosted classification accuracy**',
        f"For **hosted classification accuracy**, the empirical frontier is {accuracy}. Verifier and learned correctness variants retain the base classifier’s selected labels, so extra calls cannot improve accuracy. They can earn their cost through better probability estimates or review ranking.")
    table=['| Objective | Frontier pipeline and calibrator | API USD per 1,000 | Development metric |',
           '|---|---|---:|---:|']
    for metric,title in [('brier','Brier error ↓'),('error_auroc','Error AUROC ↑')]:
        for r in points(metric):
            table.append(f"| {title} | {r['label']} · {r['method'].replace('_',' ')} | ${r['api_usd_per_1000']:.4f} | {r[metric]:.5f} |")
    markdown=replace_paragraph(markdown,'For **hosted cost versus Brier error**',
        "For **hosted cost versus Brier error and error ranking**, the updated frontier members below include Opus. Exact equivalent cost–metric coordinates appear once; strictly increasing logistic variants can tie in AUROC. Membership is determined from all tested variants, rather than from the representative cost table. A method absent from one frontier may remain useful for another objective. The joint cost–accuracy–Brier frontier is retained in the evidence file.\n\n"+'\n'.join(table))
    local=by_id['embedding_lr.temperature']
    dominated=[r for r in hosted if local['api_usd_per_1000']<=r['api_usd_per_1000'] and local['accuracy']>=r['accuracy'] and local['brier']<=r['brier'] and local['error_auroc']>=r['error_auroc'] and local['coverage']>=r['coverage']]
    coverage=max(hosted,key=lambda r:r['coverage'])
    markdown=replace_paragraph(markdown,'When local models enter',
        f"When local models enter the **API-fee-only** comparison, temperature-scaled embedding LR is at least as good on accuracy, Brier, error AUROC and exploratory coverage as **{len(dominated)} of {len(hosted)} hosted variants**, with lower API fees. This is provider-fee dominance, not total economic dominance. The highest hosted development coverage under the Wilson rule is **{coverage['coverage']:.1%}**, from {coverage['label']} with {coverage['method'].replace('_',' ')}. These development-selected acceptance results require independent validation.")
    native=by_id['opus_same.logistic_score']
    markdown=markdown.replace('These are sensitivity assumptions, not estimates for this computer.',f"The comparison now also includes Opus native at **${native['api_usd_per_1000']:.4f}/1,000 and {native['accuracy']:.1%} accuracy**. Pairwise cost-and-accuracy dominance additionally requires the embedding accuracy to be at least as high. These are sensitivity assumptions, not estimates for this computer.")
    markdown=replace_paragraph(markdown,'Under the development-selected temperature threshold',
        f"Under the development-selected temperature threshold, embedding LR would send **105 of 1,000 messages to review**. The highest-coverage hosted variant would send **{round(1000*(1-coverage['coverage']))}**, using the same strict Wilson rule. Review cost may outweigh token charges, but these workloads are exploratory development results requiring independent validation. Cached embedding training durations cannot price fresh-message CPU inference or electricity; a fresh inference benchmark is needed to estimate v.")
    return markdown


def integrate_opus(markdown, data, calibration):
    metrics=data['metrics']['opus']; native=metrics['llm_same_raw']; learned=metrics['llm_correctness_raw']
    comparisons=data['comparisons']; gain=comparisons['opus_original_minus_learned_brier']
    gap=comparisons['embedding_minus_opus_accuracy']
    coverage=max(calibration['results'][k][m]['acceptance']['coverage'] for k in calibration['results'] if k.startswith('opus') for m in calibration['results'][k])
    gain_pct=(native['brier']-learned['brier'])/native['brier']
    def interval(row,scale=1):return f"[{row['ci95'][0]*scale:+.4f}, {row['ci95'][1]*scale:+.4f}]"
    markdown=replace_paragraph(markdown,'A frozen embedding model',
        f"A frozen embedding model with a linear classifier correctly classified **971 of 1,000 banking queries**. Opus 5.5 classified **{1000-native['errors']} ({native['accuracy']:.1%})**, Sonnet 5.5 classified 830, Haiku 5.5 classified 757, and OpenAI Decisions classified 733. These comparisons combine supervised task-specific fitting for local models with zero-shot intent-name prompts for the LLMs; they do not isolate model capability at equal training budgets.")
    markdown=replace_paragraph(markdown,'Learning correctness was useful',
        f"Keeping predictions fixed, learned correctness reduced Brier error by **59.0% for Haiku and 55.7% for Sonnet** before final calibration. For Opus, the relative reduction was **{gain_pct:.1%}**, with an absolute original-minus-learned difference of **{gain['estimate']:+.4f}** and paired 95% interval **{interval(gain)}**. Calibration results vary by confidence source; greater flexibility does not ensure better probabilities.")
    markdown=replace_paragraph(markdown,'The calibrated embedding classifier could accept',
        f"The calibrated embedding classifier could accept **895 development predictions with three errors** under the experiment’s threshold-selection rule. The best observed Opus coverage across the tested calibrators was **{coverage:.1%}**; Haiku and Sonnet had zero qualifying coverage. These are exploratory development findings, with thresholds selected on these outcomes; the official test set remains unevaluated.")
    markdown=markdown.replace('Train separate Haiku and Sonnet correctness estimators','Train separate Haiku Sonnet and Opus correctness estimators')
    markdown=markdown.replace('The Haiku and Sonnet correctness estimators','The Haiku, Sonnet and Opus correctness estimators')
    markdown=markdown.replace('Both Claude models show why a strong AUROC','Haiku and Sonnet show why a strong AUROC')
    blocks=re.split(r'\n\s*\n',markdown.strip())
    for i,b in enumerate(blocks):
        if b.startswith('| Classifier |'):
            lines=b.splitlines();line=f"| Anthropic Opus 5.5 | {native['accuracy']:.1%} | {native['macro_f1']:.3f} | {native['errors']} |"
            lines.insert(next(j for j,s in enumerate(lines) if 'Anthropic Sonnet' in s),line);blocks[i]='\n'.join(lines)
        elif b.startswith('| Candidate |'):
            best=max((calibration['results'][k][m]['acceptance'] for k in calibration['results'] if k.startswith('opus') for m in calibration['results'][k]),key=lambda a:a['coverage'])
            risk=f"{best['errors']/best['accepted']:.2%}" if best['accepted'] else '—'
            blocks[i]=b+f"\n| Opus, largest observed qualifying set | {best['accepted']} | {best['errors'] if best['accepted'] else '—'} | {risk} | {best['coverage']:.1%} |"
    markdown='\n\n'.join(blocks)+'\n'
    markdown=replace_paragraph(markdown,'The embedding classifier led Sonnet',
        f"The embedding LR minus Opus accuracy difference was **{gap['estimate']*100:+.1f} percentage points**, with a paired 95% interval **{interval(gap,100)} points**. Its lead over Sonnet was 14.1 points, over Haiku 21.4 points and over OpenAI 23.8 points. These comparisons condition on the fitted models, available task labels, supported reasoning settings and the implemented prompts.")
    rows=[]
    for title,key in [('Original confidence','llm_same_raw'),('Original plus logistic on score','llm_same_sigmoid'),('Separate verifier','llm_verifier_raw'),('Verifier plus logistic on score','llm_verifier_sigmoid'),('Learned correctness','llm_correctness_raw'),('Learned plus logistic on score','llm_correctness_sigmoid')]:
        m=metrics[key];rows.append(f"| {title} | {m['brier']:.5f} | {m['binary_log_loss']:.5f} | {m['error_auroc']:.3f} |")
    result='supports a reduction' if gain['ci95'][0]>0 else 'supports an increase' if gain['ci95'][1]<0 else 'leaves the direction uncertain'
    calibration_notes=[]
    for suffix,title in [('same','native confidence'),('correctness','learned correctness')]:
        key='opus_'+suffix; variants=calibration['results'][key]
        best_method=min(variants,key=lambda name:variants[name]['brier'])
        comparison=calibration['comparisons'][key]['isotonic_minus_logistic_score']
        low,high=comparison['paired_bootstrap_ci95']
        calibration_notes.append(f"For Opus {title}, **{best_method.replace('_',' ')}** had the lowest development Brier point estimate, **{variants[best_method]['brier']:.5f}**. Isotonic minus logistic-on-score Brier was **{comparison['brier_difference']:+.5f}**, with a conditional paired 95% interval **[{low:+.5f}, {high:+.5f}]**. Positive differences favor logistic on score. These post hoc rankings do not validate choosing a calibrator on development outcomes.")
    section=f"""## Opus extends the comparison

Opus correctly classified **{1000-native['errors']} of 1,000 messages ({native['accuracy']:.1%})**. Its paired accuracy change from Sonnet was **{comparisons['opus_minus_sonnet_accuracy']['estimate']*100:+.1f} percentage points**, with a 95% interval **{interval(comparisons['opus_minus_sonnet_accuracy'],100)} points**. Against the supervised TF-IDF baseline, the difference was **{comparisons['opus_minus_tfidf_accuracy']['estimate']*100:+.1f} points**, interval **{interval(comparisons['opus_minus_tfidf_accuracy'],100)} points**; an interval spanning zero leaves superiority unresolved.

The same data partitions, intent-name prompts, structured JSON schemas, TF-IDF support and confidence features were retained. Opus used its required adaptive thinking at the default medium effort and a 2,048-token output ceiling; Sonnet and Haiku used 256 tokens and their previously recorded thinking settings. This is a comparison of usable configurations, not model identity alone. [Anthropic Opus documentation](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5)

| Opus confidence method | Brier error ↓ | Binary log loss ↓ | Error AUROC ↑ |
|---|---:|---:|---:|
"""+'\n'.join(rows)+f"""

With Opus labels fixed, learned correctness changed Brier error from **{native['brier']:.5f} to {learned['brier']:.5f}**. The paired original-minus-learned interval **{interval(gain)}** {result} on this sample. The raw verifier comparison had an original-minus-verifier Brier difference of **{comparisons['opus_original_minus_verifier_brier']['estimate']:+.5f}**, interval **{interval(comparisons['opus_original_minus_verifier_brier'])}**. Higher absolute confidence or a stronger classifier does not itself establish better calibration; compare these paired within-Opus changes.

Opus scalar calibration used the same 1,000 calibration outcomes, with logistic regression on score, logistic regression on log-odds and isotonic regression evaluated on the separate 1,000 development outcomes. Its best observed qualifying coverage was **{coverage:.1%}**, after inspecting the tested score variants. This selection is exploratory and requires independent validation; the risk curves and probability-bin counts show where the evidence is concentrated.
"""+'\n\n'.join(calibration_notes)+'\n'
    markdown=markdown.replace('## Better error ranking did not guarantee low-risk automation',section.strip()+'\n\n## Better error ranking did not guarantee low-risk automation',1)
    markdown=replace_paragraph(markdown,'Early Haiku requests',
        "Early Haiku requests encountered truncation with adaptive thinking and a 256-token output limit; its comparison was restarted uniformly with thinking disabled. Sonnet retained 256 tokens with between_tools mode and no tools. Opus required adaptive thinking and used a 2,048-token ceiling, checked before batch submission. Reasoning tokens are included in its observed output charges. These settings constrain attribution of model differences. Historical and discarded calls remain in the ledgers.")
    markdown=replace_paragraph(markdown,'Recorded API spending',
        f"Recorded API spending was approximately **${data['spending']['combined_usd']:.2f}**, including **${data['spending']['ledgers'][-1]['usd']:.2f} for Opus** and $22.46 for Sonnet. These token-based ledger estimates include exploratory and discarded calls, exclude local compute, and are not invoices. Logistic and isotonic comparisons reused saved outputs and made no additional API calls. Batch waiting makes elapsed run duration unsuitable for a clean provider speed comparison.")
    markdown=markdown.replace('the paired Sonnet replication, and the 48 score variants from 11 confidence sources','the paired Sonnet and Opus replications, and the 60 score variants from 14 confidence sources')
    markdown=markdown.replace('runs/anthropic-sonnet-5-5, runs/openai-decisions-v2','runs/anthropic-sonnet-5-5, runs/anthropic-opus-5-5, runs/openai-decisions-v2')
    best_llm=max([(data['metrics'][f]['llm_same_raw']['accuracy'],f) for f in ('haiku','sonnet','opus')])[1]
    markdown=replace_paragraph(markdown,'**Use learned correctness',
        f"**Judge learned correctness within each retained LLM classifier.** {best_llm.title()} had the highest observed LLM classification accuracy, but review ranking, probability quality, accepted coverage and API cost remain separate objectives. Remove TF-IDF support, blind agreement and verifier features in turn to identify which information justifies extra calls and labels. Compare that system against using the supervised classifier directly.")
    return markdown
