"""Audit and compare frozen Opus development outputs; no API or test calls."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from confidence_experiment.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/banking77-findings'
RUNS = {'haiku':'anthropic-batch-v2', 'sonnet':'anthropic-sonnet-5-5',
        'opus':'anthropic-opus-5-5', 'embedding':'embeddings'}
FAMILIES = ('llm_same_raw', 'llm_verifier_raw', 'llm_correctness_raw')


def main():
    hashes = {}
    def track(p):
        p = ROOT / p
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    for evidence in ('analysis.json','sonnet-comparison.json'):
        original = read_json(track(OUT / evidence))
        for name, expected in original['source_sha256'].items():
            assert hashlib.sha256(track(name).read_bytes()).hexdigest() == expected, name
    metrics, metadata, frames, reference = {}, {}, {}, None
    for family, run in RUNS.items():
        folder = ROOT/'runs'/run
        metrics[family] = read_json(track(folder/'development/metrics.json'))
        metadata[family] = read_json(track(folder/'development/metadata.json'))
        for name, expected in read_json(track(folder/'frozen_hashes.json')).items():
            assert hashlib.sha256(track(folder/name).read_bytes()).hexdigest() == expected, name
        for method in (('embedding_lr_temperature',) if family=='embedding' else FAMILIES):
            f = pd.read_csv(track(folder/'development'/f'{method}.predictions.csv'),float_precision='round_trip').sort_values('id').reset_index(drop=True)
            assert len(f)==1000 and f.id.is_unique
            assert np.array_equal(f.correct, f.truth==f.prediction)
            if reference is None: reference=f[['id','truth']]
            else: assert f[['id','truth']].equals(reference)
            frames[f'{family}:{method}']=f
        if family!='embedding':
            assert all(frames[f'{family}:{m}'].prediction.equals(frames[f'{family}:{FAMILIES[0]}'].prediction) for m in FAMILIES)
    tfidf=pd.read_csv(track('runs/anthropic-batch-v2/development/tfidf_temperature.predictions.csv'),float_precision='round_trip').sort_values('id').reset_index(drop=True)
    assert tfidf[['id','truth']].equals(reference) and np.array_equal(tfidf.correct,tfidf.truth==tfidf.prediction)
    frames['tfidf:tfidf_temperature']=tfidf
    metrics['tfidf']=metrics['haiku']
    h,o=metadata['haiku']['config'],metadata['opus']['config']
    access=read_json(track('runs/anthropic-opus-5-5/model_access.json'))
    smoke=read_json(track('runs/anthropic-opus-5-5/smoke.json'))
    assert access['id']=='claude-opus-5-5' and smoke['examples']==3 and smoke['thinking']=='adaptive'
    for name in ('seed','data_dir','methods','max_error','threshold_rule','supervised'):
        assert h[name]==o[name],name
    for name in ('transport','batch_size'): assert h['llm'][name]==o['llm'][name],name
    assert o['llm']['model']==o['llm']['verifier_model']=='claude-opus-5-5'
    assert o['llm']['thinking']=='adaptive' and o['llm']['max_output_tokens']==2048
    assert all(metadata[f]['dataset']==metadata['opus']['dataset'] for f in ('haiku','sonnet'))
    v0,c0=joblib.load(track('runs/anthropic-batch-v2/tfidf.joblib'))
    for family in ('sonnet','opus'):
        v,c=joblib.load(track(f'runs/{RUNS[family]}/tfidf.joblib'))
        assert v.vocabulary_==v0.vocabulary_ and np.allclose(v.idf_,v0.idf_,rtol=0,atol=1e-10)
        assert np.array_equal(c.classes_,c0.classes_)
        assert np.allclose(c.coef_,c0.coef_,rtol=0,atol=1e-10)
        assert np.allclose(c.intercept_,c0.intercept_,rtol=0,atol=1e-10)
    boot={k:{m:[] for m in ('accuracy','brier','error_auroc')} for k in frames}
    rng=np.random.default_rng(20261009)
    for start in range(0,5000,250):
        ix=rng.integers(0,1000,size=(250,1000))
        for k,f in frames.items():
            y=f.correct.to_numpy(dtype=float)[ix];q=f.confidence.to_numpy()[ix];err=1-y
            n=err.sum(axis=1);ranks=rankdata(1-q,axis=1,method='average')
            numerator=(ranks*err).sum(axis=1)-n*(n+1)/2;denominator=n*(1000-n)
            boot[k]['accuracy'].append(y.mean(axis=1))
            boot[k]['brier'].append(((q-y)**2).mean(axis=1))
            boot[k]['error_auroc'].append(np.divide(numerator,denominator,out=np.full(250,np.nan),where=denominator!=0))
    boot={k:{m:np.concatenate(v) for m,v in ms.items()} for k,ms in boot.items()}
    estimates={}
    for k,ms in boot.items():
        family,method=k.split(':')
        estimates[k]={m:{'estimate':metrics[family][method][m], 'ci95':np.nanquantile(v,[.025,.975]).tolist()} for m,v in ms.items()}
    pairs=[('opus_minus_sonnet_accuracy','accuracy','opus:llm_same_raw','sonnet:llm_same_raw'),
           ('opus_minus_tfidf_accuracy','accuracy','opus:llm_same_raw','tfidf:tfidf_temperature'),
           ('opus_minus_haiku_accuracy','accuracy','opus:llm_same_raw','haiku:llm_same_raw'),
           ('embedding_minus_opus_accuracy','accuracy','embedding:embedding_lr_temperature','opus:llm_same_raw'),
           ('opus_original_minus_learned_brier','brier','opus:llm_same_raw','opus:llm_correctness_raw'),
           ('opus_original_minus_verifier_brier','brier','opus:llm_same_raw','opus:llm_verifier_raw'),
           ('opus_learned_minus_original_auroc','error_auroc','opus:llm_correctness_raw','opus:llm_same_raw')]
    comparisons={name:{'metric':m,'first':a,'second':b,'estimate':estimates[a][m]['estimate']-estimates[b][m]['estimate'],
                      'ci95':np.nanquantile(boot[a][m]-boot[b][m],[.025,.975]).tolist()} for name,m,a,b in pairs}
    ledgers=[]
    for name in ('data/anthropic_cache/spending.json','data/anthropic_spending/spending.json','data/anthropic_sonnet_cache/spending.json','data/anthropic_opus_cache/spending.json'):
        ledger=read_json(track(name));assert not ledger.get('reservations'),name
        ledgers.append({'path':name,'usd':ledger['usd'],'reservations_usd':0})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'role':'development','n':1000,
            'bootstrap':{'seed':20261009,'resamples':5000,'scope':'Paired conditional percentile intervals; no retraining or selection adjustment'},
            'config_match_verified':True,'tfidf_support_match_verified':True,
            'configuration_difference':{'opus_thinking':'adaptive','opus_max_output_tokens':2048,
                'sonnet_thinking':'between_tools','haiku_thinking':'disabled','other_max_output_tokens':256,
                'interpretation':'Configuration comparison includes model and supported reasoning/token settings'},
            'metrics':metrics,'estimates':estimates,'comparisons':comparisons,
            'spending':{'ledgers':ledgers,'combined_usd':sum(l['usd'] for l in ledgers)},
            'source_sha256':hashes,'official_test_evaluated':False}
    write_json(OUT/'opus-comparison.json',result)
    print({k:v['estimate'] for k,v in comparisons.items()},flush=True)


if __name__=='__main__':main()
