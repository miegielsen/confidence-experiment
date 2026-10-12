"""Wait for the Opus worker, then audit, calibrate and rebuild local reports.

Never submits requests or retries failed API work. Page synchronization uses the
interactive Pages tools after this worker has produced verified local artifacts.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from confidence_experiment.io import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/anthropic-opus-5-5'
OUT=ROOT/'reports/banking77-findings'


def main():
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    lock=RUN/'.comparison.lock'
    try:
        with lock.open('x') as h:h.write(str(os.getpid()))
    except FileExistsError:
        print('An Opus finisher already owns this run.',flush=True);return
    status={'state':'waiting_for_results','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'official_test_evaluated':False}
    def update(state,**extra):
        status.update(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(RUN/'comparison_status.json',status)
    try:
        update('waiting_for_results')
        while True:
            supervisor=read_json(RUN/'background_status.json')
            if supervisor['state'] in ('failed','cancelled','canceled'):
                raise RuntimeError('Opus worker '+supervisor['state']+'; no requests retried')
            if supervisor['state']=='complete':break
            time.sleep(15)
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        for state,script in [('paired_analysis','analyze_opus_comparison.py'),('calibration','compare_calibrators.py'),('cost_frontiers','analyze_cost_frontiers.py'),('report_build','build_findings_report.py')]:
            update(state)
            subprocess.run([sys.executable,str(ROOT/'scripts'/script)],env=env,check=True)
        update('pdf_export')
        chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
        if not chrome.exists():raise FileNotFoundError(chrome)
        profile=Path(os.environ['LOCALAPPDATA'])/'Temp/confidence-opus-report-preview'
        subprocess.run([str(chrome),'--headless=new','--disable-gpu','--no-first-run','--disable-extensions','--disable-background-networking','--no-pdf-header-footer',f'--user-data-dir={profile}',f'--print-to-pdf={OUT/"report.pdf"}',(OUT/'report.html').as_uri()],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        evidence=read_json(OUT/'consolidated-evidence.json')
        assert evidence['opus_included'] and evidence['calibration_variants']==60
        for name,sha in evidence['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
        pdf=(OUT/'report.pdf').read_bytes();assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-100:]
        publication=read_json(OUT/'publication.json')
        publication.update(opus_local_results_included=True,opus_page_sync_pending=True,pdf_updated=True,html_updated=True,
                           consolidated_source_hashes_verified=len(evidence['source_sha256']),pdf_pages=len(re.findall(rb'/Type\s*/Page\b',pdf)))
        for ext in ('pdf','html'):publication[ext+'_sha256']=hashlib.sha256((OUT/f'report.{ext}').read_bytes()).hexdigest()
        write_json(OUT/'publication.json',publication)
        update('complete',report=str(OUT/'report.html'),pdf=str(OUT/'report.pdf'),page_sync_pending=True)
        print('Opus local reports complete; Page sync pending.',flush=True)
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
