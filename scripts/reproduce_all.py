"""Run all offline numerical reproductions without API requests."""
from pathlib import Path
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
    env=os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    for name in ['verify_utils.py', 'validate_saved_texts.py','fit_itsa.py','recompute_validation.py','recompute_simulations.py','build_tables.py','make_figures.py']:
        print('\n=== '+name+' ===',flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)
    from verify_utils import verify_inputs,write_json
    inputs=verify_inputs()
    text_checks=json.loads((ROOT/'outputs/saved_text_checks.json').read_text())
    report={'saved_text_corpus_checks':text_checks,'status':'offline_numerical_checks_passed','python':sys.version.split()[0],'inputs':inputs,'generation_API_calls':0,'new_embedding_model_calls':0,'figure_pdfs':len(list((ROOT/'outputs/figures').glob('*.pdf'))),'table_csvs':len(list((ROOT/'outputs/tables').glob('*.csv'))),'checks':{}}
    for name in ['simulation_checks.json','validation_metrics_checks.json','empirical_table_checks.json']:
        report['checks'][name]=json.loads((ROOT/'outputs'/name).read_text())
    if report['figure_pdfs']!=11:raise AssertionError('Expected 11 separate numerical figure panels')
    write_json(ROOT/'outputs/verification_report.json',report)
    print('\nOffline numerical reproduction completed successfully. No API requests were made.')
if __name__=='__main__':main()
