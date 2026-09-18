"""Recompute classifier and relevance-screening performance from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix, hamming_loss
from verify_utils import ROOT, compare_frames, write_json
OUT = ROOT / 'outputs/tables'
EMP = ROOT / 'data/derived/empirical'
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/evaluation/classifier/test_predictions.csv')
    labels = json.loads((ROOT / 'data/evaluation/classifier/labels.json').read_text(encoding='utf-8'))
    gold = data[[f'label_{i}_{label}' for i, label in enumerate(labels)]].to_numpy()
    pred = data[['roberta_pred_' + label for label in labels]].to_numpy()
    prob = data[['roberta_prob_' + label for label in labels]].to_numpy()
    if len(data) != 2250 or data.sample_id.duplicated().any(): raise AssertionError('Invalid classifier test roster')
    if not set(np.unique(gold)) <= {0,1} or not set(np.unique(pred)) <= {0,1}: raise AssertionError('Labels must be binary')
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any(): raise AssertionError('Invalid probabilities')
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, zero_division=0)
    per = pd.DataFrame({'label': labels, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    per.to_csv(OUT / 'table4b_classifier_per_label.csv', index=False)
    overall = {}
    for average in ['micro','macro','weighted']:
        p,r,f,_ = precision_recall_fscore_support(gold, pred, average=average, zero_division=0)
        overall.update({average + '_precision': float(p), average + '_recall': float(r), average + '_f1': float(f)})
    overall.update({'samples_f1': float(f1_score(gold,pred,average='samples')), 'hamming_loss':float(hamming_loss(gold,pred)), 'exact_match_ratio':float(accuracy_score(gold,pred))})
    expected = json.loads((ROOT / 'reference/classifier_metrics.json').read_text())
    for key, value in expected.items(): np.testing.assert_allclose(overall[key],value,rtol=0,atol=1e-12,err_msg=key)
    class_check = compare_frames(per, pd.read_csv(EMP / 'chinese_roberta_wwm_ext_test_per_label_report.csv'), ['label'], atol=1e-12, rtol=0)
    write_json(OUT/'classifier_metrics_recomputed.json', overall)
    records=[]; gold_reference=None
    for path in sorted((ROOT / 'data/evaluation/relevance').glob('*.csv')):
        frame=pd.read_csv(path).sort_values('sample_id').reset_index(drop=True)
        if len(frame)!=1000 or frame.sample_id.duplicated().any(): raise AssertionError('Invalid screening roster')
        if not set(frame.gold_label.unique())<={0,1} or not set(frame.pred_label.unique())<={0,1}: raise AssertionError('Invalid screening labels')
        if gold_reference is None:gold_reference=frame[['sample_id','gold_label']]
        else:pd.testing.assert_frame_equal(gold_reference,frame[['sample_id','gold_label']],check_dtype=False)
        if frame.parsed_ok.isna().any() or not frame.parsed_ok.eq(True).all(): raise AssertionError('Unsuccessful parsed screening responses')
        elapsed=pd.to_numeric(frame.elapsed_sec).to_numpy(float)
        if not np.isfinite(elapsed).all() or (elapsed<=0).any(): raise AssertionError('Invalid saved request timing')
        for token_col in ['prompt_tokens','completion_tokens','total_tokens']:
            values=pd.to_numeric(frame[token_col]).to_numpy(float)
            if not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all(): raise AssertionError('Invalid token counts')
        p,r,f,_=precision_recall_fscore_support(frame.gold_label, frame.pred_label, average='binary',zero_division=0)
        tn,fp,fn,tp=confusion_matrix(frame.gold_label,frame.pred_label,labels=[0,1]).ravel()
        records.append({'model':path.name.replace('_relevance_predictions_1000.csv',''),'n_total':len(frame),'accuracy':accuracy_score(frame.gold_label,frame.pred_label),'precision':p,'recall':r,'f1':f,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'n_valid':len(frame),'parse_failures':0,'avg_time_sec':float(elapsed.mean()),'total_time_sec':float(elapsed.sum()),**{k:int(frame[k].sum()) for k in ['prompt_tokens','completion_tokens','total_tokens']}})
    result=pd.DataFrame(records)
    ref=pd.read_csv(EMP/'relevance_model_configuration_evaluation.csv')
    rel_check=compare_frames(result,ref,['model'],columns=['n_total','accuracy','precision','recall','f1','tn','fp','fn','tp','n_valid','parse_failures','avg_time_sec','total_time_sec','prompt_tokens','completion_tokens','total_tokens'],atol=1e-9,rtol=0)
    # Historical latency is a recorded measurement, not a new API timing experiment.
    result.to_csv(OUT/'table4a_relevance.csv',index=False)
    write_json(ROOT/'outputs/validation_metrics_checks.json',{'classifier':class_check,'relevance':rel_check,'latency_recomputed_from_request_rows':True})
    print('Table 4 accuracy and historical latency recalculated from saved per-request records.')
if __name__=='__main__':main()
