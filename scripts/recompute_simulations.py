"""Recompute semantic comparisons from frozen, indexed embedding arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import m_methods as m
import p1_methods as p1
import p3_methods as p3
from verify_utils import ROOT, array, compare_frames, write_json
DATA=ROOT/'data/semantic';OUT=ROOT/'outputs/tables';DER=ROOT/'outputs/derived/simulation'

def compute_m() -> dict:
    si=pd.read_csv(DATA/'m_generated_index.csv');ri=pd.read_csv(DATA/'m_reference_index.csv')
    se=array(DATA/'m_generated_embeddings.npy',2000);re=array(DATA/'m_reference_embeddings.npy',5426)
    if si.embedding_row.tolist()!=list(range(2000)):raise AssertionError('M row order changed')
    _,horizon,arch=m._semantic_scores(si,se,ri,re)
    check=compare_frames(arch,pd.read_csv(ROOT/'reference/architecture_text_fidelity_summary_v2b.csv'),['architecture_id','top_k'])
    check['horizon']=compare_frames(horizon,pd.read_csv(ROOT/'reference/semantic_horizon_summary_v2b.csv'),['architecture_id','month','top_k'])
    arch.to_csv(OUT/'table13_architecture_fidelity.csv',index=False)
    horizon.to_csv(OUT/'m_architecture_by_horizon.csv',index=False)
    return check

def compute_p1() -> tuple[dict,pd.DataFrame,np.ndarray]:
    index=pd.read_csv(DATA/'p1_index_N200.csv');emb=array(DATA/'p1_embeddings_N200.npy',2000)
    if index.embedding_row.tolist()!=list(range(2000)):raise AssertionError('P1 row order changed')
    all_metrics={}
    for n in [50,100,200]:
        flag=np.ones(len(index),dtype=bool) if n==200 else index[f'in_N{n}'].to_numpy(bool)
        work=index.loc[flag];assert len(work)==n*10
        keys=['profile_sim_id','policy_horizon_months','simulation_month']
        a=work.loc[work.scenario_id.eq('P0'),keys+['embedding_row','analysis_weight']]
        b=work.loc[work.scenario_id.eq('P_ABC'),keys+['embedding_row','analysis_weight']]
        pairs=a.merge(b,on=keys,suffixes=('_p0','_abc'),validate='one_to_one')
        assert len(pairs)==n*5
        np.testing.assert_allclose(pairs.analysis_weight_p0,pairs.analysis_weight_abc,rtol=0,atol=1e-12)
        e0=emb[pairs.embedding_row_p0.to_numpy(int)];e1=emb[pairs.embedding_row_abc.to_numpy(int)]
        pooled,horizon=p1.metrics_for_n({'pairs':pairs,'delta':e1-e0,'displacement':1-np.sum(e0*e1,axis=1),'weights':pairs.analysis_weight_p0.to_numpy(float)},n)
        all_metrics[n]={'pooled':pooled,'horizon':horizon}
    result=pd.DataFrame([p1.compare_n(50,100,all_metrics),p1.compare_n(100,200,all_metrics)])
    check=compare_frames(result,pd.read_csv(ROOT/'reference/p1_convergence.csv'),['small_N','large_N'])
    if not result.stable.all():raise AssertionError('Frozen convergence tolerances did not pass')
    result.to_csv(OUT/'table14_profile_convergence.csv',index=False)
    return check,index,emb

def compute_p3(p1_index:pd.DataFrame,p1_emb:np.ndarray) -> dict:
    ix=pd.read_csv(DATA/'p3_index.csv');emb=array(DATA/'p3_embeddings.npy',4500)
    if ix.embedding_row.tolist()!=list(range(4500)):raise AssertionError('P3 row order changed')
    if ix.duplicated(['scenario_id','profile_sim_id','policy_horizon_months']).any():raise AssertionError('Duplicate P3 logical key')
    shared=ix.merge(p1_index,on='request_row_id',suffixes=('_p3','_p1'),validate='one_to_one')
    if len(shared)!=1000 or not shared.request_hash_p3.eq(shared.request_hash_p1).all():raise AssertionError('P1/P3 reused requests differ')
    np.testing.assert_array_equal(emb[shared.embedding_row_p3.to_numpy(int)],p1_emb[shared.embedding_row_p1.to_numpy(int)])
    np.testing.assert_allclose(shared.analysis_weight_p3,shared.analysis_weight_p1,atol=1e-12,rtol=0)
    info=p3.build_embedding_cube(ix,emb)
    signs=p3.make_sign_matrix(p3.N_PROFILES,p3.N_PERMUTATIONS,p3.PERMUTATION_SEED)
    contract=json.loads((DATA/'analysis_parameters.json').read_text())
    if hashlib.sha256(signs.tobytes()).hexdigest().upper()!=contract['permutation']['sign_matrix_sha256']:raise AssertionError('Permutation sign matrix differs')
    result=p3.evaluate_all(info,signs)
    check={}
    for key in ['main','horizon']:
        keys=['contrast_id']+(['horizon_months'] if key=='horizon' else [])
        reference=pd.read_csv(ROOT/f'reference/p3_{key}.csv')
        q_col='BH_q_primary_8' if key=='main' else 'BH_q_secondary_40'
        reject_col='reject_BH_0_05_primary' if key=='main' else 'reject_BH_0_05_secondary'
        # Deterministic quantities remain tightly checked. Monte-Carlo p-values are
        # checked separately below because a cross-platform BLAS/einsum rounding tie
        # can move a single permutation statistic across the observed threshold.
        numeric_cols=[c for c in reference.select_dtypes(include='number').columns
                      if c not in keys and c not in {'raw_p',q_col}]
        check[key]=compare_frames(result[key],reference,keys,columns=numeric_cols)

        merged=result[key].merge(reference,on=keys,how='outer',suffixes=('_actual','_expected'),
                                 indicator=True,validate='one_to_one')
        if not merged['_merge'].eq('both').all():
            raise AssertionError('P3 randomization comparison key sets differ')
        scale=p3.N_PERMUTATIONS+1
        # p=(ge+1)/(B+1), so converting back to ge makes the intended Monte-Carlo
        # resolution explicit. One-count tolerance covers numerical ties only; it
        # does not permit a different permutation seed or a materially different test.
        ge_actual=np.rint(pd.to_numeric(merged['raw_p_actual']).to_numpy(float)*scale-1).astype(int)
        ge_expected=np.rint(pd.to_numeric(merged['raw_p_expected']).to_numpy(float)*scale-1).astype(int)
        count_diff=np.abs(ge_actual-ge_expected)
        if np.max(count_diff)>1:
            raise AssertionError(f'P3 {key} permutation exceedance count differs by more than 1')

        qa=pd.to_numeric(merged[q_col+'_actual']).to_numpy(float)
        qe=pd.to_numeric(merged[q_col+'_expected']).to_numpy(float)
        finite=np.isfinite(qa)&np.isfinite(qe)
        n_tests=8 if key=='main' else 40
        q_tolerance=n_tests/scale + 1e-12
        max_q_diff=float(np.max(np.abs(qa[finite]-qe[finite]))) if finite.any() else 0.0
        if max_q_diff>q_tolerance:
            raise AssertionError(f'P3 {key} BH q difference exceeds Monte-Carlo resolution bound')

        ra=merged[reject_col+'_actual'].astype(bool).to_numpy()
        re=merged[reject_col+'_expected'].astype(bool).to_numpy()
        if not np.array_equal(ra,re):
            raise AssertionError(f'P3 {key} BH rejection decisions differ')
        check[key]['randomization']={
            'permutations':p3.N_PERMUTATIONS,
            'max_exceedance_count_difference':int(np.max(count_diff)),
            'max_raw_p_difference':float(np.max(np.abs(
                pd.to_numeric(merged['raw_p_actual']).to_numpy(float)-
                pd.to_numeric(merged['raw_p_expected']).to_numpy(float)))),
            'max_BH_q_difference':max_q_diff,
            'BH_rejection_decisions_identical':True,
            'cross_platform_tolerance':'at most one permutation exceedance count',
        }
        result[key].to_csv(OUT/f'p3_{key}_full.csv',index=False)
    check['folds']=compare_frames(result['folds'],pd.read_csv(DATA/'p3_folds.csv'),['profile_sim_id'],atol=1e-12,rtol=0)
    result['folds'].to_csv(OUT/'p3_folds_recomputed.csv',index=False)
    result['component'].to_csv(OUT/'p3_components_full.csv',index=False)
    main=result['main']; names={
        'information':'Policy information','state_given_information':'Added implementation conditions',
        'sufficiency_A':'Homework regulation only','sufficiency_B':'Tutoring regulation only','sufficiency_C':'After-school services only',
        'necessity_A':'Homework conditions held unchanged','necessity_B':'Tutoring conditions held unchanged','necessity_C':'After-school conditions held unchanged'}
    compact=main.loc[main.contrast_id.isin(names),['contrast_id','omnibus_temporal_magnitude','relative_omnibus_magnitude_to_full','temporal_alignment_with_full_direction','crossfit_directional_contribution_to_full','BH_q_primary_8']].rename(columns={'omnibus_temporal_magnitude':'shift','relative_omnibus_magnitude_to_full':'relative','temporal_alignment_with_full_direction':'alignment','crossfit_directional_contribution_to_full':'projection','BH_q_primary_8':'bh_q'})
    compact.insert(1,'contrast',compact.contrast_id.map(names));compact.to_csv(OUT/'table3_semantic_contrasts.csv',index=False)
    series={'full_policy':'Full policy','information':'Policy information','state_given_information':'Implementation state'}
    h=result['horizon'];h=h.loc[h.contrast_id.isin(series)].copy();h['series']=h.contrast_id.map(series)
    h[['series','horizon_months','effect_magnitude','raw_p','BH_q_secondary_40']].to_csv(DER/'p3_information_state_by_horizon.csv',index=False)
    rows=[]
    for component in 'ABC':
        for prefix,measure1,measure2 in [('sufficiency','Standalone magnitude','Standalone projection'),('necessity','State-ablation magnitude','State-ablation projection')]:
            r=main.loc[main.contrast_id.eq(prefix+'_'+component)].iloc[0]
            rows += [{'component':component,'measure':measure1,'value':r.relative_omnibus_magnitude_to_full},{'component':component,'measure':measure2,'value':r.crossfit_directional_contribution_to_full}]
    pd.DataFrame(rows).to_csv(DER/'p3_component_ratios_long.csv',index=False)
    check['shared_p1_embedding_rows_checked']=len(shared)
    return check

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);DER.mkdir(parents=True,exist_ok=True)
    mc=compute_m();pc,ix,emb=compute_p1();qc=compute_p3(ix,emb)
    write_json(ROOT/'outputs/simulation_checks.json',{'M':mc,'P1':pc,'P3':qc,'new_generation_api_calls':0,'new_embedding_model_calls':0})
    print('Tables 3, 13, 14 and Figure 4 numerical inputs recomputed and checked.')
if __name__=='__main__':main()
